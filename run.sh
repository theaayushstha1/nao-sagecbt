#!/usr/bin/env bash
#
# One-shot dev launcher for Nao-OpenAI-Morgan-Assist.
#
#   1. Validates .env (no PASTE_* placeholders left)
#   2. Detects this Mac's IP on the same subnet as the NAO
#   3. Rsyncs nao/ to the robot
#   4. Kills any stale main.py on the robot (prevents the multi-session bug)
#   5. Starts the local server (Flask by default, uvicorn when USE_WS=1)
#   6. Waits for /health to respond
#   7. Launches a single main.py on the robot with the right env vars
#   8. Tails both logs (Ctrl-C to stop)
#
# Phase 1 (PRD v2) introduces a FastAPI + WebSocket transport at
# server/app_ws.py. Set USE_WS=1 in .env (or run `./run.sh ws`) to boot
# uvicorn instead of Flask. USE_WS=0 / unset keeps the legacy path.
#
# Usage:  ./run.sh                  # all (deploy + server + robot + tail)
#         ./run.sh deploy-only      # just rsync, don't launch
#         ./run.sh server-only      # just start the local server
#         ./run.sh ws               # force USE_WS=1 (FastAPI + WS) for this run
#         ./run.sh stop             # kill local server (Flask or uvicorn) + remote main.py

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# ─────────── helpers ───────────
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { printf "${CYAN}▸${NC} %s\n" "$*"; }
ok()   { printf "${GREEN}✓${NC} %s\n" "$*"; }
warn() { printf "${YELLOW}⚠${NC} %s\n" "$*"; }
die()  { printf "${RED}✗${NC} %s\n" "$*" >&2; exit 1; }

[ -f .env ] || die ".env not found at $PROJECT_ROOT/.env"

# ─────────── python interpreter ───────────
# Bare `python` resolves to whatever is first on PATH — on a machine with
# conda/pyenv that is NOT the project venv, so the server dies at import on
# a missing dep (structlog, fastapi, agents…). Pin the venv explicitly when
# it exists; fall back to PATH python for setups that manage envs another way.
if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    PY="$PROJECT_ROOT/.venv/bin/python"
else
    PY="$(command -v python || command -v python3)"
    [ -n "$PY" ] || die "no python found on PATH and no .venv at $PROJECT_ROOT/.venv"
fi

# Read .env into the current shell. set -a auto-exports each var.
set -a
# shellcheck disable=SC1091
source .env
set +a

# ─────────── ws override (CLI flag forces USE_WS=1 regardless of .env) ───────────
# `./run.sh ws` is shorthand for "boot the FastAPI/WS server for this run."
# We rewrite $1 to `all` after flipping the flag so the rest of the dispatch
# logic stays a single switch. `ws-only` is the WS counterpart of server-only.
WS_OVERRIDE=0
case "${1:-all}" in
    ws)        WS_OVERRIDE=1; set -- all ;;
    ws-only)   WS_OVERRIDE=1; set -- server-only ;;
esac

# ─────────── validate keys ───────────
[ -n "${OPENAI_API_KEY:-}" ] && [[ "${OPENAI_API_KEY}" != PASTE_* ]] \
    || die "OPENAI_API_KEY is not set in .env (still PASTE_OPENAI_KEY_HERE?)"
[ -n "${DEEPGRAM_API_KEY:-}" ] && [[ "${DEEPGRAM_API_KEY}" != PASTE_* ]] \
    || warn "DEEPGRAM_API_KEY missing; server will fall back to Whisper."
[ -n "${NAO_IP:-}" ]            || die "NAO_IP not set in .env"
[ -n "${SERVER_PORT:-}" ]       || SERVER_PORT=5050
[ -n "${NAO_SHARED_SECRET:-}" ] || warn "NAO_SHARED_SECRET empty (open mode)."

# ─────────── ssh transport ───────────
# sshpass exists only to feed NAO_PASSWORD to non-interactive ssh/rsync. When
# key auth is set up (an installed ~/.ssh key + a Host entry), it is redundant
# — plain ssh authenticates silently. Prefer sshpass when both it and a
# password are available, else fall back to bare ssh/rsync so the launcher
# works on machines without Homebrew. BatchMode on the key path turns a
# missing/rejected key into an immediate error instead of a password prompt
# that would hang this script forever.
SSH_OPTS="-o ConnectTimeout=8 -o StrictHostKeyChecking=no"
if command -v sshpass >/dev/null 2>&1 && [ -n "${NAO_PASSWORD:-}" ]; then
    ok "ssh auth: sshpass (NAO_PASSWORD from .env)"
    nao_ssh()   { sshpass -p "$NAO_PASSWORD" ssh $SSH_OPTS "$@"; }
    nao_rsync() { sshpass -p "$NAO_PASSWORD" rsync "$@"; }
else
    ok "ssh auth: ssh key (sshpass not installed)"
    nao_ssh()   { ssh $SSH_OPTS -o BatchMode=yes "$@"; }
    nao_rsync() { rsync "$@"; }
fi

# ─────────── pick transport mode ───────────
# USE_WS=1 in .env (or `./run.sh ws`) -> FastAPI + WebSocket via uvicorn.
# Anything else -> legacy Flask (current main behavior).
USE_WS="${USE_WS:-0}"
if [ "$WS_OVERRIDE" = "1" ]; then
    USE_WS=1
fi
export USE_WS

# WS bind defaults match server/config.py. Honor what's already in .env.
WS_HOST="${WS_HOST:-0.0.0.0}"
WS_PORT="${WS_PORT:-$SERVER_PORT}"
export WS_HOST WS_PORT

if [ "$USE_WS" = "1" ]; then
    SERVER_MODE="ws"
    SERVER_BIND_PORT="$WS_PORT"
    ok "python: $PY"
    ok "transport: FastAPI + WebSocket (uvicorn) on :$WS_PORT  [USE_WS=1]"
else
    SERVER_MODE="flask"
    SERVER_BIND_PORT="$SERVER_PORT"
    ok "transport: Flask (legacy) on :$SERVER_PORT  [USE_WS=0]"
fi

# ─────────── detect this Mac's IP that the robot can reach ───────────
# Pick the first non-loopback IPv4 on the same /24 as NAO_IP, otherwise
# fall back to the default-route interface address.
detect_local_ip() {
    local subnet ip
    subnet="$(echo "$NAO_IP" | awk -F. '{print $1"."$2"."$3"."}')"
    ip="$(ifconfig | awk '/inet /{print $2}' | grep "^$subnet" | head -n1 || true)"
    if [ -z "$ip" ]; then
        ip="$(route -n get default 2>/dev/null | awk '/interface:/{print $2}' \
              | xargs -I{} ipconfig getifaddr {} 2>/dev/null || true)"
    fi
    [ -n "$ip" ] || die "could not detect local IP on subnet $subnet"
    echo "$ip"
}
LOCAL_IP="$(detect_local_ip)"
ok "local IP for NAO callback: $LOCAL_IP"

# ─────────── stop everything cleanly ───────────
# Each run.sh used to spawn fresh `tail -f` + awk processes for log mirroring
# without killing the old ones. Hitting Ctrl+C only killed the ssh tail in the
# foreground; the local server tail in `&` background and any prior runs'
# tails just kept living. After 4 invocations every log line was printed 4
# times. We now sweep them up explicitly.
kill_local_tails() {
    ps -ef \
        | grep -iE "tail -f $PROJECT_ROOT/logs/(server|nao)\.log|tail -f $ROBOT_LOG_REMOTE|awk -v p .*\[server\]|awk -v p .*\[robot\]" \
        | grep -v grep \
        | awk '{print $2}' \
        | xargs -r kill -9 2>/dev/null || true
}

do_stop() {
    # Kill whatever is bound to either the Flask port or the WS port — we
    # don't always know which mode the previous run used, so sweep both.
    log "stopping local server on :$SERVER_PORT (Flask) and :$WS_PORT (WS)"
    lsof -ti ":$SERVER_PORT" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    if [ "$WS_PORT" != "$SERVER_PORT" ]; then
        lsof -ti ":$WS_PORT" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    fi
    # Belt-and-suspenders: any stray uvicorn pointed at server.app_ws.
    log "stopping any uvicorn server.app_ws"
    pkill -f "uvicorn .*server\.app_ws" 2>/dev/null || true
    log "stopping main.py on robot"
    nao_ssh "nao@$NAO_IP" 'pkill -f "python.*main.py" 2>/dev/null; true' || true
    log "killing any stale local log tails"
    kill_local_tails
    ok "stopped"
}

# ─────────── deploy NAO files to robot ───────────
do_deploy() {
    log "deploying nao/ to nao@$NAO_IP:/home/nao/nao_assist/"
    nao_rsync -az --delete \
        --exclude='*.pyc' --exclude='__pycache__' --exclude='nao.log' \
        --exclude='.last_user.json' \
        -e "ssh $SSH_OPTS" \
        "$PROJECT_ROOT/nao/" "nao@$NAO_IP:/home/nao/nao_assist/"
    # Wipe any stale .pyc on the robot — Python 2 prefers cached bytecode
    # over .py source when timestamps look close, which made VAD threshold
    # tweaks silently no-op.
    nao_ssh "nao@$NAO_IP" 'find /home/nao/nao_assist -name "*.pyc" -delete 2>/dev/null; true'
    ok "deploy complete (.pyc cleared)"
}

# ─────────── start Flask server ───────────
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
SERVER_LOG="$LOG_DIR/server.log"
ROBOT_LOG_REMOTE="/home/nao/nao_assist/nao.log"

# Poll /health on the given port until it 200s, or die after ~10 s. Reused
# by both the Flask path and the uvicorn path so they have identical
# semantics — if either app forgets to expose /health, the launcher fails
# loud instead of racing the robot.
wait_for_health() {
    local port="$1" pid="$2" label="$3"
    for i in $(seq 1 25); do
        if curl -sf "http://localhost:$port/health" >/dev/null 2>&1; then
            ok "$label healthy on :$port (pid $pid)"
            return 0
        fi
        sleep 0.4
    done
    die "$label failed to come up — see $SERVER_LOG"
}

start_server() {
    local port pid
    port="$SERVER_BIND_PORT"
    if lsof -ti ":$port" >/dev/null 2>&1; then
        warn "port $port already in use, killing"
        lsof -ti ":$port" | xargs -r kill -9 2>/dev/null || true
        sleep 1
    fi
    if [ "$SERVER_MODE" = "ws" ]; then
        # FastAPI + WebSocket transport (Phase 1 of v2 rework). The app
        # module is still being merged across worktrees — when this command
        # is printed but uvicorn fails, that's the expected gap. The
        # launcher itself is exercised end-to-end up to the import.
        log "starting uvicorn server.app_ws on $WS_HOST:$port (log: $SERVER_LOG)"
        log "cmd: uvicorn server.app_ws:app --host $WS_HOST --port $port --log-level info"
        nohup "$PY" -m uvicorn server.app_ws:app \
            --host "$WS_HOST" --port "$port" --log-level info \
            > "$SERVER_LOG" 2>&1 &
        pid=$!
        echo "$pid" > "$LOG_DIR/server.pid"
        wait_for_health "$port" "$pid" "uvicorn (FastAPI/WS)"
    else
        log "starting Flask on 0.0.0.0:$port (log: $SERVER_LOG)"
        nohup "$PY" -m flask --app server.server run \
            --host 0.0.0.0 --port "$port" \
            > "$SERVER_LOG" 2>&1 &
        pid=$!
        echo "$pid" > "$LOG_DIR/server.pid"
        wait_for_health "$port" "$pid" "Flask"
    fi
}

# ─────────── launch main.py on robot ───────────
start_robot() {
    log "killing any stale main.py on robot"
    nao_ssh "nao@$NAO_IP" 'pkill -f "python.*main.py" 2>/dev/null; sleep 1; true' || true

    log "launching main.py on $NAO_IP (server callback: $LOCAL_IP:$SERVER_BIND_PORT, mode: $SERVER_MODE)"
    # naoqi bindings live at /opt/aldebaran on this NAO image; without
    # this PYTHONPATH the `import qi` at the top of main.py fails.
    #
    # USE_WS is forwarded so the robot-side main.py can pick the WS client
    # vs the legacy HTTP /turn flow without us redeploying. The same value
    # the local run is using is the value the robot uses — no drift.
    nao_ssh "nao@$NAO_IP" "PYTHONPATH=/opt/aldebaran/lib/python2.7/site-packages \
LD_LIBRARY_PATH=/opt/aldebaran/lib:/opt/aldebaran/lib/naoqi \
SERVER_IP='$LOCAL_IP' SERVER_PORT='$SERVER_BIND_PORT' \
NAO_IP='127.0.0.1' NAO_PORT='${NAO_PORT:-9559}' \
USE_WS='$USE_WS' \
NAO_SHARED_SECRET='$NAO_SHARED_SECRET' \
IMAGE_PER_TURN='${IMAGE_PER_TURN:-1}' \
MIC_CHANNEL='${MIC_CHANNEL:-left}' \
ENGAGE_POSTURE='${ENGAGE_POSTURE:-Sit}' \
SPEAKING_GESTURES='${SPEAKING_GESTURES:-1}' \
nohup python -u /home/nao/nao_assist/main.py \
> $ROBOT_LOG_REMOTE 2>&1 </dev/null &"
    sleep 2
    # confirm exactly one process is running
    local count
    count="$(nao_ssh "nao@$NAO_IP" 'ps aux | grep -v grep | grep -c "python.*main.py"' || echo 0)"
    if [ "$count" = "1" ]; then
        ok "main.py running on robot (1 process)"
    else
        warn "expected 1 main.py on robot, found $count — check $ROBOT_LOG_REMOTE"
    fi
}

# ─────────── tail both logs side by side ───────────
# macOS BSD sed crashes on some multi-byte / unicode input ("Assertion failed:
# (advance > 0)"). Robot replies contain smart quotes, em-dashes, and
# transcripts in other languages (Nepali, Spanish), all of which trip BSD
# sed. We use awk instead — POSIX byte-safe, ships everywhere — and fall back
# to plain `cat` if even awk hiccups so a colorize bug never tears down the
# whole session and leaves orphaned server/robot processes.
do_tail() {
    log "killing any stale local log tails"
    kill_local_tails
    log "tailing logs — Ctrl-C to stop, server keeps running"
    log "(set RAW_LOGS=1 in env to see all log lines incl. per-chunk debug)"
    echo
    local server_prefix="${CYAN}[server]${NC} "
    local robot_prefix="${YELLOW}[robot]${NC}  "

    # ────────────────────────────────────────────────────────────────
    # ALLOWLIST log filter — only signal lines, no per-chunk noise.
    # Set RAW_LOGS=1 to disable filtering and see everything raw.
    #
    # We KEEP a line if it matches any of these patterns (egrep OR):
    #   • transcript=                  user speech (STT result)
    #   • reply_preview=               NAO's text reply
    #   • [stream_tts] enqueue:        what NAO is about to speak
    #   • [tts_trace] play_done|...    audio playback timing/result
    #   • [speaking_gesture] ->        gesture firing
    #   • wake_event / wake_engaged    engagement
    #   • voice_profile_set            voice changed
    #   • motion_match                 short-circuit motion command
    #   • [vision] / [vision_trace]    image pipeline
    #   • vision_status=               vision result
    #   • mic_resumed / camera_announce
    #   • session_open / ws_connected / ws_closed / ws_reconnect
    #   • outcome=ok/rejected/client_dropped
    #   • crisis / safety
    #   • Error|ERROR|Traceback|Exception|Warning
    #   • [volume] / [audio_module]    hardware issues
    #   • [wake_state]                 wake state machine events
    #   • boot_start / broker_ready    startup
    #   • Application startup / Uvicorn / GET /health (uvicorn boot)
    if [[ "${RAW_LOGS:-0}" == "1" ]]; then
        local _filter_cmd='cat'
    else
        # Pre-stage: drop ALL `[nao.ws_client] {…}` JSON lines (robot's
        # own structured-log mirror — every event is also logged on the
        # server side in human-readable form). Then run the allowlist.
        # Two-stage so the allowlist regex stays readable.
        local _prefilter='grep --line-buffered -v "\[nao\.ws_client\] {"'
        # ALLOWLIST — only truly essential signal lines.
        #
        # Tier 1 — must-have (one line per real event):
        #   • stt_legacy ... transcript=...     user speech
        #   • turn_complete ... reply_preview=...  NAO's text reply
        #   • [stream_tts] enqueue: ...         NAO about to speak
        #   • [tts_trace] play_done|play_failed audio playback timing
        #   • [speaking_gesture] -> name        gesture firing
        #   • wake_engaged                      engagement fired
        #   • voice_profile_set                 voice changed
        #   • motion_match                      short-circuit motion
        #   • [vision_trace] image stashed      image arrived at server
        #   • vision_status=success|failed      vision result
        #
        # Tier 2 — startup confirmations (fire once each, then never):
        #   • Application startup / Uvicorn running   server up
        #   • [wake_state] face|touch ... active      wake machine ready
        #   • [audio_module] FIRST PCM captured       mic working
        #   • [stream_tts] amixer pin                 hw mixer pinned
        #
        # Tier 3 — errors/warnings always shown:
        #   • Error|ERROR|Traceback|Exception|Warning
        #   • reject_reason=                         turn rejected (echo, etc.)
        #
        # Everything else (per-chunk JSON, mp3_probe, audio.start_ok, etc.)
        # is dropped. Set RAW_LOGS=1 to disable filtering.
        local _filter_cmd='grep --line-buffered -E "transcript=.*[a-zA-Z]|reply_preview=.*[a-zA-Z]|\[stream_tts\] enqueue:|\[tts_trace\] (blocking_play_done|play_done|play_failed)|\[speaking_gesture\] ->|wake_engaged|voice_profile_set|motion_match|\[vision_trace\] image stashed|\[vision\] image snapped|vision_status=(success|failed)|reject_reason=|Error |ERROR|Traceback|Exception|^[^[]*[Ww]arning[: ]|Application startup complete|Uvicorn running on|FIRST PCM captured|amixer pin Master|face subscription active|touch loop active"'
    fi

    # Build the per-side filter pipeline: prefilter (drop json mirror)
    # → allowlist (keep signal). When RAW_LOGS=1 both stages are 'cat'.
    local _pipeline="${_prefilter:-cat} | $_filter_cmd"

    (tail -f "$SERVER_LOG" 2>/dev/null \
        | eval "$_pipeline" \
        | awk -v p="$server_prefix" '{ print p $0; fflush(); }' \
        || cat "$SERVER_LOG") &
    SERVER_TAIL_PID=$!
    nao_ssh "nao@$NAO_IP" "tail -f $ROBOT_LOG_REMOTE" 2>/dev/null \
        | eval "$_pipeline" \
        | awk -v p="$robot_prefix" '{ print p $0; fflush(); }' &
    ROBOT_TAIL_PID=$!
    # Trap Ctrl+C: kill BOTH tail pipelines and any awk children. Without
    # this, hitting Ctrl+C dropped the foreground awk but left `tail -f` and
    # ssh-tail orphaned, accumulating across runs.
    trap '
        kill $SERVER_TAIL_PID $ROBOT_TAIL_PID 2>/dev/null
        kill_local_tails
        exit 0
    ' INT
    wait $SERVER_TAIL_PID $ROBOT_TAIL_PID 2>/dev/null || true
}

# ─────────── dispatch ───────────
case "${1:-all}" in
    stop)         do_stop ;;
    deploy-only)  do_deploy ;;
    server-only)  start_server; do_tail ;;
    all|"")       do_deploy; start_server; start_robot; do_tail ;;
    *)            die "unknown command: $1 (use: all | deploy-only | server-only | ws | ws-only | stop)" ;;
esac
