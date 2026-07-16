# Nao-OpenAI-Morgan-Assist

## Project Overview

NAO humanoid robot assistant for Morgan State University, built on the **OpenAI Agents SDK** with multi-agent routing (router + chat, chatbot, skills, therapist with CBT/grounding/MI sub-agents). Integrates Whisper (STT), GPT-4.1/4o (chat + vision), and OpenAI TTS. Multimodal emotion detection via GPT-4o vision.

Everything — ears, brain, and voice — is OpenAI. `OPENAI_API_KEY` is
**mandatory**: `config.py` reads it with `os.environ[...]`, so a missing key is
a `KeyError` at import and the server never boots. Anthropic can only replace
the brain (there is no Anthropic STT or TTS); it is wired into exactly one
place today — the crisis classifier (`SAGE_SAFETY_PROVIDER=claude`).

Pinecone is **gone** — RAG is now an HTTP proxy to Cloud Run
(`server/tools/cs_navigator.py`, `vertex_search.py`). No embeddings anywhere.

## Repo Layout

```
Nao-OpenAI-Morgan-Assist/
├── nao/         NAO-side Python 2.7 code — copy this to the robot
├── server/      Python 3.11+ Flask server + Agents SDK graph
├── docs/        Design specs, implementation plans, reference docs
├── README.md
├── CLAUDE.md
├── LICENSE
└── pytest.ini
```

## Architecture

**The live path is FastAPI + WebSocket** (`USE_WS=1` in `.env` → `server/app_ws.py`,
served by uvicorn). `server/server.py` (Flask `POST /turn`) and
`nao/conversation.py` are the **legacy** path — still present, not what runs.
Check `USE_WS` before assuming which file you're debugging. `realtime_proxy.py`
is unreachable under `USE_WS=1`.

```
NAO Robot (Python 2.7 / naoqi SDK) — everything under nao/
  nao/main.py           -> Wake loop + wake state machine
  nao/wake_state.py     -> Face + touch gates (NO voice wake word)
  nao/ws_client.py      -> WebSocket client: streams PCM, plays TTS, runs actions
  nao/audio_module.py   -> Mic capture (fragment recorder; MIC_CHANNEL selects mic)

Server (Python 3.11+) — everything under server/
  server/app_ws.py        LIVE: FastAPI + WebSocket /ws/{username}
  server/server.py        LEGACY: Flask POST /turn (USE_WS=0 only)
  server/safety.py        Pre-dispatch crisis gate (keyword + LLM, 988 hotline)
  server/session.py       SQLiteSession wrapper + camera consent + therapy recaps
  server/agents/          Agent graph (router, chat, chatbot, skills, therapist, cbt_coach, grounding_coach)
  server/tools/           Tool modules (nao_actions, pinecone_search, emotion, skills_tools)
```

## Key Files

### NAO side (Python 2.7, `nao/`)
| File | Purpose |
|------|---------|
| `nao/main.py` | Entry; wake -> conversation loop |
| `nao/wake_listener.py` | Wake phrase detection + `extract_hint()` |
| `nao/conversation.py` | Single mode loop (record, POST, speak, dispatch actions) |
| `nao/audio_handler.py` | Mic recording with VAD |
| `nao/processing_announcer.py` | Background "please wait" speaker |
| `nao/config.py` | Env-driven NAO IP/SERVER IP |
| `nao/utils/camera_capture.py` | JPEG capture including `snap_quick()` for per-turn vision |
| `nao/utils/nao_execute.py` | Dispatches `{name, args}` actions from server to naoqi calls |
| `nao/utils/face_naoqi.py` | Face recognition/learning via ALFaceDetection |
| `nao/utils/ask_name_utils.py` | Ask user for name via audio round-trip |
| `nao/utils/exit_detection.py` | Regex-based exit intent |
| `nao/utils/name_utils.py` | Extract name from speech |
| `nao/utils/speech.py` | Phrase pools + expressive TTS |

### Server side (Python 3.11+)
| File | Purpose |
|------|---------|
| `server/server.py` | Flask `POST /turn` + `GET /health` |
| `server/config.py` | Env config (models, Pinecone, IPs, SQLite path) |
| `server/safety.py` | `crisis_check()` + hardcoded 988 hotline reply |
| `server/session.py` | SQLiteSession + `{get,set}_camera_consent` + `{save,load}_recap` |
| `server/agents/router.py` | Triage agent with handoffs |
| `server/agents/chat.py` | General chat + NAO actions |
| `server/agents/chatbot.py` | Morgan CS RAG |
| `server/agents/skills.py` | Time/weather/timers/todos |
| `server/agents/therapist.py` | Empathetic + CBT/grounding handoffs |
| `server/agents/cbt_coach.py` | Thought record walker |
| `server/agents/grounding_coach.py` | 5-4-3-2-1, box breathing, body scan |
| `server/tools/nao_actions.py` | 18 NAO action tools (append to `actions_queue`) |
| `server/tools/pinecone_search.py` | RAG tool |
| `server/tools/emotion.py` | `observe_face`, `log_emotion`, `identify_distortion`, `suggest_reframe`, `set_camera_consent`, `recap_session` |
| `server/tools/skills_tools.py` | Utility tools |

## Development Guidelines

- **NAO-side** (everything in `nao/`): **Python 2.7 compatible**. `from __future__ import print_function`, `str.format()`, no f-strings, no type hints. On the robot, copy `nao/` contents to `/home/nao/nao_assist/` and run `python /home/nao/nao_assist/main.py`.
- **Server-side** (`server/`): **Python 3.11+**. Modern idioms fine.
- IPs/ports read from env or `config.py` — never hardcode.
- Agents SDK: `openai-agents>=0.0.5` (currently 0.13.6).
- `pytest.ini` at repo root pins rootdir so the SDK's `agents` module doesn't get shadowed by `server/agents/`.
- NAO action tools append `{name, args}` records to a context-scoped `actions_queue`; after `Runner.run()` returns, the queue is read out and sent to NAO in the response JSON. NAO-side `utils/nao_execute.py` dispatches them.
- Crisis gate runs **before** the agent sees the user message. Agent cannot override.
- Camera consent persists in `user_prefs` table; therapist tool `set_camera_consent` toggles it; NAO honors the `suppress_image` flag in responses.

## Obsidian Vault

Knowledge vault for this codebase at `~/Documents/Obsidian Vault/Nao-OpenAI-Morgan-Assist/wiki/`. Read `wiki/index.md` first for context. Pattern: `raw/` (immutable) + `wiki/` (LLM-maintained).

## NAO Robot — Connection

- **IP:** `172.20.95.127` (confirmed reachable 2026-07-15 on the CS network; may change if the lease drops — see below for making it static)
- **Hostname:** `nao.local` (mDNS fallback)
- **User:** `nao`
- **Password:** stored in `.env` as `NAO_PASSWORD` (do NOT commit the password; `.env` is gitignored). Only needed for the initial key push — passwordless SSH is now configured (see below).
- **OS:** Aldebaran RT Linux, kernel 4.4.185, x86_64; **Python 2.7.15** at `/usr/bin/python`.

### SSH

Passwordless key auth is **set up and confirmed working (2026-07-15)** — an ed25519 key (`~/.ssh/id_ed25519`) is installed on the robot and a `Host nao` alias is in `~/.ssh/config`. Just use:

```bash
ssh nao                   # key auth, no password
ssh nao@172.20.95.127     # equivalent, explicit host
ssh nao@nao.local         # elsewhere, if mDNS resolves
```

`~/.ssh/config` entry:

```
Host nao
  HostName 172.20.95.127
  User nao
  IdentityFile ~/.ssh/id_ed25519
```

To re-provision the key on a fresh machine: `ssh-copy-id nao@172.20.95.127` (uses `NAO_PASSWORD` from `.env` once). VS Code Remote-SSH picks up the `Host nao` alias automatically.

### Making the IP static

Best path: file a ticket with Morgan IT giving them the NAO's WiFi MAC address (`ifconfig wlan0 | grep ether` on the robot) and request a DHCP reservation for `172.20.95.127`. That survives firmware updates and doesn't require touching the robot.

Fallback: configure a fixed IP via Choregraphe (Settings → Network → "Use a fixed IP address") or via `connmanctl` on the robot directly.

## The three computers

The robot is a body, not a brain. Every turn is: NAO records audio → ships it
over WiFi to a **server** → server calls OpenAI → speech + actions come back.
The server runs on one of two machines, and *which one* is the single biggest
source of confusion:

| | Runs | Address | When |
|---|---|---|---|
| **NAO robot** | `nao/` (Python 2.7) | `172.20.95.127` | always |
| **Raspberry Pi 4** | `server/` via systemd `nao-server` | `172.20.95.106` | production / demos |
| **Your laptop** | `server/` via `./run.sh` | your LAN IP | development |

The Pi is the always-on brain so the robot works with nobody's laptop around
(see README "Always-on deployment"). `./run.sh` stands your laptop in for it.

- **Robot code** reaches the robot by **rsync** (`./run.sh`) — never git.
- **Server code** reaches the Pi by **git push → Pi `git pull`** — never rsync.

## Running it (development)

```bash
./run.sh              # deploy nao/ + clear .pyc + start server + relaunch robot + tail
./run.sh deploy-only  # rsync only
./run.sh server-only  # server only
./run.sh stop         # kill server + robot main.py
```

**After every robot reboot you must run `./run.sh`.** On power-up a Choregraphe
behavior autostarts `main.py` with no `SERVER_IP`, so it falls back to
`nao/config.py`'s default — the **Pi at `.106`**. If the Pi is off, the robot
wakes, calls a dead host, and silently does nothing. `run.sh` kills that process
and relaunches pointed at your laptop.

Do **not** hand-roll the rsync. `run.sh` also clears `.pyc` (Python 2 prefers
stale bytecode, which makes edits silently no-op) and excludes `nao.log` —
a bare `rsync --delete nao/ ...` **deletes the robot's live log file**.

### Robot-side env knobs (forwarded by `run.sh`)

| Var | Default | Notes |
|---|---|---|
| `MIC_CHANNEL` | `left` | Which of NAO's 4 mics to record. Measured: LEFT RMS≈1045 transcribes cleanly; FRONT ≈730 is weakest and made Whisper emit Chinese for English. `left\|right\|front\|rear\|all`. |
| `ENGAGE_POSTURE` | `Sit` (code) | Posture on wake; `.env` currently overrides to `Stand`. `Sit` saves battery and mic noise; `none` disables. |
| `SPEAKING_GESTURES` | `1` | `0` stops arm micro-gestures during TTS. |

### Server-side gotchas

- **`FFMPEG_BIN`** must point at a real ffmpeg or `OPENAI_TTS_GAIN_DB` is
  *silently skipped* and NAO is barely audible. No Homebrew on this Mac; ffmpeg
  lives in a conda env. Check `logs/server.log` for `ffmpeg unavailable`.
- `run.sh` pins `.venv/bin/python`. A bare `python` resolves to miniconda,
  which lacks the deps — the server dies at import on `structlog`.
- `USE_DEEPGRAM=1` / `USE_ELEVENLABS_TTS=1` in `.env` are **no-ops** without
  their keys/voice IDs — everything falls through to OpenAI. Docs record that
  Deepgram was benchmarked and rejected (slow on the CS network).
- `pytest-asyncio` is **not installed**, so every `@pytest.mark.asyncio` test
  **silently skips**. Drive async tests with `asyncio.run()` instead.
- ~32 tests fail on a clean tree (pre-existing). Diff against a stash before
  blaming your change.

## Identity / face recognition

Two stores, joined only by a **name string**:

- **Robot:** `/home/nao/.local/share/vision/facerecognition/default` — OMRON
  OKAO (`OKAOFR70`) feature templates, ~1.4 KB for several faces. Never images;
  you cannot reconstruct a face from it. Inspect with
  `ALFaceDetection.getLearnedFacesList()`.
- **Server:** `users` table in `server/nao.db` (`face_id` → `display_name`).
  `face_id` holds the **learned name**, not a number.

`ALFaceDetection` extra_info: `[0]` is a transient internal id renumbered on
every detection (same person was 12, then 18); `[2]` is the recognised name.
`wake_state.identity_key_for_face()` prefers `[2]` and falls back to `[0]`.

Say **"remember me as X"** to trigger the `learn_face` fast-path — it teaches
the robot's face DB *and* upserts the `users` row (`_emit_motion` →
`memory.ensure_user`). Both must happen or recognition silently never works.

## Debugging the robot

- **Never diagnose reachability with `ping`.** The gateway and the robot
  ignore/drop ICMP; "100% packet loss" told us the robot was dead three times
  while SSH was wide open. Test the port you actually need:
  `nc -z -G 4 172.20.95.127 22`.
- Robot logs: `/home/nao/nao_assist/nao.log` (recreated by `run.sh` on launch).
- `NAO_SHARED_SECRET` must not be inline in commands (the safety classifier
  blocks plaintext secrets, and `ps` exposes it on the robot). Source an env file.
- `pkill -f main.py` from your laptop can match your own SSH command and kill
  the connection; use `pgrep -af "[m]ain\.py"`.
- Wake gates are **face and touch only — there is no voice wake word.** Eyes:
  gray=idle, soft blue=aware, solid blue=engaged, cyan=listening. Touch while
  ENGAGED means *barge-in*; rear-head touch means *stop*.
- The face loop can wedge on `ALMemory read error` spam and stop streaming audio
  entirely (zero `audio_chunk`) while still answering touch. `./run.sh` clears it.
- The robot's battery drains fast; standing drains it faster. Keep it charged.
- Its deployed tree can diverge from local `nao/`. Diff before `--delete`.
