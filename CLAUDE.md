# Nao-OpenAI-Morgan-Assist

## Project Overview

NAO humanoid robot assistant for Morgan State University, built on the **OpenAI Agents SDK** with multi-agent routing (router + chat, chatbot, skills, therapist with CBT/grounding/MI sub-agents). Multimodal emotion detection via camera vision.

**The stack is no longer all-OpenAI** (changed 2026-07-28):

| Layer | Provider | Detail |
|---|---|---|
| **Ears** | ElevenLabs | Scribe `scribe_v1` via REST, falls back to OpenAI `gpt-4o-transcribe` |
| **Brain** | Anthropic | Sonnet 5 for conversation lanes, Haiku 4.5 for router/skills/action |
| **Voice** | ElevenLabs | `eleven_flash_v2_5`, falls back to OpenAI `tts-1` |
| **Safety + CBT tools + vision** | OpenAI | `CRISIS_MODEL` / `VISION_MODEL` — the remaining OpenAI holdouts |

`OPENAI_API_KEY` is still **mandatory**: `config.py:17` reads it with
`os.environ[...]`, so a missing key is a `KeyError` at import and the server
never boots — and the fallback paths, vision, and safety all still need it.

**Two indirection layers make provider a config choice, not a code edit:**

- `server/model_factory.py` — for agents on the Agents SDK. `resolve_model()`
  passes OpenAI ids through as strings and wraps `claude-*` ids in
  `LitellmModel`. Any `*_MODEL` env var can name either provider.
- `server/llm_compat.py` — for the seven call sites that use a provider client
  *directly* and never touch the SDK (crisis classifier in `safety.py`,
  `identify_distortion` / `suggest_reframe` / vision in `tools/emotion.py`,
  rollups in `memory_rollup.py`). `chat()` takes OpenAI-shaped messages and
  dispatches on model name. **If you add a direct client call, route it
  through here or it silently pins that feature to one provider.**

Gotcha: `llm_compat` drops `temperature` for Anthropic on purpose. Sampling
params are removed on current Claude models — Opus 5 returns
`400 temperature is deprecated for this model`.

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
| `server/config.py` | Env config (per-agent models, IPs, SQLite path) |
| `server/model_factory.py` | `resolve_model()` — OpenAI string vs Claude `LitellmModel`, for Agents SDK agents |
| `server/llm_compat.py` | `chat()` — provider-agnostic direct calls (system/JSON/image translation) |
| `server/safety.py` | `crisis_check()` + hardcoded 988 hotline reply (keyword gate, LLM only on soft-trigger match) |
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

- **IP:** `172.20.95.123` (confirmed 2026-07-28). **Do not trust this number** —
  the lease moved `.127` → `.123` between 2026-07-15 and 2026-07-28, and the Pi
  moved too. Always resolve before use:
  `dscacheutil -q host -a name nao.local | grep ip_address`, and update `NAO_IP`
  in `.env` when it changes (`run.sh` reads it from there). See below for making
  it static.
- **Hostname:** `nao.local` (mDNS fallback)
- **User:** `nao`
- **Password:** stored in `.env` as `NAO_PASSWORD` (do NOT commit the password; `.env` is gitignored). Only needed for the initial key push — passwordless SSH is now configured (see below).
- **OS:** Aldebaran RT Linux, kernel 4.4.185, x86_64; **Python 2.7.15** at `/usr/bin/python`.

### SSH

Passwordless key auth is **set up and confirmed working (2026-07-15)** — an ed25519 key (`~/.ssh/id_ed25519`) is installed on the robot and a `Host nao` alias is in `~/.ssh/config`. Just use:

```bash
ssh nao                   # key auth, no password
ssh nao@172.20.95.123     # equivalent, explicit host
ssh nao@nao.local         # elsewhere, if mDNS resolves
```

`~/.ssh/config` entry:

```
Host nao
  HostName 172.20.95.123
  User nao
  IdentityFile ~/.ssh/id_ed25519
```

To re-provision the key on a fresh machine: `ssh-copy-id nao@172.20.95.123` (uses `NAO_PASSWORD` from `.env` once). VS Code Remote-SSH picks up the `Host nao` alias automatically.

### Making the IP static

Best path: file a ticket with Morgan IT giving them the NAO's WiFi MAC address (`ifconfig wlan0 | grep ether` on the robot) and request a DHCP reservation for `172.20.95.123`. That survives firmware updates and doesn't require touching the robot.

Fallback: configure a fixed IP via Choregraphe (Settings → Network → "Use a fixed IP address") or via `connmanctl` on the robot directly.

## The three computers

The robot is a body, not a brain. Every turn is: NAO records audio → ships it
over WiFi to a **server** → server calls OpenAI → speech + actions come back.
The server runs on one of two machines, and *which one* is the single biggest
source of confusion:

| | Runs | Address | When |
|---|---|---|---|
| **NAO robot** | `nao/` (Python 2.7) | `172.20.95.123` | always |
| **Raspberry Pi 4** | `server/` via systemd `nao-server` | `172.20.95.126` | production / demos |
| **Your laptop** | `server/` via `./run.sh` | your LAN IP | development |

The Pi is the always-on brain so the robot works with nobody's laptop around
(see README "Always-on deployment"). `./run.sh` stands your laptop in for it.

- **Robot code** reaches the robot by **rsync** (`./run.sh`) — never git.
- **Server code** reaches the Pi by **git push → Pi `git pull`** — never rsync.
- **`.env` reaches the Pi by neither.** It is gitignored, so every key and model
  setting must be recreated there by hand. A Pi that pulled fine but behaves
  like an older build is almost always an out-of-date `.env`.

### Pi access (as of 2026-07-28: blocked)

Its hostname is `naoserver`, and it runs `avahi-daemon`, so
`dscacheutil -q host -a name naoserver.local` finds it without scanning — much
faster than a subnet sweep, and it survives lease changes.

**There is no working login from this laptop.** SSH key auth is set up for the
robot but was never installed on the Pi, and the `nao` password (hashed in the
SD card's `user-data`) is not in `.env` or any doc. Attempting to install a key
by adding `ssh_authorized_keys` to `user-data` and bumping `instance-id` in
`meta-data` did **not** work — cloud-init did not re-apply. Remaining options:
a monitor + keyboard on the Pi, or the password from whoever imaged it.
Backups of the originals are on the card as `user-data.bak.orig` /
`meta-data.bak.orig`.

Note `ssh`/`ssh-copy-id` password prompts **cannot** be answered from a
non-interactive shell (including tool-run commands) — they fail instantly with
`Permission denied` having never prompted. Run those in a real terminal.

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
- **`ELEVENLABS_STT_MODEL` must be `scribe_v1`.** `config.py` defaults it to
  `scribe_v2_realtime`, which the REST endpoint rejects with
  `400 unsupported_model`. The realtime *WebSocket* additionally 403s without a
  realtime entitlement, so `USE_ELEVENLABS_STT_REALTIME` defaults to `0` and the
  batch REST path is what actually runs.
- **`USE_DEEPGRAM=1` / `USE_ELEVENLABS_TTS=1` are no-ops without their keys** —
  everything falls through to OpenAI. Read the boot log, not the flag.
- **Dead env vars are a recurring trap.** A `*_MODEL` var only works if
  something reads `config.X`. `CBT_MODEL` / `GROUNDING_MODEL` existed for months
  and were read by nothing (both coaches read `THERAPIST_MODEL`) — setting them
  looked like it worked. Before trusting a knob:
  `grep -rn "config\.YOUR_VAR" server --include="*.py"`.
- **`CRISIS_MODEL` is badly named** — it drives the crisis classifier *and*
  `identify_distortion` / `suggest_reframe` (the clinical core of the CBT flow)
  *and* memory rollups. Changing it to tune CBT also changes the safety gate.
  Vision is separate (`VISION_MODEL`).
- **`SAFETY_MODEL_CLAUDE` / `SAGE_SAFETY_PROVIDER` do nothing at
  `SAGE_TOPOLOGY=passthrough`** (the default). They configure
  `topologies/safety_agent.py`, which only runs under the SAGE research
  topologies. The live gate is `safety.py` + `CRISIS_MODEL`.

### Known bugs (unfixed as of 2026-07-28)

- **Silero VAD shares one model across sessions.** `vad_silero._model` is a
  process-wide singleton, but Silero v5 is a stateful RNN and each session gets
  its own `StreamingSilero` wrapper feeding it. Concurrent sessions interleave
  audio into one hidden state and confidence collapses to ~0; one session's
  `reset()` calls `_model.reset_states()` and wipes another's state mid-turn.
  Symptom: `reject_reason=silero_no_speech` **with a full, correct transcript**
  — STT heard you fine, the gate in front of the agent threw it away. Gets worse
  as WebSocket reconnects accumulate; `./run.sh` clears it. Fix is a per-session
  model instance (the VAD model is ~1-2 MB, so copies are cheap).
- **The crisis gate only consults the LLM when a keyword matches.** No match in
  `_SOFT_TRIGGERS` → returns `clean` without any model call. It matches
  contracted forms only, so *"I do not want to be here anymore"* misses (the
  list has `"don't want to be here"`), as do *"I want to disappear"* and
  *"I feel like a burden"*. Upgrading `CRISIS_MODEL` does not help — the model
  never sees those messages.
- **The therapy lane does not persist.** Routing is recomputed per turn from
  keywords, so a follow-up phrased without a trigger word silently drops from
  `therapist` back to `chat`, losing the CBT tools and coach handoffs
  mid-conversation. The router also costs a second sequential model call
  (measured 5.7s therapist alone vs 8.9s via router) even though
  `pick_initial_agent` already matched the emotional keyword in Python.

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
  `nc -z -G 4 172.20.95.123 22`.
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
