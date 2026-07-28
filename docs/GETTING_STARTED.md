<!--
title: Getting Started (new maintainer onboarding)
tags: [onboarding, setup, dev-environment]
related: [README, DECISIONS, PRD_v2]
status: living-document
-->

# Getting Started

For someone inheriting this project with no prior NAO experience. Read top to
bottom the first time. Each stage ends with a **checkpoint** — don't move on
until it passes.

---

## 0. The one-paragraph mental model

The NAO robot is **not smart**. It is a microphone, a speaker, a camera, and
some motors with a small Python 2.7 program (`nao/`) whose only job is: capture
audio, ship it over a WebSocket to a server, and play back whatever audio and
motion commands come back. All the intelligence — speech-to-text, the LLM
agents, safety checks, text-to-speech — runs in the Python 3 server (`server/`)
on your laptop (or on a Raspberry Pi). If the server isn't running, the robot
does nothing but stand there.

```
You speak  →  NAO mic  →  WebSocket  →  server: STT → safety gate → LLM agent → TTS
                                            ↓
NAO speaker + motors  ←  WebSocket  ←  audio chunks + action frames
```

So: **most of your work happens on the server side, on your own machine.** You
can develop and test the entire pipeline without the robot powered on.

---

## 1. Things that will confuse you (read this before anything else)

The repo has been through a big rework. Some docs describe the *old* design.
These are the traps, in the order you'll hit them:

| Trap | Reality |
|---|---|
| `CLAUDE.md` describes a Flask `POST /turn` server and Pinecone RAG | **Stale.** The real server is FastAPI + WebSocket (`server/app_ws.py`), and RAG goes through the CS Navigator HTTP API. Trust `README.md` and the code. |
| `.env.example` ships `USE_WS=0` | **`USE_WS=0` is broken.** `nao/main.py` only knows how to speak WebSocket — `nao/conversation.py` (the old HTTP path) is dead code that nothing imports. You must set **`USE_WS=1`** or launch with `./run.sh ws`. |
| `run.sh` defaults to booting Flask | Same reason. It will start `server/server.py`, the robot will fail to connect, and the failure is quiet. |
| README quick start says `python3.11 -m venv` | You don't have Python 3.11 installed (you have 3.13). See §2 — this matters. |
| README quick start goes straight to `./run.sh` | `run.sh` needs `sshpass`, which you don't have, and it refuses to start without `NAO_IP` + `NAO_PASSWORD` — even for `server-only`. |

Nothing here is broken *code*; it's just documentation that drifted ahead of /
behind the implementation. You'll be fine once you know.

---

## 2. Set up your machine

### 2.1 Install Homebrew

You have no package manager. Everything else depends on this.

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Follow the "Next steps" it prints at the end — it will tell you to run two
`echo ... >> ~/.zprofile` lines to put `brew` on your `PATH`. Do them, then
close and reopen your terminal.

**Checkpoint:** `brew --version` prints a version.

### 2.2 Install Python 3.11

The server targets Python 3.11. Your system has 3.13. Two dependencies in
`server/requirements.txt` are the reason this matters:

- `webrtcvad==2.0.10` — ships no prebuilt wheel for 3.13; pip will try to
  compile it from C source and probably fail.
- `torch` (pulled in by `silero-vad`) — a ~2 GB download; version support
  lags new Python releases.

Save yourself the afternoon:

```bash
brew install python@3.11
```

**Checkpoint:** `python3.11 --version` prints `Python 3.11.x`.

### 2.3 Install the CLI tools `run.sh` shells out to

```bash
brew install ffmpeg rsync
brew install hudochenkov/sshpass/sshpass
```

- **`sshpass`** — lets `run.sh` log into the robot with a password
  non-interactively. Not in Homebrew core (it's considered a footgun), hence
  the tap. Required: `run.sh` calls it for every remote command.
- **`ffmpeg`** — used server-side to boost TTS volume before sending audio to
  the robot. *Optional:* without it, `server/openai_tts.py` logs
  `ffmpeg unavailable, skipping gain` and carries on — NAO just speaks quietly.
  The robot has its own copy at `/usr/bin/ffmpeg` already.

**Checkpoint:** `sshpass -V` and `ffmpeg -version` both print something.

### 2.4 Create the virtualenv and install the server

```bash
cd ~/Desktop/nao-sagecbt
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r server/requirements.txt
```

This pulls torch. It is slow (several minutes, ~2 GB). Let it run.

> If `webrtcvad` still fails to build: it is genuinely optional. Comment it out
> in `server/requirements.txt` and reinstall. The code's own comment says
> `_has_voice()` "falls back to permissive if this is missing." You lose one of
> two voice-activity filters; the neural one (`silero-vad`) still runs.

Remember: **every time you open a new terminal**, run `source .venv/bin/activate`
first. If you forget, you'll get `ModuleNotFoundError` and think something is
broken.

**Checkpoint:**

```bash
python -c "import fastapi, agents, torch; print('deps ok')"
```

---

## 3. Configure your secrets

```bash
cp .env.example .env
```

Now open `.env` and edit it. `.env` is gitignored — **never commit it.**

### The only key you truly need

```ini
OPENAI_API_KEY=sk-...
```

`server/config.py:17` reads this with `os.environ["OPENAI_API_KEY"]` — a bare
subscript, not `.get()`. Without it the server crashes on import with a
`KeyError` before it prints anything useful. Get a key from
<https://platform.openai.com/api-keys>.

### The one line you must change from the default

```ini
USE_WS=1
```

See §1. Leave it at `0` and the robot cannot talk to the server.

### Everything else degrades gracefully

| Variable | Leave blank and… |
|---|---|
| `DEEPGRAM_API_KEY` | speech-to-text falls back to OpenAI Whisper (slower, works fine) |
| `ELEVENLABS_API_KEY` | text-to-speech falls back to OpenAI TTS (slower, works fine) |
| `CS_NAVIGATOR_URL` | the "Morgan CS course questions" agent politely apologizes; everything else works |
| `NAO_SHARED_SECRET` | server runs in "open mode" — anyone on the network can connect. Fine on your laptop, logs a warning at boot |
| `ANTHROPIC_API_KEY` | unused unless you set `SAGE_SAFETY_PROVIDER=claude` |

### Robot-only variables

Fill these in later, once you've done §5. `run.sh` refuses to start without
them, which is why §4 bypasses `run.sh` entirely.

```ini
NAO_IP=172.20.95.127
NAO_PASSWORD=<ask whoever handed you this project>
```

**Checkpoint:** `grep -c PASTE_ .env` prints `0` for the keys you filled in.
(Placeholders you're intentionally leaving blank are fine — `run.sh` only hard-fails on `OPENAI_API_KEY`, `NAO_IP`, and `NAO_PASSWORD`.)

---

## 4. Run the server with no robot

This is where you should live for your first few days. The robot adds an entire
category of problems (network, hardware, Python 2.7) that you don't need yet.

Do **not** use `run.sh` for this — it demands `NAO_PASSWORD` even in
`server-only` mode. Start uvicorn directly:

```bash
source .venv/bin/activate
python -m uvicorn server.app_ws:app --host 0.0.0.0 --port 5050 --log-level info
```

In a second terminal:

```bash
curl http://localhost:5050/health
# {"ok":true,"version":"phase-1"}
```

**Checkpoint:** that `curl` returns JSON. If it doesn't, read the uvicorn
output — a missing `OPENAI_API_KEY` or a failed import will be the first
traceback in it.

### Now exercise the pipeline without hardware

Unit and integration tests (~30 of them, no network calls):

```bash
pytest -q
```

Scripted end-to-end scenarios that drive the real FastAPI app in-process with
mocked STT/TTS/LLM — face wake, a therapy turn, a barge-in, an echo-rejection:

```bash
python -m sim.scenarios              # list them
python -m sim.scenarios 01_face_wake # run one
python -m sim.scenarios all          # run all six
```

Talk to the server with your Mac's own microphone, as if you were the robot:

```bash
brew install portaudio
pip install sounddevice numpy pydub
python sim/live_nao.py
```

**Checkpoint:** `pytest -q` is green and `python -m sim.scenarios 01_face_wake`
prints `ok`. At this point you have the whole brain working on your laptop and
you understand more than half of this system.

---

## 5. Meet the robot

### 5.1 Get on the same network

The robot lives at `172.20.95.127` on the Morgan CS network. You must be on
that same network — this is a LAN-only setup, there is no cloud tunnel.

```bash
ping -c 3 172.20.95.127
```

If it doesn't answer:
- Press the robot's chest button once. It will **say its own IP address out
  loud.** This is the single most useful NAO trick.
- The IP is a DHCP lease and may have changed. Update `NAO_IP` in `.env`.
- Try `ping nao.local` (mDNS fallback).

### 5.2 SSH in

```bash
ssh nao@172.20.95.127     # password: NAO_PASSWORD from .env
```

Look around. The deployed code lives at `/home/nao/nao_assist/`. It's a copy of
this repo's `nao/` folder — you never edit it in place, you edit locally and
redeploy.

Make life easier by installing your SSH key so you stop typing the password:

```bash
ssh-copy-id nao@172.20.95.127
```

**Checkpoint:** you get a shell prompt on the robot, and
`ls /home/nao/nao_assist/` shows `main.py`.

### 5.3 Understand what's already running

Read this before you launch anything, or you will spend a day debugging a
conflict you created.

Per the README, the project was left in an **always-on deployment**:

- A **Raspberry Pi at `172.20.95.106`** runs the server as a systemd service
  (`nao-server.service`), so the robot works without anyone's laptop.
- The robot **autostarts** `main.py` at boot via a Choregraphe default behavior
  (`nao-therapy-autostart`), which runs `/home/nao/launch_nao_assist.sh`.

So the robot is probably already talking to the Pi right now. Check:

```bash
ssh nao@172.20.95.127 'ps aux | grep -v grep | grep main.py'
ssh nao@172.20.95.106 'sudo systemctl is-active nao-server'
```

When you run `./run.sh` from your Mac it kills the robot's `main.py` and
relaunches it pointed at *your laptop's* IP. That's correct and intended for
development. Just know that:

- **Only one `main.py` may run at a time.** Two processes fight over the
  microphone and speaker. `run.sh` kills stale ones for you; if things get
  weird, `ssh nao@$NAO_IP 'pkill -9 -f main.py'` and start over.
- **Rebooting the robot** re-triggers the Choregraphe autostart, pointing it
  back at the Pi. Re-run `./run.sh` to reclaim it.

---

## 6. Run the whole thing

```bash
source .venv/bin/activate
./run.sh ws
```

`ws` forces `USE_WS=1` for this run regardless of `.env` — use it until you're
confident `.env` is right.

What it does, in order: rsyncs `nao/` to the robot → kills any stale `main.py`
→ boots uvicorn on `:5050` → waits for `/health` → launches one `main.py` on
the robot pointed at your Mac's IP → tails both logs side by side, filtered
down to signal lines.

Useful variants:

```bash
./run.sh ws          # everything (this is the one you want)
./run.sh deploy-only # just push nao/ to the robot
./run.sh stop        # kill the local server + robot main.py
RAW_LOGS=1 ./run.sh ws   # unfiltered logs, for when you're debugging
```

`Ctrl-C` stops the log tail but **leaves the server and robot running.** Use
`./run.sh stop` to actually shut down.

### Say hello

Walk into the robot's field of view, or say "Hey NAO". Its eyes turn cyan when
it's listening. Then try:

| Say | What should happen |
|---|---|
| "Hey NAO, how are you?" | short spoken reply, ~1 s |
| "Wave at my friend" | it waves |
| "What am I wearing?" | it takes a photo and describes you |
| "Remember me as \<your name\>" | it learns your face; next session it greets you by name |
| "I'm anxious about finals" | routes to the therapist agent |

Tap its head sensors to interrupt it mid-sentence.

**Checkpoint:** you said something, the `[server]` log shows a
`transcript=` line with your words, and the robot spoke back.

---

## 7. When it doesn't work

Debug in the direction the audio flows. The `[server]` / `[robot]` prefixes in
the `run.sh` log tell you which half you're in.

| Symptom | Where to look |
|---|---|
| Robot silent, no `[robot]` log lines | `main.py` isn't running. `ssh nao@$NAO_IP 'tail -50 /home/nao/nao_assist/nao.log'` |
| Robot connects then immediately drops | `NAO_SHARED_SECRET` mismatch between `.env` and what the robot was launched with. Simplest fix: blank it on both sides while developing |
| No `transcript=` line when you speak | mic or VAD. Look for `FIRST PCM captured` (mic works) and `[silero_trace]` (VAD scoring). The VAD threshold is `0.15` in `server/vad_silero.py` |
| Transcript appears, no reply | agent/LLM side. Check `OPENAI_API_KEY` and look for a traceback in `logs/server.log` |
| Reply text in logs, no sound | TTS or playback. Grep the robot log for `[tts_trace]` and `[stream_tts] enqueue` |
| Robot talks to itself in a loop | echo rejection. Look for `reject_reason=` — it's hearing its own speaker |
| Edited a `nao/` file, no change on the robot | stale `.pyc`. `./run.sh deploy-only` clears them; it's a known Python 2 gotcha |

Two logs, always:

```bash
tail -f logs/server.log                              # server, full detail
ssh nao@$NAO_IP 'tail -f /home/nao/nao_assist/nao.log'  # robot
```

---

## 8. What to read next, in order

1. **`README.md`** — the mermaid diagrams. Especially "Voice + mic lifecycle"
   and "Wake state machine". Skim now, re-read after your first working turn.
2. **`docs/DECISIONS.md`** — twelve hard problems and why they were solved the
   way they were. This is the highest-value doc in the repo; it will stop you
   from "fixing" things that are deliberate.
3. **`docs/PRD_v2.md`** — the full spec, if you need the why behind a phase.
4. **`server/app_ws.py`** — the ~3200-line heart of the system. Start at
   `@app.websocket("/ws/{username}")` (near line 3090) and follow one turn
   through.

Read `server/safety.py` before you touch anything therapy-related. The crisis
gate runs **before** the LLM ever sees the user's message and returns a
hardcoded 988 hotline response. It is not an agent, it cannot be overridden by
one, and that is on purpose.

---

## 9. Your first week, suggested

1. Get `pytest -q` green and `sim.scenarios all` passing on your laptop. *(No robot.)*
2. Run `sim/live_nao.py` and hold a conversation with the server through your Mac's mic. *(No robot.)*
3. Read `DECISIONS.md` end to end.
4. Get `./run.sh ws` working with the real robot and complete one voice turn.
5. Make one tiny change — e.g. edit a phrase in `nao/utils/speech.py` — deploy
   it, and hear the robot say it. That closes the loop and the rest is detail.
