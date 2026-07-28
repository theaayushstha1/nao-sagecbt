# Orientation — NAO + Raspberry Pi, in plain words

A beginner-friendly map of what this project is, how the pieces talk to each
other, and what to keep in mind before you touch anything. If you read only one
doc first, read this one.

## The big idea

The robot can't think. It's a body. The thinking happens on a *separate*
computer (the Raspberry Pi, or your laptop), which asks OpenAI's cloud for the
actual answers. Everything else is just plumbing between those three.

```
   ┌──────────────┐        ┌──────────────────┐         ┌─────────────┐
   │  NAO robot   │  WiFi  │  Raspberry Pi    │ internet│   OpenAI    │
   │  "the body"  │◄──────►│  "the brain"     │◄───────►│  "the AI"   │
   │              │  :5050 │  (the server)    │         │             │
   └──────────────┘        └──────────────────┘         └─────────────┘
   mic, speaker,           runs your server/             Whisper (hears),
   camera, arms,           code, always on,              GPT-4o (thinks),
   Python 2.7              Python 3.11+                   TTS (voices)
```

**One turn of conversation, step by step:**

1. You speak. The robot's mic records it.
2. The robot sends the audio over WiFi to the Pi (port 5050).
3. The Pi sends it to OpenAI: Whisper turns your voice into text, GPT-4o decides
   what to say.
4. The Pi turns that reply into speech and sends it back to the robot, plus any
   actions like "wave."
5. The robot speaks the reply and moves. Back to step 1.

If the Pi is off, or OpenAI can't be reached, the robot has no brain and just
sits there.

## The three computers (the thing to really get)

| | What it is | Runs | Language |
|---|---|---|---|
| **NAO robot** | The physical robot — mic, speaker, camera, motors | `nao/` folder, copied to the robot | **Python 2.7** |
| **Raspberry Pi** | Small always-on computer = the server/brain | `server/` folder | **Python 3.11+** |
| **Your laptop** | Where you write code + can stand in for the Pi | the repo | Python 3.11+ |

The Pi and your laptop do the *same job* (run the server). The difference: the
Pi is always on so the robot works with nobody around; your laptop is for
development and dies when you close it.

## How the Raspberry Pi setup works

The Pi is set up as a **permanent server** so the robot "just works." Concretely:

- The Pi runs Ubuntu and boots the server automatically as a background service
  called **`nao-server`** (via systemd — Linux's "start this on boot and keep it
  alive" system). You never manually start it; it's already running.
- The robot, on power-up, **autostarts itself** too: NAOqi boots → a Choregraphe
  behavior fires → runs `launch_nao_assist.sh` → launches `main.py`. It then
  finds the Pi and starts talking.
- So the daily reality is: **power on the robot, and it works.** No laptop, no
  commands.

When you change the server code, you don't "run" anything new — you update the
Pi and restart its service:

```bash
ssh nao@<pi-ip>                                    # get onto the Pi
cd ~/nao-sagecbt && git pull                        # get your new code
sudo systemctl restart nao-server                   # restart the brain
sudo journalctl -u nao-server -f                    # watch it live
```

## The two ways to run everything

**Development (your laptop is the brain):** `./run.sh` copies `nao/` to the
robot, starts the server on your laptop, and points the robot at your laptop.
Good for testing changes fast. Dies when you close the laptop.

**Production (the Pi is the brain):** already running on the Pi, 24/7. You just
`git pull` + restart. This is the always-on setup.

## Numbers and names you'll keep seeing

- **9559** — the robot's internal port. Same on every NAO, never changes, not a
  secret.
- **5050** — the server's port. What the robot calls into.
- **22** — SSH, for logging into machines. The *only* thing passwords are for.
- **`172.20.95.127`** — the robot. **`172.20.95.106`** — the Pi. Both on the lab
  network. (These can change — see point 4 below.)
- **Two machines, both log in as user `nao`, but two different passwords.** This
  trips everyone up.

## Things to know before you touch anything

1. **Two Pythons, don't mix them.** Code in `nao/` runs on the robot and must be
   **old Python 2.7** — no f-strings, no fancy syntax. Code in `server/` is
   modern Python 3.11+. Write a modern feature in `nao/` and it works on your
   laptop but crashes on the robot.

2. **The robot is a dumb terminal.** When something misbehaves, 9 times out of 10
   the bug is in the *server* (`server/`), not the robot. The robot just records
   and plays.

3. **`.env` is the control panel, and it's secret.** Every setting — keys, IPs,
   passwords — lives in `.env`. It's deliberately kept out of git (never commit
   it). Never type an IP or key directly into code; read it from `.env`.

4. **The robot's IP drifts.** It's handed out by the network and can change. If
   you can't reach it, press the robot's chest button once — it says its current
   IP out loud.

5. **Everything must be on the same WiFi.** Robot, Pi, and your laptop have to
   share a network to talk. (Common wall: your laptop is on the general campus
   WiFi while the robot and Pi are on a separate lab network — different
   networks can't reach each other.)

6. **Stale code caches on the robot.** Old Python leaves cached `.pyc` files that
   can silently run instead of your edits. `./run.sh` clears them automatically —
   but if a robot change "does nothing," this is usually why.

7. **There's a safety gate you can't bypass.** Before the AI ever sees a message,
   a crisis-check runs (self-harm etc. → 988 hotline). It's intentional and runs
   first, every time.

8. **You don't need the robot to work.** The simulator (`sim/`) runs the whole
   pipeline on your laptop. The only thing it needs to give *real* answers is an
   OpenAI key in `.env`.

## Passwords — the honest truth

A password can't be "gotten" or looked up. It was never stored in readable form
anywhere — only a scrambled one-way hash lives on the machine. So there are
exactly two ways to have one:

- **Someone tells you** — whoever set the machine up (a labmate or admin). This
  is the fast, non-destructive path. Ask for all of them at once: robot
  password, Pi password, and the WiFi name + password the robot/Pi use.
- **You reset it** — which needs *physical* access. For the Pi you can pull its
  microSD card and reset locally. For the NAO there's no gentle reset: the only
  way without the current password is a factory reflash from USB, which **wipes
  the robot** (its Choregraphe autostart behavior, `launch_nao_assist.sh`, face
  training, and WiFi config are all lost — the `nao/` code itself is safe because
  it's in this repo). Treat a NAO reflash as a last resort.

## The folder tree, in plain words

```
nao-sagecbt/
│
├── nao/                  ← THE ROBOT'S CODE (Python 2.7, copied to the robot)
│   ├── main.py               the starting point — wakes up, runs the loop
│   ├── wake_listener.py      listens for "hey NAO" + which mode you want
│   ├── conversation.py       the core loop: record → send → speak → move
│   ├── audio_handler.py      recording from the mic
│   ├── config.py             reads robot settings from .env
│   └── utils/                camera, face recognition, motion helpers
│
├── server/               ← THE BRAIN'S CODE (Python 3.11+, runs on Pi or laptop)
│   ├── server.py             old entry point (Flask)
│   ├── app_ws.py             new entry point (the one the Pi runs)
│   ├── safety.py             the crisis gate (runs first, always)
│   ├── session.py            remembers who you are between turns
│   ├── config.py             reads server settings from .env
│   ├── agents/               the different "personalities":
│   │   ├── router.py             decides which agent handles you
│   │   ├── chat.py               general conversation
│   │   ├── chatbot.py            Morgan CS questions (looks things up)
│   │   ├── skills.py             time, weather, timers, to-dos
│   │   ├── therapist.py          empathetic / mental-health support
│   │   ├── cbt_coach.py          guided thought exercises
│   │   └── grounding_coach.py    breathing / grounding exercises
│   └── tools/                things the agents can DO (move the robot,
│                             search the knowledge base, read emotions)
│
├── sim/                  ← THE SIMULATOR (run the whole thing on your laptop)
│   ├── live_nao.py           talk with your real mic + speakers
│   └── scenarios/            scripted headless test conversations
│
├── docs/                 ← design notes, guides (this file lives here)
├── .env                  ← YOUR SECRETS + SETTINGS (never commit this)
├── run.sh                ← the dev launcher (laptop-as-server)
└── CLAUDE.md / README.md ← project overview
```

**The mental model to hold onto:** `nao/` is the body's reflexes, `server/` is
the mind, `sim/` lets you run the mind without the body, `.env` is the settings
behind everything, and the Pi is just a small computer whose whole job is to keep
`server/` running so the robot always has a mind to talk to.
