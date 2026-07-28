# -*- coding: utf-8 -*-
"""
Configuration for NAO ⇄ OpenAI integration.
Reads everything from environment variables.
"""

import os

# NAO connection settings.
# This module only ever runs ON the robot, and naoqi listens on the robot's
# own loopback -- so the default must be 127.0.0.1, never a routable address.
# A LAN default breaks on every DHCP lease change (and on the Choregraphe
# autostart after a reboot, which sets no env at all).
NAO_IP   = os.environ.get("NAO_IP", "127.0.0.1")
NAO_PORT = int(os.environ.get("NAO_PORT", "9559"))

# OpenAI settings

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER", "openai")

# Server IP (for NAO-side scripts to reach the Flask server)
SERVER_IP = os.environ.get("SERVER_IP", "172.20.95.106")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "5050"))

# Shared secret sent as X-NAO-Secret on every HTTP request and as
# {"secret": "..."} on the /chat_realtime WebSocket handshake. Must match
# the server's NAO_SHARED_SECRET. Empty = OPEN mode (server warns at boot).
NAO_SHARED_SECRET = os.environ.get("NAO_SHARED_SECRET", "")

# Realtime chat/morgan VAD tuning. NAO is a far-field robot mic, so the
# threshold is lower than the OpenAI default 0.5.
REALTIME_VAD_THRESHOLD = float(os.environ.get("REALTIME_VAD_THRESHOLD", "0.55"))
REALTIME_VAD_PREFIX_MS = int(os.environ.get("REALTIME_VAD_PREFIX_MS", "500"))
REALTIME_VAD_SILENCE_MS = int(os.environ.get("REALTIME_VAD_SILENCE_MS", "700"))

# Echo gate for Realtime chat. NAO's microphones hear NAO's own speaker, so
# acoustic barge-in is disabled by default in Realtime. Head-touch still
# interrupts instantly. Enable acoustic barge only after tuning on the robot.
REALTIME_ECHO_GATE_ENABLED = os.environ.get("REALTIME_ECHO_GATE_ENABLED", "1") == "1"
REALTIME_ECHO_TAIL_MS = int(os.environ.get("REALTIME_ECHO_TAIL_MS", "1400"))
REALTIME_ACOUSTIC_BARGE_ENABLED = os.environ.get("REALTIME_ACOUSTIC_BARGE_ENABLED", "0") == "1"
REALTIME_BARGE_THRESHOLD = float(os.environ.get("REALTIME_BARGE_THRESHOLD", "9500"))
REALTIME_BARGE_SUSTAIN_MS = int(os.environ.get("REALTIME_BARGE_SUSTAIN_MS", "600"))
REALTIME_BARGE_DEADZONE_MS = int(os.environ.get("REALTIME_BARGE_DEADZONE_MS", "1200"))

# Safety: proactive greetings make NAO speak without an explicit user command.
# Keep this off unless it is intentionally enabled for a demo/research run.
PROACTIVE_GREET_ENABLED = os.environ.get("PROACTIVE_GREET_ENABLED", "0") == "1"

# Barge-in while NAO is speaking. The threshold is intentionally higher than
# normal VAD because NAO's microphones hear its own speaker output.
BARGE_ENABLED = os.environ.get("BARGE_ENABLED", "0") == "1"
# Threshold is intentionally high; without acoustic echo cancellation NAO's own
# speaker output bleeds 5000-10000 into the front mic. 8000 + 1.2s deadzone +
# 500ms sustain means only sustained loud user voice can interrupt.
BARGE_THRESHOLD = float(os.environ.get("BARGE_THRESHOLD", "8000"))
BARGE_SUSTAIN_MS = int(os.environ.get("BARGE_SUSTAIN_MS", "500"))
BARGE_DEADZONE_MS = int(os.environ.get("BARGE_DEADZONE_MS", "1200"))
BARGE_POLL_MS = int(os.environ.get("BARGE_POLL_MS", "30"))

# Camera snap per turn. Off by default for low latency. Set IMAGE_PER_TURN=1
# to enable; the therapist agent's observe_face tool can request snaps anyway.
IMAGE_PER_TURN = os.environ.get("IMAGE_PER_TURN", "0") == "1"

# Audio storage
AUDIO_SAVE_PATH = os.environ.get("AUDIO_SAVE_PATH", "./audio/")
