# -*- coding: utf-8 -*-
"""
NaoAudioStreamer — live PCM mic streaming on NAO V6.

This is the Phase 1 replacement for ``audio_handler.record_audio``'s
file-based ``ALAudioRecorder`` path. It opens a *live* stream of 16 kHz mono
PCM16 from the front microphone via the naoqi ``ALAudioDevice`` subscriber
protocol and hands 20 ms slices to a queue that the WS sender thread drains.

Why an ALModule
---------------
``ALAudioRecorder.startMicrophonesRecording`` writes a WAV to disk and hands
the *path* back when you ``stopMicrophonesRecording``. Live streaming
requires the **callback** path: a class that subclasses ``naoqi.ALModule``,
calls ``ALAudioDevice.subscribe(name)``, and exposes a method named
``processRemote(nbOfChannels, nbOfSamplesByChannel, timeStamp, inputBuffer)``.
The audio device pushes ~170 ms of audio per call (NAOqi default) which we
re-slice into 20 ms chunks before queuing.

Channel constants (per Aldebaran ``ALAudioDevice`` docs)
--------------------------------------------------------
``setClientPreferences(name, sampleRate, deinterleave_or_channels, deinterleave)``
The signature varies between NAOqi 2.5 (``(name, sampleRate, channels, deinterleave)``)
and earlier docs (``(name, sampleRate, deinterleave_mask, ...)``). Both
accept the **channel index constants**:

    ALL    = 0   (4 channels interleaved)
    LEFT   = 1
    RIGHT  = 2
    FRONT  = 3
    REAR   = 4

We use FRONT (3) because that mic is closest to the user when the robot is
oriented toward them and matches ``audio_handler.CHANNELS_MASK = (0,0,1,0)``.

Firmware fallback
-----------------
If ``ALAudioDevice.subscribe`` raises (some firmware revisions ship without
the public subscriber API enabled), we fall back to ``ALAudioRecorder``
short-fragment recording: 250 ms WAV files, read off disk and pushed
through the same queue. Latency is ~250 ms higher; documented in
``docs/spike_results.md``. This module exposes which mode is active via
``streamer.mode`` (``"alaudio_device"`` or ``"alaudio_recorder_fragment"``).

Public API
----------
    streamer = NaoAudioStreamer(broker_ip, broker_port, broker)
    streamer.start()                  # begin streaming
    for seq, ts_ms, b64 in streamer.read_chunks():
        ws.send_audio_chunk(seq, ts_ms, b64)
    streamer.gate(True)               # mute mic during TTS
    streamer.gate(False)              # un-mute when TTS done
    streamer.stop()                   # tear down, drain queue, reset seq

The class is broker-aware so the caller (``ws_client.py``) constructs the
broker once and threads it in.
"""
from __future__ import print_function

import base64
import audioop
import logging
import os
import sys
import threading
import time
import traceback
import wave

# Standard library queue is a single module renamed in py3. Robot is py2.7.
try:
    import Queue as _queue
except ImportError:  # pragma: no cover — only here so py3 import doesn't crash
    import queue as _queue

# naoqi is only present on the robot. On a developer Mac running unit tests
# the import will fail; we guard so the module is at least importable for
# syntax / smoke checks.
try:
    import naoqi  # noqa: F401  — used for ALModule + ALProxy
    from naoqi import ALModule, ALProxy
    _NAOQI_AVAILABLE = True
except ImportError:
    naoqi = None
    _NAOQI_AVAILABLE = False

    class ALModule(object):  # noqa: D401
        """Stub for off-robot import. Real ALModule comes from naoqi."""
        def __init__(self, name):
            self.name = name

    def ALProxy(*args, **kwargs):  # noqa: D401
        raise RuntimeError("ALProxy unavailable: naoqi not importable here")


# ── Audio constants ─────────────────────────────────────────────────────────
SAMPLE_RATE_HZ = 16000
SAMPLE_WIDTH = 2                # 16-bit signed
CHANNELS = 1                    # mono (front mic only)
BYTES_PER_SECOND = SAMPLE_RATE_HZ * SAMPLE_WIDTH * CHANNELS  # 32_000
CHUNK_MS = 20
BYTES_PER_CHUNK = BYTES_PER_SECOND * CHUNK_MS // 1000        # 640

# Queue cap. 200 chunks * 20 ms = 4 000 ms = 4 s of audio buffered. Bigger
# than that means the WS sender is so backed up the conversation already
# fell apart; drop the oldest frames so we don't OOM and so the next
# successful flush is "now" not "5 seconds ago".
QUEUE_MAXSIZE = 200

# Fragment-mode (fallback) settings.
FRAGMENT_MS = 250
FRAGMENT_DIR = "/home/nao/recordings/_stream"
# Which of NAO's four mics to record. Measured on the robot with all four
# recorded simultaneously during 12s of speech:
#
#   LEFT   RMS=1045  PEAK=6135  -> "one two three, one two three, ..."
#   RIGHT  RMS=969   PEAK=5952  -> "One, two, three, ..."
#   FRONT  RMS=730   PEAK=4483  -> "一、二、三,一、二、三..."   <- Chinese!
#   REAR   RMS=705   PEAK=4880  -> "One two three, ..."
#
# FRONT was the weakest of the four and the only one we recorded. Whisper
# rendered that channel's English as Chinese characters — what a marginal
# signal does to STT. LEFT is ~43% hotter and transcribes cleanly, so it is
# the default. Override with MIC_CHANNEL=left|right|front|rear|all to
# A/B another mic on the robot without a redeploy.
_MIC_MASKS = {
    "left":  (1, 0, 0, 0),
    "right": (0, 1, 0, 0),
    "front": (0, 0, 1, 0),
    "rear":  (0, 0, 0, 1),
    "all":   (1, 1, 1, 1),
}
_MIC_CHANNEL = os.environ.get("MIC_CHANNEL", "left").strip().lower()
# An unknown value must not silently record a dead channel — fall back to
# the measured-best mic rather than whatever a typo maps to.
FRAGMENT_CHANNELS_MASK = _MIC_MASKS.get(_MIC_CHANNEL, _MIC_MASKS["left"])
FRAGMENT_HEADER_WAIT_S = 1.20
FRAGMENT_STALL_RESTART_S = 2.0
FRAGMENT_ZERO_PCM_RESTART_S = 6.0
FRAGMENT_ZERO_PCM_LOG_S = 1.0
FRAGMENT_LOW_PCM_RESTART_S = 8.0
FRAGMENT_LOW_PCM_LOG_S = 2.0
FRAGMENT_LOW_PCM_RESTART_COOLDOWN_S = 20.0
FRAGMENT_LOW_PCM_MAX = 64
FRAGMENT_LOW_PCM_RMS = 4

# ALAudioDevice channel index constants (Aldebaran docs).
AL_CHANNEL_ALL = 0
AL_CHANNEL_LEFT = 1
AL_CHANNEL_RIGHT = 2
AL_CHANNEL_FRONT = 3
AL_CHANNEL_REAR = 4

DEFAULT_MODULE_NAME = "NaoAudioStreamer"

logger = logging.getLogger(__name__)


def _b64_text(raw_bytes):
    """Return a *text* base64 string suitable for JSON envelopes."""
    encoded = base64.b64encode(raw_bytes)
    if isinstance(encoded, bytes):
        try:
            encoded = encoded.decode("ascii")
        except Exception:
            encoded = str(encoded)
    return encoded


def _now_ms():
    """Wall-clock ms (float). Robot has no monotonic clock that returns ms."""
    return time.time() * 1000.0


# ── Main class ──────────────────────────────────────────────────────────────
class NaoAudioStreamer(ALModule):
    """ALModule that subscribes to ALAudioDevice front-mic frames.

    Parameters
    ----------
    broker_ip : str
        IP that ``ALBroker`` is bound to. For an in-process broker started by
        the caller, pass the broker's IP (typically ``"0.0.0.0"`` for local).
    broker_port : int
        Port the broker listens on. The streamer uses this only when
        instantiating ``ALProxy``.
    nao_ip : str
        Address of the *NAO* ALMain broker (where ``ALAudioDevice`` lives).
        On the robot this is normally ``"127.0.0.1"`` or the value from
        ``config.NAO_IP``.
    nao_port : int, default 9559
    name : str, default ``"NaoAudioStreamer"``
        Naoqi module name. MUST be globally unique inside the broker process
        and match the variable name we expose into ``__main__`` (see
        ``_register_in_main``). Multiple instances must use distinct names.
    queue_maxsize : int, default 200
    chunk_ms : int, default 20
    """

    def __init__(self,
                 broker_ip="0.0.0.0",
                 broker_port=0,
                 nao_ip=None,
                 nao_port=9559,
                 name=DEFAULT_MODULE_NAME,
                 queue_maxsize=QUEUE_MAXSIZE,
                 chunk_ms=CHUNK_MS):
        # ALModule.__init__ registers the module name with the running
        # broker so the C++ audio device can dispatch processRemote() back
        # to this instance. On some NAO firmware revisions (observed on
        # this V6 head: 2.8.x) the SWIG-generated autoBind chain inside
        # ALDocable.__init__ raises AttributeError because the metaclass
        # __getattr__ falls through to object.__getattr__ which doesn't
        # exist on Python 2.7. We catch it here, mark the module as
        # "live-stream incapable," and force fragment fallback in start().
        # That's the documented Phase 0.5 risk path from the PRD.
        self._module_init_ok = False
        try:
            ALModule.__init__(self, name)
            self._module_init_ok = True
        except (AttributeError, RuntimeError) as exc:
            logger.warning(
                "[audio_module] ALModule.__init__ failed (%s: %s). "
                "Live-PCM subscriber unavailable on this firmware; will "
                "use file-fragment fallback.",
                type(exc).__name__, exc,
            )

        self.module_name = name
        self.broker_ip = broker_ip
        self.broker_port = broker_port
        # NAO_IP defaults — pull from env if caller didn't provide.
        self.nao_ip = nao_ip or os.environ.get("NAO_IP", "127.0.0.1")
        self.nao_port = int(nao_port)
        self.chunk_ms = int(chunk_ms)
        self.bytes_per_chunk = (
            SAMPLE_RATE_HZ * SAMPLE_WIDTH * CHANNELS * self.chunk_ms // 1000
        )
        if self.bytes_per_chunk <= 0:
            raise ValueError("chunk_ms produced zero bytes_per_chunk")

        # Queue holds (seq, ts_ms, b64_str) triples.
        self._queue = _queue.Queue(maxsize=int(queue_maxsize))
        self._queue_lock = threading.Lock()

        # State flags.
        self._streaming = False
        self._gate_closed = False
        self._subscribed = False
        self._seq = 0
        self._dropped = 0
        # PCM tail kept across processRemote() calls so a non-multiple-of-20ms
        # incoming buffer aligns cleanly to 20 ms chunks.
        self._tail = b""

        # Proxies — created lazily on start().
        self._audio_dev = None
        self._recorder = None

        # Mode set by start(). One of:
        #   "alaudio_device"            — preferred subscriber path
        #   "alaudio_recorder_fragment" — disk-fragment fallback
        #   None                        — not started
        self.mode = None

        # Fallback-mode worker.
        self._fragment_thread = None
        self._fragment_stop = threading.Event()

        # Register the instance into __main__ so the broker can resolve
        # ``self.module_name`` for processRemote dispatch. This is the
        # canonical naoqi ALModule pattern. Only useful when ALModule init
        # actually succeeded; skip it on firmwares where we fell back to
        # fragment mode (no remote dispatch happens there).
        if self._module_init_ok:
            self._register_in_main()

    # ── Lifecycle ───────────────────────────────────────────────────────────
    def _register_in_main(self):
        """Make the streamer reachable as ``__main__.<module_name>``.

        The naoqi ALBroker resolves an ALModule by looking up
        ``__main__.<name>`` when it dispatches a remote call. If the
        instance isn't there, ``processRemote`` will never be invoked.
        """
        try:
            main_module = sys.modules.get("__main__")
            if main_module is not None:
                setattr(main_module, self.module_name, self)
        except Exception:
            # Non-fatal — caller may have set this up themselves.
            pass

    def _connect_proxies(self):
        """Open ALAudioDevice + ALAudioRecorder proxies. Called on start()."""
        if not _NAOQI_AVAILABLE:
            raise RuntimeError(
                "naoqi import failed; cannot connect ALAudioDevice. "
                "This module is intended to run on the robot."
            )
        self._audio_dev = ALProxy("ALAudioDevice", self.nao_ip, self.nao_port)
        # Recorder proxy is only used in fallback mode but cheap to obtain.
        try:
            self._recorder = ALProxy(
                "ALAudioRecorder", self.nao_ip, self.nao_port
            )
        except Exception:
            self._recorder = None

    def start(self):
        """Begin streaming. Picks the best available capture mode.

        Returns the chosen mode string (also stored on ``self.mode``).
        """
        if self._streaming:
            return self.mode

        self._connect_proxies()
        self._seq = 0
        self._dropped = 0
        self._tail = b""
        self._gate_closed = False

        # Live subscriber path is only viable when ALModule.__init__ succeeded.
        # On firmwares where SWIG autoBind broke that init, we skip straight
        # to the fragment recorder so we don't waste 1-2 s on the ALAudioDevice
        # round trip we already know will fail.
        print("[mic_trace] using_almodule_subscriber={0}".format(
            self._module_init_ok))
        sys.stderr.flush()
        if self._module_init_ok:
            try:
                self._setup_subscriber()
                self._do_subscribe()
                self.mode = "alaudio_device"
                self._streaming = True
                logger.info(
                    "[audio_module] subscribed to ALAudioDevice "
                    "(rate=%d hz, chunk=%d ms, mode=%s)",
                    SAMPLE_RATE_HZ, self.chunk_ms, self.mode,
                )
                return self.mode
            except Exception as exc:
                logger.warning(
                    "[audio_module] ALAudioDevice subscribe failed (%s): %s — "
                    "falling back to fragment recorder",
                    type(exc).__name__, exc,
                )
        else:
            logger.info(
                "[audio_module] skipping ALAudioDevice subscribe (ALModule "
                "init failed at construct time); going straight to fragment "
                "fallback."
            )

        print("[mic_trace] using_fragment_recorder=true (recorder_proxy={0})".format(
            self._recorder is not None))
        sys.stderr.flush()

        # Fallback: short-fragment recorder.
        try:
            self._start_fragment_recorder()
            self.mode = "alaudio_recorder_fragment"
            self._streaming = True
            logger.info(
                "[audio_module] running in FRAGMENT fallback mode "
                "(fragment_ms=%d)", FRAGMENT_MS,
            )
            return self.mode
        except Exception as exc:
            self._streaming = False
            self.mode = None
            raise RuntimeError(
                "Both ALAudioDevice subscribe and ALAudioRecorder fragment "
                "fallback failed: " + repr(exc)
            )

    def stop(self):
        """Stop streaming. Drains the queue and resets sequence numbers."""
        if not self._streaming:
            return

        self._streaming = False

        if self.mode == "alaudio_device":
            try:
                self._do_unsubscribe()
            except Exception as exc:
                logger.warning("[audio_module] unsubscribe on stop failed: %s", exc)
        elif self.mode == "alaudio_recorder_fragment":
            self._stop_fragment_recorder()

        # Drain queue.
        with self._queue_lock:
            try:
                while True:
                    self._queue.get_nowait()
            except _queue.Empty:
                pass

        self._seq = 0
        self._tail = b""
        self.mode = None
        logger.info(
            "[audio_module] stopped. dropped_frames=%d", self._dropped,
        )

    def gate(self, closed):
        """Mute / unmute the mic (idempotent).

        Called from the WS receive thread when TTS playback starts/ends.
        Closing the gate ``unsubscribe()``s so the audio device stops
        delivering frames at all (also stops feeding our process buffer);
        the next ALAudioDevice frame boundary is at most 20 ms away so
        worst-case leak after this call is ~20 ms. That comfortably beats
        the Phase 1 < 50 ms target.

        In fragment-mode the gate stops/starts the recorder; latency for
        the gate to take effect is up to one fragment (250 ms).
        """
        closed = bool(closed)
        if closed == self._gate_closed:
            return  # idempotent — already in requested state

        self._gate_closed = closed

        if not self._streaming:
            # Gate state stored, will apply on next start().
            return

        if self.mode == "alaudio_device":
            if closed:
                try:
                    self._do_unsubscribe()
                except Exception as exc:
                    logger.warning("[audio_module] gate-close unsubscribe failed: %s", exc)
            else:
                try:
                    self._do_subscribe()
                except Exception as exc:
                    logger.warning("[audio_module] gate-open subscribe failed: %s", exc)
                    # Try one re-setup in case prefs got dropped.
                    try:
                        self._setup_subscriber()
                        self._do_subscribe()
                    except Exception as exc2:
                        logger.error(
                            "[audio_module] gate-open recovery failed: %s", exc2
                        )

        elif self.mode == "alaudio_recorder_fragment":
            if closed:
                self._stop_fragment_recorder()
            else:
                # Re-arm the fragment loop. Reset the stop flag and spin a new
                # worker; the previous one has already exited.
                self._fragment_stop = threading.Event()
                try:
                    self._start_fragment_recorder()
                except Exception as exc:
                    logger.error(
                        "[audio_module] fragment re-arm after gate-open failed: %s",
                        exc,
                    )

    # ── ALAudioDevice subscriber path ───────────────────────────────────────
    def _setup_subscriber(self):
        """Tell ALAudioDevice what audio format we want."""
        # Some firmware exposes ``setClientPreferences`` with positional
        # ``(name, sampleRate, channels, deinterleave)``. Using channels=3
        # (FRONT mic) and deinterleave=0 to receive only that single channel
        # in the inputBuffer. We request 16 kHz mono.
        self._audio_dev.setClientPreferences(
            self.module_name,
            SAMPLE_RATE_HZ,
            AL_CHANNEL_FRONT,
            0,
        )

    def _do_subscribe(self):
        if self._subscribed:
            return
        self._audio_dev.subscribe(self.module_name)
        self._subscribed = True

    def _do_unsubscribe(self):
        if not self._subscribed:
            return
        try:
            self._audio_dev.unsubscribe(self.module_name)
        finally:
            # Even if unsubscribe raised, we don't want to keep believing
            # we're subscribed — the next start() will re-attempt.
            self._subscribed = False

    def processRemote(self, nbOfChannels, nbOfSamplesByChannel, timeStamp, inputBuffer):
        """ALAudioDevice callback. Slice into 20 ms chunks and enqueue.

        ``inputBuffer`` is a Python ``str`` (py2) of raw little-endian PCM16
        samples for one channel (we asked for FRONT only). Length =
        nbOfSamplesByChannel * 2 bytes.
        """
        if not self._streaming or self._gate_closed:
            return

        try:
            # On py2.7 ``inputBuffer`` is a str; on py3 it could be bytes.
            data = inputBuffer if isinstance(inputBuffer, (bytes, str)) else bytes(inputBuffer)
            if isinstance(data, str):
                # py2: str is bytes; py3: this branch shouldn't run on robot.
                pcm = data
            else:
                pcm = data

            # Some firmwares deliver multi-channel even when prefs ask for one.
            # Handle that defensively: pick the FRONT-equivalent channel.
            if nbOfChannels and nbOfChannels > 1:
                # Interleaved ch0..chN-1 samples. We requested FRONT (idx 3),
                # but if more than 1 channel arrived it's because firmware
                # ignored our prefs and we got ALL channels. Trust the order
                # documented by ALAudioDevice: 0=LEFT,1=RIGHT,2=FRONT,3=REAR.
                pcm = self._extract_channel(
                    pcm, nbOfChannels, channel_index=2  # 2 = FRONT in 4-ch ALL
                )

            # Concatenate with the previous tail so 20 ms boundaries align.
            buf = self._tail + pcm
            sliced = []
            offset = 0
            while offset + self.bytes_per_chunk <= len(buf):
                sliced.append(buf[offset:offset + self.bytes_per_chunk])
                offset += self.bytes_per_chunk
            self._tail = buf[offset:]

            # Build (seq, ts_ms, b64) triples and push.
            base_ts_ms = _now_ms()
            for i, chunk in enumerate(sliced):
                seq = self._seq
                self._seq = (self._seq + 1) & 0xFFFFFFFF  # 32-bit wrap-around
                # Per-chunk timestamp: best estimate, base_ts is "now" at
                # callback receipt; subtract back from end for chunk start.
                chunk_ts_ms = base_ts_ms + (i * self.chunk_ms)
                self._enqueue(seq, chunk_ts_ms, chunk)

        except Exception:
            # processRemote runs on a naoqi worker thread; any uncaught
            # exception will be eaten silently by the broker. Log explicitly.
            logger.error(
                "[audio_module] processRemote failed:\n%s",
                traceback.format_exc(),
            )

    def _extract_channel(self, interleaved_pcm, n_channels, channel_index):
        """Pull one channel out of an interleaved PCM16 buffer.

        Each frame is ``n_channels * 2`` bytes; the wanted channel sits at
        ``channel_index * 2 .. channel_index * 2 + 2`` within that frame.
        """
        if n_channels <= 0 or channel_index < 0 or channel_index >= n_channels:
            return interleaved_pcm
        frame_bytes = n_channels * 2
        if frame_bytes == 0 or len(interleaved_pcm) < frame_bytes:
            return interleaved_pcm
        # Slice every nth frame. Fast enough for ~170 ms (~5440 bytes) at
        # 20 Hz callback rate; no numpy on robot, so use bytes concat.
        # On py2.7 str is bytes so b"".join works on str slices identically.
        chan_start = channel_index * 2
        parts = []
        for i in range(0, len(interleaved_pcm) - frame_bytes + 1, frame_bytes):
            parts.append(interleaved_pcm[i + chan_start:i + chan_start + 2])
        if not parts:
            return interleaved_pcm
        return type(parts[0])().join(parts)

    # ── ALAudioRecorder fallback path ───────────────────────────────────────
    def _start_fragment_recorder(self):
        """Spin a thread that records 250 ms WAVs and slices them to 20 ms."""
        if self._recorder is None:
            raise RuntimeError("ALAudioRecorder proxy unavailable for fallback")
        try:
            os.makedirs(FRAGMENT_DIR)
        except OSError:
            pass  # already exists

        # Stdlib logger has no handler in this process; print to stderr so
        # the operator can see fragment-mode actually engaged.
        print("[audio_module] fragment recorder STARTING (dir={0}, fragment_ms={1})".format(
            FRAGMENT_DIR, FRAGMENT_MS))
        sys.stderr.flush()

        self._fragment_stop.clear()
        self._fragment_thread = threading.Thread(target=self._fragment_worker)
        self._fragment_thread.daemon = True
        self._fragment_thread.start()

    def _parse_stream_wav_header(self, path):
        """Return WAV stream metadata once the recorder header is usable."""
        try:
            with open(path, "rb") as fh:
                hdr = fh.read(512)
        except Exception as exc:
            return None, "read_failed:{0}".format(exc)

        if len(hdr) < 44:
            return None, "too_small:{0}".format(len(hdr))
        if hdr[:4] != b"RIFF" or hdr[8:12] != b"WAVE":
            return None, "bad_magic"

        try:
            import struct as _struct
            pos = 12
            nchan = None
            width = None
            data_offset = None
            while pos + 8 <= len(hdr):
                chunk_id = hdr[pos:pos + 4]
                chunk_size = _struct.unpack("<I", hdr[pos + 4:pos + 8])[0]
                body = pos + 8

                if chunk_id == b"fmt ":
                    if body + 16 > len(hdr):
                        return None, "fmt_incomplete"
                    fmt = _struct.unpack("<HHIIHH", hdr[body:body + 16])
                    nchan = int(fmt[1])
                    bits_per_sample = int(fmt[5])
                    width = max(1, bits_per_sample // 8)
                elif chunk_id == b"data":
                    data_offset = body
                    break

                pos = body + chunk_size + (chunk_size & 1)
        except Exception as exc:
            return None, "parse_failed:{0}".format(exc)

        if nchan is None:
            return None, "missing_fmt"
        if data_offset is None:
            return None, "missing_data"
        if width != SAMPLE_WIDTH:
            return None, "bad_sample_width:{0}".format(width)
        return {
            "nchan": nchan,
            "width": width,
            "header_size": data_offset,
        }, None

    def _wait_for_stream_wav_header(self, path, timeout_s=FRAGMENT_HEADER_WAIT_S):
        """Wait briefly for ALAudioRecorder to finish writing a valid header."""
        deadline = time.time() + timeout_s
        last_error = None
        while (time.time() < deadline and
               not self._fragment_stop.is_set()):
            parsed, last_error = self._parse_stream_wav_header(path)
            if parsed is not None:
                return parsed, None
            time.sleep(0.05)
        return None, last_error or "timeout"

    def _restart_fragment_recording(self, path, reason):
        """Stop/start the continuous recorder and return fresh WAV metadata."""
        print("[audio_module] restarting recorder reason={0}".format(reason))
        sys.stderr.flush()
        try:
            self._recorder.stopMicrophonesRecording()
        except Exception as exc:
            print("[audio_module] stopMicrophonesRecording on restart: {0}".format(exc))
            sys.stderr.flush()
        try:
            if os.path.exists(path):
                os.unlink(path)
        except Exception:
            pass
        self._tail = b""
        try:
            self._recorder.startMicrophonesRecording(
                path, "wav", SAMPLE_RATE_HZ, FRAGMENT_CHANNELS_MASK,
            )
            print("[audio_module] recorder restarted reason={0}".format(reason))
            sys.stderr.flush()
        except Exception as exc:
            print("[audio_module] startMicrophonesRecording on restart: {0}".format(exc))
            sys.stderr.flush()
            return None

        parsed, err = self._wait_for_stream_wav_header(path)
        if parsed is None:
            print("[audio_module] recorder restart produced invalid WAV header reason={0} err={1}".format(
                reason, err))
            sys.stderr.flush()
        return parsed

    def _stop_fragment_recorder(self):
        self._fragment_stop.set()
        try:
            self._recorder.stopMicrophonesRecording()
        except Exception:
            pass
        if self._fragment_thread is not None:
            self._fragment_thread.join(timeout=1.0)
        self._fragment_thread = None

    def _fragment_worker(self):
        """One continuous ALAudioRecorder recording + tail the growing WAV.

        Rapid start/stop (250 ms cycles) does NOT work on this NAO V6
        firmware — every short recording produced a 44-byte header-only
        WAV with zero audio frames. ALAudioRecorder needs ~300-500 ms of
        spin-up time before it actually captures samples.

        Instead we start ONE long recording on engage, then poll the file
        size every ~50 ms and read whatever new bytes appeared. The WAV
        writer flushes data to disk continuously while recording, so the
        file grows in real time. Stopping happens on disengage / shutdown.
        """
        print("[audio_module] fragment_worker thread alive (continuous mode)")
        sys.stderr.flush()
        big_path = os.path.join(FRAGMENT_DIR, "stream.wav")
        # Clean any leftover file so we start fresh.
        try:
            if os.path.exists(big_path):
                os.unlink(big_path)
        except Exception:
            pass
        # Stop any prior recording, then start the long one.
        try:
            self._recorder.stopMicrophonesRecording()
        except Exception:
            pass
        try:
            self._recorder.startMicrophonesRecording(
                big_path, "wav", SAMPLE_RATE_HZ, FRAGMENT_CHANNELS_MASK,
            )
            print("[audio_module] continuous recording -> {0}".format(big_path))
            sys.stderr.flush()
        except Exception as exc:
            print("[audio_module] startMicrophonesRecording FAILED: {0}: {1}".format(
                type(exc).__name__, exc))
            sys.stderr.flush()
            return

        # Wait for ALAudioRecorder to spin up + write the WAV header.
        header, header_err = self._wait_for_stream_wav_header(big_path)
        if header is None:
            print("[audio_module] invalid WAV header after start err={0}".format(
                header_err))
            sys.stderr.flush()
            header = self._restart_fragment_recording(big_path, "bad_header_initial")
            if header is None:
                return

        # Read the WAV header once so we know nchan / width before tailing.
        nchan = header["nchan"]
        width = header["width"]
        header_size = header["header_size"]
        print("[audio_module] WAV header: nchan={0} width={1} data_offset={2}".format(
            nchan, width, header_size))
        sys.stderr.flush()

        # start() flips _streaming immediately after spawning this worker.
        # If the recorder writes its header very quickly, avoid exiting the
        # loop before that flag is visible.
        stream_wait_deadline = time.time() + 2.0
        while (not self._streaming and
               not self._fragment_stop.is_set() and
               time.time() < stream_wait_deadline):
            time.sleep(0.01)
        if not self._streaming:
            return

        sample_stride = nchan * width
        front_idx = AL_CHANNEL_FRONT - 1  # 0-based
        if front_idx >= nchan:
            front_idx = 0

        last_offset = header_size
        first_pcm_logged = False
        last_size_check = time.time()
        stall_started_at = None  # wall-clock when stall first detected
        zero_pcm_started_at = None
        zero_pcm_last_log = 0.0
        low_pcm_started_at = None
        low_pcm_last_log = 0.0
        low_pcm_last_restart_at = 0.0
        # Tighter than 5s so a user speaking through a stall only loses
        # ~2s of audio instead of ~5s. Real ALAudioRecorder wedges
        # always last more than 2s, so false-positive restarts are rare.
        while not self._fragment_stop.is_set() and self._streaming:
            if self._gate_closed:
                time.sleep(0.05)
                continue
            try:
                cur_size = os.path.getsize(big_path)
            except Exception:
                time.sleep(0.05)
                continue
            if cur_size <= last_offset:
                # Once a second, log if file isn't growing — that means
                # ALAudioRecorder isn't actually capturing.
                now = time.time()
                if stall_started_at is None:
                    stall_started_at = now
                if now - last_size_check > 1.0:
                    last_size_check = now
                    stall_s = now - stall_started_at
                    print("[audio_module] WAV not growing (size={0} last={1}) stalled_s={2:.1f}".format(
                        cur_size, last_offset, stall_s))
                    sys.stderr.flush()
                # Recovery: if stall persists past STALL_RESTART_S, kill
                # the recorder and restart it. Real-world cause is the
                # firmware's ALAudioRecorder getting wedged after long
                # uptime; the only reliable cure is a stop/start cycle.
                if now - stall_started_at >= FRAGMENT_STALL_RESTART_S:
                    header = self._restart_fragment_recording(big_path, "stalled")
                    if header is None:
                        return
                    nchan = header["nchan"]
                    width = header["width"]
                    header_size = header["header_size"]
                    sample_stride = nchan * width
                    front_idx = AL_CHANNEL_FRONT - 1
                    if front_idx >= nchan:
                        front_idx = 0
                    last_offset = header_size
                    stall_started_at = None
                    zero_pcm_started_at = None
                    low_pcm_started_at = None
                    first_pcm_logged = False
                    last_size_check = time.time()
                    continue
                time.sleep(0.02)
                continue
            # File grew — clear stall marker.
            stall_started_at = None
            try:
                with open(big_path, "rb") as fh:
                    fh.seek(last_offset)
                    new_bytes = fh.read(cur_size - last_offset)
            except Exception:
                time.sleep(0.05)
                continue
            # Align to a frame boundary so we don't split a sample.
            usable = (len(new_bytes) // sample_stride) * sample_stride
            if usable <= 0:
                time.sleep(0.02)
                continue
            new_bytes = new_bytes[:usable]
            last_offset += usable

            if nchan == 1:
                pcm = new_bytes
            else:
                # Deinterleave: pull every nchan-th sample starting at front_idx.
                buf = bytearray()
                offset = front_idx * width
                while offset + width <= len(new_bytes):
                    buf.extend(new_bytes[offset:offset + width])
                    offset += sample_stride
                pcm = bytes(buf)

            if pcm:
                now = time.time()
                if pcm == (b"\x00" * len(pcm)):
                    if zero_pcm_started_at is None:
                        zero_pcm_started_at = now
                    if now - zero_pcm_last_log >= FRAGMENT_ZERO_PCM_LOG_S:
                        zero_pcm_last_log = now
                        print("[audio_module] exact-zero PCM captured bytes={0} silent_s={1:.1f}".format(
                            len(pcm), now - zero_pcm_started_at))
                        sys.stderr.flush()
                    if now - zero_pcm_started_at >= FRAGMENT_ZERO_PCM_RESTART_S:
                        header = self._restart_fragment_recording(big_path, "exact_zero_pcm")
                        if header is None:
                            return
                        nchan = header["nchan"]
                        width = header["width"]
                        header_size = header["header_size"]
                        sample_stride = nchan * width
                        front_idx = AL_CHANNEL_FRONT - 1
                        if front_idx >= nchan:
                            front_idx = 0
                        last_offset = header_size
                        stall_started_at = None
                        zero_pcm_started_at = None
                        low_pcm_started_at = None
                        low_pcm_last_restart_at = time.time()
                        first_pcm_logged = False
                        last_size_check = time.time()
                        continue
                else:
                    zero_pcm_started_at = None
                    try:
                        pcm_max = audioop.max(pcm, width)
                        pcm_rms = audioop.rms(pcm, width)
                    except Exception:
                        pcm_max = FRAGMENT_LOW_PCM_MAX + 1
                        pcm_rms = FRAGMENT_LOW_PCM_RMS + 1
                    if (pcm_max <= FRAGMENT_LOW_PCM_MAX and
                            pcm_rms <= FRAGMENT_LOW_PCM_RMS):
                        if low_pcm_started_at is None:
                            low_pcm_started_at = now
                        if now - low_pcm_last_log >= FRAGMENT_LOW_PCM_LOG_S:
                            low_pcm_last_log = now
                            print("[audio_module] near-zero PCM captured bytes={0} silent_s={1:.1f} max={2} rms={3}".format(
                                len(pcm), now - low_pcm_started_at,
                                pcm_max, pcm_rms))
                            sys.stderr.flush()
                        can_restart = (
                            now - low_pcm_last_restart_at >=
                            FRAGMENT_LOW_PCM_RESTART_COOLDOWN_S)
                        if (can_restart and
                                now - low_pcm_started_at >=
                                FRAGMENT_LOW_PCM_RESTART_S):
                            header = self._restart_fragment_recording(
                                big_path, "near_zero_pcm")
                            if header is None:
                                return
                            nchan = header["nchan"]
                            width = header["width"]
                            header_size = header["header_size"]
                            sample_stride = nchan * width
                            front_idx = AL_CHANNEL_FRONT - 1
                            if front_idx >= nchan:
                                front_idx = 0
                            last_offset = header_size
                            stall_started_at = None
                            zero_pcm_started_at = None
                            low_pcm_started_at = None
                            low_pcm_last_restart_at = time.time()
                            first_pcm_logged = False
                            last_size_check = time.time()
                            continue
                    else:
                        low_pcm_started_at = None
                if not first_pcm_logged:
                    print("[audio_module] FIRST PCM captured: {0} bytes (file size={1})".format(
                        len(pcm), cur_size))
                    sys.stderr.flush()
                    first_pcm_logged = True
                self._slice_and_enqueue(pcm)
            time.sleep(0.02)

        # Shutdown: stop the recorder.
        try:
            self._recorder.stopMicrophonesRecording()
        except Exception:
            pass
        # Best-effort cleanup of the big file.
        try:
            os.unlink(big_path)
        except Exception:
            pass

    def _read_wav_pcm(self, path):
        """Read 16 kHz mono PCM16 bytes out of a WAV file.

        Handles ALAudioRecorder's quirks:
          - Multi-channel WAVs (NAOqi sometimes ignores the channel mask
            and writes 4-channel; pull channel 2 = front mic).
          - File-flush race: stopMicrophonesRecording is async, the file
            may have only the WAV header when we open it. Caller already
            has a settle delay; this fn just defends against partial files.
        """
        try:
            file_size = os.path.getsize(path)
        except Exception:
            file_size = 0
        if file_size <= 44:  # 44-byte WAV header with no samples
            print("[audio_module] WAV too small ({0} bytes) — partial flush?".format(
                file_size))
            sys.stderr.flush()
            return b""
        try:
            wf = wave.open(path, "rb")
            try:
                nchan = wf.getnchannels()
                width = wf.getsampwidth()
                if width != 2:
                    print("[audio_module] WAV bad sampwidth={0} (need 2)".format(width))
                    sys.stderr.flush()
                    return b""
                raw = wf.readframes(wf.getnframes())
                if not raw:
                    return b""
                if nchan == 1:
                    return raw
                # Multi-channel: take the FRONT mic (channel index 2 of 4).
                # NAO V6 channel order is rear-left, rear-right, front, side.
                # Each frame is `nchan * width` bytes; we copy every (front)
                # sample out.
                front_idx = AL_CHANNEL_FRONT - 1  # 0-based
                if front_idx >= nchan:
                    front_idx = 0
                step = nchan * width
                out = bytearray()
                offset = front_idx * width
                while offset + width <= len(raw):
                    out.extend(raw[offset:offset + width])
                    offset += step
                return bytes(out)
            finally:
                wf.close()
        except Exception as exc:
            print("[audio_module] WAV read error on {0}: {1}".format(path, exc))
            sys.stderr.flush()
            return b""

    def _slice_and_enqueue(self, pcm):
        """Slice raw PCM into 20 ms chunks and enqueue (used by fallback)."""
        buf = self._tail + pcm
        offset = 0
        base_ts_ms = _now_ms()
        chunk_idx = 0
        while offset + self.bytes_per_chunk <= len(buf):
            chunk = buf[offset:offset + self.bytes_per_chunk]
            offset += self.bytes_per_chunk
            seq = self._seq
            self._seq = (self._seq + 1) & 0xFFFFFFFF
            chunk_ts_ms = base_ts_ms + (chunk_idx * self.chunk_ms)
            self._enqueue(seq, chunk_ts_ms, chunk)
            chunk_idx += 1
        self._tail = buf[offset:]

    # ── Queue + drop policy ─────────────────────────────────────────────────
    def _enqueue(self, seq, ts_ms, raw_chunk):
        """Push (seq, ts_ms, b64) into the queue, dropping oldest on full."""
        b64 = _b64_text(raw_chunk)
        item = (seq, ts_ms, b64)
        try:
            self._queue.put_nowait(item)
        except _queue.Full:
            # Drop oldest. Best-effort — a concurrent reader may steal it
            # first; that's fine, the goal is to make space.
            try:
                _ = self._queue.get_nowait()
                self._dropped += 1
                if self._dropped == 1 or self._dropped % 50 == 0:
                    logger.warning(
                        "[audio_module] queue full; dropped %d frames so far",
                        self._dropped,
                    )
            except _queue.Empty:
                pass
            try:
                self._queue.put_nowait(item)
            except _queue.Full:
                # Reader is faster than us shedding; skip this item.
                self._dropped += 1

    def read_chunks(self, timeout=0.1):
        """Generator that yields (seq, ts_ms, b64) until ``stop()`` is called.

        ``timeout`` is the per-get blocking window in seconds. The generator
        loops indefinitely; the consumer should break out (e.g. on session
        close) by checking its own state. The generator exits cleanly when
        ``stop()`` has been called AND the queue is drained.
        """
        while True:
            if not self._streaming and self._queue.empty():
                return
            try:
                item = self._queue.get(timeout=timeout)
            except _queue.Empty:
                if not self._streaming:
                    return
                continue
            yield item

    # ── Introspection ───────────────────────────────────────────────────────
    @property
    def streaming(self):
        return self._streaming

    @property
    def gate_closed(self):
        return self._gate_closed

    @property
    def dropped_frames(self):
        return self._dropped

    @property
    def queue_depth(self):
        return self._queue.qsize()
