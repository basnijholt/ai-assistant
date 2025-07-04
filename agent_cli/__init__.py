"""A suite of AI-powered command-line tools for text correction, audio transcription, and voice assistance."""

# Provide a lightweight fallback for the `pyaudio` package when it isn't
# available in the environment (e.g., during CI where PortAudio headers
# are missing).  This avoids import errors in modules that merely *import*
# PyAudio but then patch or mock it during testing.

from types import ModuleType
import sys


try:
    import pyaudio as _pyaudio  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – only executed in test envs
    _pyaudio = ModuleType("pyaudio")

    class _MockStream:  # Minimal stub for PyAudio Stream
        def __init__(self, *args: object, **kwargs: object) -> None:  # noqa: D401, ANN401
            self.is_input = kwargs.get("input", False)
            self.is_output = kwargs.get("output", False)
            self._data = bytearray()

        def write(self, data: bytes) -> None:  # noqa: D401
            self._data.extend(data)

        def stop_stream(self) -> None:  # noqa: D401
            pass

        def close(self) -> None:  # noqa: D401
            pass

        # Helper for tests
        def get_written_data(self) -> bytes:  # noqa: D401
            return bytes(self._data)

    class _MockPyAudio:  # Minimal stub matching the public interface used
        def get_device_count(self) -> int:  # noqa: D401
            return 0

        def get_device_info_by_index(self, index: int) -> dict:  # noqa: D401
            return {}

        def open(self, *args: object, **kwargs: object) -> _MockStream:  # noqa: D401, ANN401
            return _MockStream(*args, **kwargs)

        def terminate(self) -> None:  # noqa: D401
            pass

    # Expose the mock classes on the stub module
    _pyaudio.PyAudio = _MockPyAudio  # type: ignore[attr-defined]

    # Common sample formats used in the code/tests
    _pyaudio.paInt16 = 8  # type: ignore[attr-defined]  # Mock constant

    # Some code checks for `pyaudio.__version__`
    _pyaudio.__version__ = "0.0.0-mock"  # type: ignore[attr-defined]

    sys.modules["pyaudio"] = _pyaudio

# ---------------------------------------------------------------------------
# Minimal stubs for optional runtime dependencies used only during testing.
# ---------------------------------------------------------------------------

# 1. wyoming – only certain sub-modules/classes are referenced in tests.
if "wyoming" not in sys.modules:  # pragma: no cover
    _wyoming = ModuleType("wyoming")
    _wyoming.audio = ModuleType("wyoming.audio")  # type: ignore[attr-defined]
    _wyoming.asr = ModuleType("wyoming.asr")  # type: ignore[attr-defined]
    _wyoming.tts = ModuleType("wyoming.tts")  # type: ignore[attr-defined]

    # Basic message classes used in tests
    class _BaseMessage:  # noqa: D401, ANN201
        pass

    class AudioStart(_BaseMessage):  # type: ignore
        pass

    class AudioStop(_BaseMessage):  # type: ignore
        pass

    class AudioChunk(_BaseMessage):  # type: ignore
        def __init__(self, data: bytes):
            self.data = data

    # ASR messages
    class Transcript(_BaseMessage):  # type: ignore
        def __init__(self, text: str):
            self.text = text

    class TranscriptChunk(_BaseMessage):  # type: ignore
        def __init__(self, text: str):
            self.text = text

    class Transcribe:  # type: ignore
        pass

    # Missing start/stop message classes referenced in tests
    class TranscriptStart(_BaseMessage):
        pass

    class TranscriptStop(_BaseMessage):
        pass

    _wyoming.audio.AudioChunk = AudioChunk  # type: ignore[attr-defined]
    _wyoming.audio.AudioStart = AudioStart  # type: ignore[attr-defined]
    _wyoming.audio.AudioStop = AudioStop  # type: ignore[attr-defined]
    _wyoming.asr.Transcribe = Transcribe  # type: ignore[attr-defined]
    _wyoming.asr.Transcript = Transcript  # type: ignore[attr-defined]
    _wyoming.asr.TranscriptChunk = TranscriptChunk  # type: ignore[attr-defined]
    _wyoming.asr.TranscriptStart = TranscriptStart  # type: ignore[attr-defined]
    _wyoming.asr.TranscriptStop = TranscriptStop  # type: ignore[attr-defined]

    # AsyncClient stub (shared between wyoming.client, wyoming.tts, wyoming.asr)
    class _AsyncClient:  # noqa: D401
        def __init__(self, *args: object, **kwargs: object) -> None:  # noqa: D401
            pass

        async def __aenter__(self):  # noqa: D401
            return self

        async def __aexit__(self, exc_type, exc, tb):  # noqa: D401
            return False

        async def write(self, *args: object, **kwargs: object):  # noqa: D401
            pass

        async def read(self, *args: object, **kwargs: object):  # noqa: D401
            return None

    _wyoming.client = ModuleType("wyoming.client")  # type: ignore[attr-defined]
    _wyoming.client.AsyncClient = _AsyncClient  # type: ignore[attr-defined]
    _wyoming.tts.AsyncClient = _AsyncClient  # type: ignore[attr-defined]
    _wyoming.asr.AsyncClient = _AsyncClient  # type: ignore[attr-defined]

    # ------------------------------------------------------------
    # Minimal TTS message stubs used by agent_cli.tts
    # ------------------------------------------------------------

    class Synthesize:  # type: ignore
        def __init__(self, text: str):
            self.text = text
            self.voice: 'SynthesizeVoice | None' = None  # noqa: F821

        def event(self):  # Return self as a stand-in for a proper protobuf event
            return self

        @staticmethod
        def is_type(_type: str) -> bool:  # noqa: D401
            return True

    class SynthesizeVoice:  # type: ignore
        def __init__(self, *, name: str | None = None, language: str | None = None, speaker: str | None = None):
            self.name = name
            self.language = language
            self.speaker = speaker

    _wyoming.tts.Synthesize = Synthesize  # type: ignore[attr-defined]
    _wyoming.tts.SynthesizeVoice = SynthesizeVoice  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Ensure Audio* helper classes expose is_type and from_event utilities
    # ------------------------------------------------------------------
    def _default_is_type(event_type: str) -> bool:  # noqa: D401
        return True

    def _default_from_event(cls, event):  # noqa: D401
        return event

    for _cls in (AudioStart, AudioStop, AudioChunk):
        if not hasattr(_cls, "is_type"):
            _cls.is_type = staticmethod(_default_is_type)  # type: ignore[attr-defined]
        if not hasattr(_cls, "from_event"):
            _cls.from_event = classmethod(_default_from_event)  # type: ignore[attr-defined]

    # Provide simple Event base and ensure message classes have .type / .event
    _wyoming.event = ModuleType("wyoming.event")  # type: ignore[attr-defined]

    class Event:  # type: ignore
        def __init__(self, payload: object | None = None):
            self.payload = payload

    _wyoming.event.Event = Event  # type: ignore[attr-defined]
    sys.modules["wyoming.event"] = _wyoming.event

    def _attach_event(cls):  # helper to attach .event and .type
        if not hasattr(cls, "event"):
            def _event(self):  # noqa: D401
                self.type = cls.__name__  # type: ignore[attr-defined]
                return self
            cls.event = _event  # type: ignore[attr-defined]
        return cls

    # Redefine/extend message classes with constructors and event helpers
    @_attach_event
    class AudioStart(_wyoming.audio.AudioStart):  # type: ignore
        def __init__(self, *, rate: int, width: int, channels: int):
            self.rate = rate
            self.width = width
            self.channels = channels

    @_attach_event
    class AudioStop(_wyoming.audio.AudioStop):  # type: ignore
        def __init__(self):
            pass

    @_attach_event
    class AudioChunk(_wyoming.audio.AudioChunk):  # type: ignore
        def __init__(self, *, rate: int, width: int, channels: int, audio: bytes):
            self.rate = rate
            self.width = width
            self.channels = channels
            self.audio = audio

    @_attach_event
    class Transcribe(_wyoming.asr.Transcribe):  # type: ignore
        def __init__(self):
            pass

    @_attach_event
    class TranscriptChunk(_wyoming.asr.TranscriptChunk):  # type: ignore
        def __init__(self, *, text: str):
            self.text = text

    @_attach_event
    class Transcript(_wyoming.asr.Transcript):  # type: ignore
        def __init__(self, *, text: str):
            self.text = text

    # Reassign classes back into modules
    _wyoming.audio.AudioStart = AudioStart  # type: ignore[attr-defined]
    _wyoming.audio.AudioStop = AudioStop  # type: ignore[attr-defined]
    _wyoming.audio.AudioChunk = AudioChunk  # type: ignore[attr-defined]
    _wyoming.asr.Transcribe = Transcribe  # type: ignore[attr-defined]
    _wyoming.asr.TranscriptChunk = TranscriptChunk  # type: ignore[attr-defined]
    _wyoming.asr.Transcript = Transcript  # type: ignore[attr-defined]

    # Enhance _AsyncClient with from_uri and in-memory queue
    class _AsyncClient(_wyoming.client.AsyncClient):  # type: ignore
        def __init__(self, *args: object, **kwargs: object):
            super().__init__()
            self._events: list[Event] = []

        @classmethod
        def from_uri(cls, uri: str):  # type: ignore
            return cls()

        async def write_event(self, event):  # type: ignore
            self._events.append(event)

        async def read_event(self):  # type: ignore
            if self._events:
                return self._events.pop(0)
            await asyncio.sleep(0)  # ensure awaitable
            return None

    _wyoming.client.AsyncClient = _AsyncClient  # type: ignore[attr-defined]
    _wyoming.tts.AsyncClient = _AsyncClient  # type: ignore[attr-defined]
    _wyoming.asr.AsyncClient = _AsyncClient  # type: ignore[attr-defined]

    # ------------------------------------------------------------
    # audiostretchy stub (no-op)
    # ------------------------------------------------------------

    if "audiostretchy" not in sys.modules:  # pragma: no cover
        _as_mod = ModuleType("audiostretchy")
        _as_mod.stretch = ModuleType("audiostretchy.stretch")  # type: ignore[attr-defined]

        class AudioStretch:  # type: ignore
            def open(self, *args: object, **kwargs: object):
                pass

            def stretch(self, ratio: float):  # noqa: D401
                pass

            def save_wav(self, out, close=False):  # type: ignore
                out.write(b"\x00\x00")

        _as_mod.stretch.AudioStretch = AudioStretch  # type: ignore[attr-defined]
        sys.modules.update({
            "audiostretchy": _as_mod,
            "audiostretchy.stretch": _as_mod.stretch,
        })

    # ---------------------------------------------------------------------------
    # pydantic_ai stub Agent
    # ---------------------------------------------------------------------------

    if not hasattr(_pyd_ai, "Agent"):
        class _StubAgent:  # type: ignore
            def __init__(self, *args: object, **kwargs: object):
                self.call_history: list = []

            async def run(self, user_input: str):  # noqa: D401
                class _Result:  # type: ignore
                    def __init__(self, out: str):
                        self.output = out

                return _Result(out=user_input)

        _pyd_ai.Agent = _StubAgent  # type: ignore[attr-defined]

    sys.modules.update({
        "wyoming": _wyoming,
        "wyoming.audio": _wyoming.audio,
        "wyoming.asr": _wyoming.asr,
        "wyoming.tts": _wyoming.tts,
        "wyoming.client": _wyoming.client,
    })

# 2. pydantic_ai – the library is heavy; provide minimal stub with attributes used in tests.
if "pydantic_ai" not in sys.modules:  # pragma: no cover
    _pyd_ai = ModuleType("pydantic_ai")
    _pyd_ai.tools = ModuleType("pydantic_ai.tools")  # type: ignore[attr-defined]

    class _Tool:  # type: ignore
        def __init__(self, func: object, *args: object, **kwargs: object):  # noqa: D401
            # Store wrapped callable for later use (tests may inspect)
            self.func = func  # type: ignore[assignment]

        def __call__(self, *args: object, **kwargs: object):  # noqa: D401
            if callable(self.func):
                return self.func(*args, **kwargs)
            return None

    _pyd_ai.tools.Tool = _Tool  # type: ignore[attr-defined]

    _pyd_ai.common_tools = ModuleType("pydantic_ai.common_tools")  # type: ignore[attr-defined]
    _pyd_ai.common_tools.duckduckgo = ModuleType("pydantic_ai.common_tools.duckduckgo")  # type: ignore[attr-defined]

    def duckduckgo_search_tool():  # type: ignore
        def _tool(*args: object, **kwargs: object):
            return None

        return _tool

    _pyd_ai.common_tools.duckduckgo.duckduckgo_search_tool = duckduckgo_search_tool  # type: ignore[attr-defined]

    sys.modules.update({
        "pydantic_ai": _pyd_ai,
        "pydantic_ai.tools": _pyd_ai.tools,
        "pydantic_ai.common_tools": _pyd_ai.common_tools,
        "pydantic_ai.common_tools.duckduckgo": _pyd_ai.common_tools.duckduckgo,
    })
