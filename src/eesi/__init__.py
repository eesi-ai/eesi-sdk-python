"""EESI Nur SDK: characters that talk, listen, remember and act, in real time.

::

    from eesi import NurClient

    nur = NurClient()  # EESI_API_KEY

    kael = nur.npc(
        name="Kael",
        persona="A mysterious elven merchant who distrusts kings.",
        voice="Atlas",
        memory=True,
    )

    @kael.tool
    async def open_gate() -> str:
        \"\"\"Open the northern castle gate.\"\"\"
        return await game.open_gate()

    async with kael.connect(player_id="player_42") as session:
        await session.context.update({"location": "The Northern Gate"})
        await session.run()

Everything runs on Nur Live (``nur-live-v1``): one realtime conversation per
session, full-duplex, with your character, your tools, your live context and
your players' memories. The REST client for the rest of the API is
:class:`EESIClient`, re-exported from ``eesi_sdk``.
"""

from eesi_sdk import EESIClient

from . import events
from ._version import __version__
from .audio import LocalAudio, WavInput, WavOutput
from .character import Character
from .client import (
    ClientSecret,
    NurClient,
    PhoneCall,
    PhoneNumber,
    Voice,
    VoiceDesign,
    WebhookVerificationError,
    verify_webhook,
)
from .context import LiveContext
from .errors import (
    ApiError,
    AudioDeviceError,
    AuthenticationError,
    ConfigurationError,
    ContextError,
    EESIError,
    MemoryStoreError,
    RealtimeConnectionError,
    SessionClosedError,
    ToolDefinitionError,
)
from .events import Event
from .languages import describe as describe_language
from .memory import (
    ALL_PLAYERS,
    CharacterMemory,
    InMemoryStore,
    LocalMemoryStore,
    Memory,
    MemoryPolicy,
    MemoryRecord,
    MemoryStore,
    PlayerMemory,
)
from .multiplayer import Player
from .presets import PRESETS, Preset
from .proactive import Proactive, Trigger
from .room import Room
from .session import Session
from .tools import Started, Tool, ToolContext, tool
from .trace import Trace

__all__ = [
    "ALL_PLAYERS",
    "ApiError",
    "AudioDeviceError",
    "AuthenticationError",
    "Character",
    "CharacterMemory",
    "ClientSecret",
    "ConfigurationError",
    "ContextError",
    "EESIClient",
    "EESIError",
    "Event",
    "InMemoryStore",
    "LiveContext",
    "LocalAudio",
    "LocalMemoryStore",
    "Memory",
    "MemoryPolicy",
    "MemoryRecord",
    "MemoryStore",
    "MemoryStoreError",
    "NurClient",
    "PhoneCall",
    "PhoneNumber",
    "PRESETS",
    "Player",
    "PlayerMemory",
    "Preset",
    "Proactive",
    "RealtimeConnectionError",
    "Room",
    "Session",
    "SessionClosedError",
    "Started",
    "Tool",
    "ToolContext",
    "ToolDefinitionError",
    "Trace",
    "Trigger",
    "Voice",
    "VoiceDesign",
    "WavInput",
    "WavOutput",
    "WebhookVerificationError",
    "__version__",
    "describe_language",
    "events",
    "tool",
    "verify_webhook",
]
