"""Talk to a character through this computer's microphone and speaker.

    pip install "eesi-sdk[audio]"
    export EESI_API_KEY=sk-eesi-...
    python examples/quickstart.py

Kael greets you. Talk over him and he stops. Press Ctrl+C to end; what was
worth remembering is kept in .eesi/memory for next time.
"""

from eesi import NurClient

nur = NurClient()

kael = nur.npc(
    name="Kael",
    persona="A mysterious elven merchant who distrusts kings.",
    voice="Atlas",
    memory=True,
)

kael.run()  # blocks until Ctrl+C; in async code, `await kael.start()`
