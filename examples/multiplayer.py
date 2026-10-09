"""One character, two players at once: separate conversations, one consistent story.

    python examples/multiplayer.py

Ana and Ben each talk to Kael in their own session (typed here, so you can
run it anywhere; give each `audio=` for voices). Their memories never mix,
but with `multiplayer_npc` every session sees a short log of what Kael told
the others, so when Ben asks where Ana went, Kael knows.
"""

import asyncio

from eesi import NurClient

nur = NurClient()
kael = nur.multiplayer_npc(
    name="Kael",
    persona="An elven merchant at the northern gate who sees everyone come and go.",
    voice="Atlas",
    greeting=False,
    memory=nur.memory.in_memory(),  # nothing kept after the run
)


async def say(session, who: str, text: str) -> None:
    print(f"{who}: {text}")
    reply = await session.ask(text)
    print(f"Kael to {who}: {reply.text}\n")


async def main() -> None:
    async with (
        kael.connect("ana", player_name="Ana") as ana,
        kael.connect("ben", player_name="Ben") as ben,
    ):
        await say(
            ana,
            "Ana",
            "I'm heading to the old mill to look for my brother. Don't tell the guards.",
        )
        await say(ben, "Ben", "Have you seen my friend Ana? Where did she go?")
        await say(ana, "Ana", "Did anyone ask about me?")


asyncio.run(main())
