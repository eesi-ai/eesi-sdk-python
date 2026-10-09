"""A personal companion with long memory, preferences and moods.

    python examples/companion.py            # talk; Ctrl+C to end
    python examples/companion.py --memory   # show what Luna remembers about you

Luna remembers you between runs (./luna-memory), recalls what you mention,
and learns from each conversation when it ends. Say "please keep it short"
and she will, today and next time.
"""

import argparse
import asyncio

from eesi import MemoryPolicy, NurClient, ToolContext

nur = NurClient()
memory = nur.memory.local("./luna-memory", policy=MemoryPolicy(retention_days=365))

luna = nur.companion(
    name="Luna",
    persona="A curious, thoughtful companion who loves astronomy and long walks.",
    voice="Nova",
    traits=["warm", "playful", "honest"],
    goals=["Help them keep the habits they care about."],
    memory=memory,
)
ME = "me"


@luna.tool
async def remember_preference(how: str, ctx: ToolContext) -> str:
    """Remember how the person wants you to talk with them, when they ask (e.g. "keep it short").

    Args:
        how: The preference in their words.
    """
    await ctx.session.preferences.set(style=how)
    return "Noted for this and every later conversation."


async def talk() -> None:
    async with luna.connect(
        ME, player_name="friend", audio=nur.audio.local()
    ) as session:

        @session.on("memory.retrieved")
        def recalled(event):
            for record in event.records:
                print(f"  (Luna recalls: {record.text})")

        await session.set_mood("cheerful", reason="it is a clear night")
        print("Talking to Luna. Ctrl+C to end.")
        await session.run()


async def show() -> None:
    you = luna.memory.for_player(ME)
    for record in await you.list():
        print(f"{record.kind:<12} {record.text}")
    print("preferences:", await you.preferences())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--memory", action="store_true", help="show what Luna remembers"
    )
    args = parser.parse_args()
    try:
        asyncio.run(show() if args.memory else talk())
    except KeyboardInterrupt:
        pass
