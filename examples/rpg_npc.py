"""An RPG NPC wired to a game: live state, tools, cues and per-player memory.

    python examples/rpg_npc.py                       # talk through your microphone
    python examples/rpg_npc.py --player alice        # a different player: a different relationship

The "game" here is a small simulation: the sun sets, guards patrol, and after
a while the player draws a sword. Kael sees all of it as live state and can
act on the world only through his tools, which the game checks.
"""

import argparse
import asyncio
from dataclasses import dataclass, field
from typing import Literal

from eesi import NurClient, Started, ToolContext


@dataclass
class World:
    """The game's own state: authoritative, whatever anyone says aloud."""

    hour: int = 18
    north_gate: str = "closed"
    reputation: dict[str, str] = field(default_factory=lambda: {"alice": "trusted"})
    inventory: dict[str, list[str]] = field(
        default_factory=lambda: {"kael": ["elven cloak", "rope", "lantern"]}
    )
    gold: dict[str, int] = field(default_factory=dict)

    def time_of_day(self) -> str:
        return (
            "night"
            if self.hour >= 20 or self.hour < 6
            else "dusk"
            if self.hour >= 18
            else "day"
        )


world = World()
nur = NurClient()

kael = nur.npc(
    name="Kael",
    persona="An elven merchant who keeps the northern gate of Eldermoor. He distrusts the king's men.",
    voice="Atlas",
    traits=["wry", "watchful", "fair to honest folk"],
    boundaries=["Never reveal the location of the hidden vault under the old mill."],
    knowledge=[
        "The king raised the bridge toll last spring.",
        "Wolves have been seen near the river ford.",
    ],
    memory=nur.memory.local("./kael-memory"),
    context={"realm": "Eldermoor, a river kingdom under an unpopular king"},
)


def trusted(ctx: ToolContext) -> bool | str:
    if world.reputation.get(ctx.player_id or "", "unknown") != "trusted":
        return "Kael does not open the gate for strangers"
    return True


@kael.tool(allow=trusted)
async def open_gate(gate: Literal["north"] = "north") -> str:
    """Open the northern gate for the player."""
    world.north_gate = "open"
    return "The north gate is open."


@kael.tool
def inspect_inventory() -> dict:
    """What Kael has for sale."""
    return {
        "for_sale": world.inventory["kael"],
        "prices_in_gold": {"elven cloak": 40, "rope": 2, "lantern": 5},
    }


@kael.tool(confirm=True)
def sell_item(item: str, price: int, ctx: ToolContext) -> str:
    """Sell the player an item from Kael's stock at an agreed price.

    Args:
        item: One of the items for sale.
        price: The price in gold the player agreed to.
    """
    if item not in world.inventory["kael"]:
        return f"Kael has no {item}."
    world.inventory["kael"].remove(item)
    world.gold[ctx.player_id or "?"] = world.gold.get(ctx.player_id or "?", 100) - price
    return f"Sold the {item} for {price} gold."


@kael.tool
async def fetch_lantern_oil() -> Started:
    """Fetch lantern oil from the storehouse. Takes a while."""

    async def walk() -> str:
        await asyncio.sleep(20)
        world.inventory["kael"].append("lantern oil")
        return "Kael is back with a flask of lantern oil."

    return Started("Kael heads to the storehouse.", asyncio.ensure_future(walk()))


async def game_loop(session) -> None:
    """The game ticking: state flows to Kael as it changes; some moments make him speak."""
    tick = 0
    while not session.ended:
        await kael.context.update(
            {"time_of_day": world.time_of_day(), "hour": f"{world.hour}:00"}
        )
        await session.context.update(
            {
                "north_gate": world.north_gate,
                "player_reputation": world.reputation.get(session.player_id, "unknown"),
                "guards_nearby": tick % 4 == 0,
            }
        )
        if tick == 6:
            session.moment(
                "A wolf howls somewhere past the river ford.",
                kind="wolves",
                cooldown=60,
            )
        if tick == 12:
            await session.context.update({"player_weapon": "sword drawn"}, ttl=20)
            await session.cue(
                "The player just drew a sword, right in front of the gate."
            )
        if tick % 5 == 4:
            world.hour = (world.hour + 1) % 24
        tick += 1
        await asyncio.sleep(3)


async def main(player: str) -> None:
    async with kael.connect(
        player, audio=nur.audio.local(), trace=f"traces/kael-{player}.jsonl"
    ) as session:

        @session.on("tool.completed")
        def did(event):
            print(f"  [{event.name}] {event.output}")

        @session.on("tool.failed")
        def refused(event):
            print(f"  [{event.name} refused] {event.error}")

        game = asyncio.create_task(game_loop(session))
        print(f"Kael is at the gate. You are {player}. Ctrl+C to leave.")
        try:
            await session.run()
        finally:
            game.cancel()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", default="sam")
    args = parser.parse_args()
    try:
        asyncio.run(main(args.player))
    except KeyboardInterrupt:
        pass
