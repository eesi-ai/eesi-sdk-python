"""Measure a character: reply latency, tool success, memory recall and session reliability.

    python examples/benchmark.py --runs 3

Each run is two real sessions with a scripted player: one that introduces
themselves and asks for a tool, and one that asks whether the character
remembers them. Runs go one after another unless --parallel is set; mind your
organization's concurrency. Latency is measured in the SDK, from the end of
the player's turn to the first audio of the reply reaching this machine.
"""

import argparse
import asyncio
import re
import statistics
import time
from typing import Literal

from eesi import NurClient
from eesi.testing import run_script

NAMES = ["Sam", "Ana", "Tomas", "Leila", "Ravi", "Mei"]


async def one_run(nur: NurClient, index: int) -> dict:
    name = NAMES[index % len(NAMES)]
    kael = nur.npc(
        name="Kael",
        persona="An elven merchant at the northern gate.",
        voice="Atlas",
        memory=nur.memory.in_memory(),
    )
    calls: list[str] = []

    @kael.tool
    def open_gate(gate: Literal["north", "south"] = "north") -> str:
        """Open one of the castle gates for the player."""
        calls.append(gate)
        return f"The {gate} gate is open."

    started = time.monotonic()
    try:
        first = await run_script(
            kael,
            [
                f"Hi! My name is {name}, I'm a weaver.",
                ("Please open the north gate for me.", 11.0),
            ],
            player_id=f"p{index}",
        )
        second = await run_script(
            kael, ["Hello again. What's my name?"], player_id=f"p{index}"
        )
    except (
        Exception
    ) as error:  # a session that failed to run counts against reliability
        return {"ok": False, "error": str(error)}
    latencies = [
        e.latency
        for r in (first, second)
        for e in r.events
        if e.type == "response.completed" and e.latency
    ]
    # The name as the character heard it ("Tomas" may be heard as "Thomas"): recall is about memory, not spelling.
    heard = re.search(r"name is (\w+)", " ".join(first.heard), re.IGNORECASE)
    names = {name.lower(), heard.group(1).lower() if heard else name.lower()}
    return {
        "ok": True,
        "latencies": latencies,
        "tool_ok": calls == ["north"],
        "recall_ok": any(candidate in second.said_text.lower() for candidate in names),
        "interruptions": first.metrics["interruptions"]
        + second.metrics["interruptions"],
        "reconnects": first.metrics["reconnects"] + second.metrics["reconnects"],
        "seconds": round(time.monotonic() - started, 1),
    }


async def main(runs: int, parallel: int) -> None:
    async with NurClient() as nur:
        gate = asyncio.Semaphore(parallel)

        async def limited(index: int) -> dict:
            async with gate:
                return await one_run(nur, index)

        results = await asyncio.gather(*(limited(i) for i in range(runs)))
    ok = [r for r in results if r["ok"]]
    latencies = sorted(x for r in ok for x in r["latencies"])
    print(f"runs: {runs}  completed: {len(ok)}")
    if latencies:
        p95 = latencies[min(len(latencies) - 1, int(0.95 * (len(latencies) - 1) + 0.5))]
        print(
            f"reply latency: p50 {statistics.median(latencies):.2f}s  p95 {p95:.2f}s  ({len(latencies)} replies)"
        )
    if ok:
        print(f"tool success:  {sum(r['tool_ok'] for r in ok)}/{len(ok)}")
        print(f"memory recall: {sum(r['recall_ok'] for r in ok)}/{len(ok)}")
        print(
            f"interruptions: {sum(r['interruptions'] for r in ok)}  reconnects: {sum(r['reconnects'] for r in ok)}"
        )
    for r in results:
        if not r["ok"]:
            print("failed:", r["error"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--parallel", type=int, default=1)
    args = parser.parse_args()
    asyncio.run(main(args.runs, args.parallel))
