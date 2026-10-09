"""A repeatable conversation with a character, for regression tests and demos.

    python examples/scripted_run.py
    python examples/scripted_run.py --record kael.wav --trace kael.jsonl

A scripted player (EESI text to speech, Nova's voice) says each line into a
real Nur Live session with a pause after it, the character answers, and the
run checks what must hold: the tool was called, and in a second session the
character remembers the player. Exits 1 when a check fails. Results vary the
way a model's do, so it checks behaviour, never exact words.
"""

import argparse
import asyncio
import sys
from typing import Literal

from eesi import NurClient
from eesi.testing import run_script


async def main(record: str | None, trace: str | None) -> int:
    async with NurClient() as nur:
        kael = nur.npc(
            name="Kael",
            persona="An elven merchant who keeps the northern gate of Eldermoor.",
            voice="Atlas",
            memory=nur.memory.in_memory(),
        )
        opened: list[str] = []

        @kael.tool
        def open_gate(gate: Literal["north", "south"] = "north") -> str:
            """Open one of the castle gates for the player."""
            opened.append(gate)
            return f"The {gate} gate is open."

        first = await run_script(
            kael,
            [
                "Hello there! My name is Sam. I'm a blacksmith from the river villages.",
                ("Could you open the north gate for me, please?", 11.0),
            ],
            player_id="sam",
            record=record,
            trace=trace,
        )
        second = await run_script(
            kael, ["Hello again! Do you remember my name?"], player_id="sam"
        )

        for label, result in (("first", first), ("second", second)):
            print(f"--- {label} session ({result.session_id})")
            for line in result.heard:
                print(f"  heard: {line}")
            for line in result.said:
                print(f"  said:  {line}")
            print(f"  tools: {[(t.name, t.output) for t in result.tool_calls]}")
            print(f"  learned: {[r.text for r in result.learned]}")
            print(f"  metrics: {result.metrics}")

        checks = {
            "the gate was opened when asked": opened == ["north"],
            "something was learned about Sam": any(
                "Sam" in r.text for r in first.learned
            ),
            "Sam was remembered next time": "sam" in second.said_text.lower(),
        }
        for name, ok in checks.items():
            print(f"{'PASS' if ok else 'FAIL'}  {name}")
        return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--record", help="write Kael's voice from the first session to this WAV"
    )
    parser.add_argument(
        "--trace", help="write the first session's trace to this JSONL file"
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.record, args.trace)))
