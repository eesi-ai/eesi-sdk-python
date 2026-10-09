"""Real Nur Live sessions through the SDK. Skipped unless EESI_LIVE_TESTS=1 and EESI_API_KEY are set.

These cost usage and depend on a deployment's models, so they assert what
must hold (a tool runs when asked, a player is remembered, two players stay
apart), never exact words. EESI_API_URL selects the deployment.
"""

from __future__ import annotations

import os
from typing import Literal

import pytest
from conftest import run_async
from eesi import NurClient
from eesi.testing import run_script

pytestmark = pytest.mark.skipif(
    not (os.environ.get("EESI_LIVE_TESTS") == "1" and os.environ.get("EESI_API_KEY")),
    reason="set EESI_LIVE_TESTS=1 and EESI_API_KEY to run real sessions",
)


@run_async(timeout=300)
async def test_a_character_acts_learns_and_remembers_only_its_player():
    async with NurClient() as nur:
        kael = nur.npc(
            name="Kael",
            persona="An elven merchant at the northern gate.",
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
                "Hello! My name is Sam and I'm a blacksmith.",
                ("Could you open the north gate for me, please?", 11.0),
            ],
            player_id="sam",
        )
        assert first.heard, "the player was transcribed"
        assert first.said, "the character answered"
        assert opened == ["north"], first.events
        assert any("sam" in record.text.lower() for record in first.learned)

        again = await run_script(
            kael, ["Hello again! What is my name?"], player_id="sam"
        )
        assert "sam" in again.said_text.lower()

        stranger = await run_script(
            kael, ["Hello! Do you know who I am?"], player_id="ada"
        )
        assert "blacksmith" not in stranger.said_text.lower()
