"""``python -m eesi``: talk to a character, list voices, inspect memory, read a trace.

python -m eesi talk --name Luna --persona "A curious, thoughtful companion." --preset companion
python -m eesi chat --name Kael --preset npc --persona "An elven merchant."   # type instead of speaking
python -m eesi voices
python -m eesi memory list --character luna
python -m eesi memory erase --character luna --player local
python -m eesi trace ./traces/luna.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from .memory import ALL_PLAYERS


def _character(nur: Any, args: argparse.Namespace) -> Any:
    options: dict[str, Any] = {"persona": args.persona or ""}
    if args.preset:
        options["preset"] = args.preset
    if args.voice:
        options["voice"] = args.voice
    if args.language:
        options["language"] = args.language
    if args.instructions:
        options["instructions"] = args.instructions
    options["memory"] = False if args.no_memory else (args.memory or True)
    return nur.character(args.name, **options)


async def _talk(args: argparse.Namespace) -> None:
    from .client import NurClient

    async with NurClient(base_url=args.base_url) as nur:
        character = _character(nur, args)
        await character.start(args.player, echo=args.echo, trace=args.trace)


async def _chat(args: argparse.Namespace) -> None:
    from .client import NurClient

    async with NurClient(base_url=args.base_url) as nur:
        character = _character(nur, args)
        output = None
        if args.speak:
            from .audio import local

            output = local(echo="gate")
        session = character.connect(args.player, output=output, trace=args.trace)

        @session.on("response.completed")
        def said(event: Any) -> None:
            if event.text:
                print(f"{character.name}: {event.text}", flush=True)

        @session.on("tool.completed")
        def used(event: Any) -> None:
            print(f"  [{event.name} → {event.output[:120]}]", flush=True)

        @session.on("memory.updated")
        def learned(event: Any) -> None:
            for record in event.added:
                print(f"  [remembered: {record.text}]", flush=True)

        async with session:
            print(
                f"Talking to {character.name}. Type a line and press Enter; an empty line ends.",
                flush=True,
            )
            loop = asyncio.get_running_loop()
            while True:
                line = await loop.run_in_executor(None, sys.stdin.readline)
                if not line or not line.strip():
                    break
                try:
                    await session.ask(line.strip(), timeout=60)
                except asyncio.TimeoutError:
                    print("  [no reply within 60 s]", flush=True)


async def _voices(args: argparse.Namespace) -> None:
    from .client import NurClient

    async with NurClient(base_url=args.base_url) as nur:
        for voice in await nur.voices():
            kind = "built-in" if voice.is_builtin else "yours"
            details = ", ".join(
                x for x in (voice.language, voice.gender, voice.category) if x
            )
            print(f"{voice.id:<16} {voice.name:<14} {kind:<9} {details}")


async def _memory(args: argparse.Namespace) -> None:
    from .memory import CharacterMemory, LocalMemoryStore, Memory

    store = LocalMemoryStore(args.dir)
    try:
        memory = CharacterMemory(
            Memory(store, namespace=args.namespace),
            character_id=args.character,
            character_name=args.character,
        )
        player = (
            ALL_PLAYERS
            if args.player is None
            else (None if args.player == "-" else args.player)
        )
        if args.action == "list":
            for record in await memory.list(
                player_id=player, include_inactive=args.all
            ):
                who = record.player_id or "(world)"
                flag = " [superseded]" if record.superseded_by else ""
                print(
                    f"{record.id}  {who:<14} {record.kind:<12} {record.importance:.1f}  {record.text}{flag}"
                )
        elif args.action == "export":
            data = await memory.export(args.out, player_id=player)
            if not args.out:
                print(json.dumps(data, ensure_ascii=False, indent=2))
        elif args.action == "erase":
            if args.player is None and not args.yes:
                print(
                    "Refusing to erase every player's memories without --yes.",
                    file=sys.stderr,
                )
                sys.exit(2)
            print(f"Erased {await memory.erase(player_id=player)} memories.")
        elif args.action == "forget":
            print(f"Forgot {await memory.forget(*args.ids)} memories.")
    finally:
        await store.close()


def _trace(args: argparse.Namespace) -> None:
    from .trace import format_trace

    lines = [
        json.loads(line)
        for line in Path(args.path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(format_trace(lines))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m eesi", description="EESI Nur SDK tools."
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="API base URL (default: EESI_API_URL or https://api.eesi.ai)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def character_options(command: argparse.ArgumentParser) -> None:
        command.add_argument("--name", required=True)
        command.add_argument("--persona", default="")
        command.add_argument(
            "--preset",
            default=None,
            help="npc, multiplayer_npc, companion, storyteller, support_agent, assistant",
        )
        command.add_argument(
            "--voice", default=None, help="A voice name (Atlas, Nova, …) or id (ev_…)"
        )
        command.add_argument("--language", default=None)
        command.add_argument("--instructions", default=None)
        command.add_argument(
            "--memory", default=None, help="Memory directory (default .eesi/memory)"
        )
        command.add_argument("--no-memory", action="store_true")
        command.add_argument(
            "--player", default="local", help="Who you are, for memory"
        )
        command.add_argument("--trace", default=None, help="Write a JSONL trace here")

    talk = commands.add_parser("talk", help="Talk through the microphone and speaker")
    character_options(talk)
    talk.add_argument(
        "--echo",
        default="auto",
        choices=["auto", "gate", "off"],
        help="off = headphones",
    )
    chat = commands.add_parser("chat", help="Type to a character")
    character_options(chat)
    chat.add_argument(
        "--speak", action="store_true", help="Play the voice through the speaker too"
    )
    commands.add_parser("voices", help="List voices")
    memory = commands.add_parser("memory", help="Inspect, export or erase local memory")
    memory.add_argument("action", choices=["list", "export", "erase", "forget"])
    memory.add_argument("ids", nargs="*", help="Record ids (forget)")
    memory.add_argument(
        "--character",
        required=True,
        help="The character id (its name in lowercase-dashes)",
    )
    memory.add_argument(
        "--player",
        default=None,
        help="One player (or - for world memory); default every player",
    )
    memory.add_argument("--dir", default=".eesi/memory")
    memory.add_argument("--namespace", default="default")
    memory.add_argument("--out", default=None)
    memory.add_argument(
        "--all", action="store_true", help="Include superseded and expired records"
    )
    memory.add_argument("--yes", action="store_true")
    trace = commands.add_parser("trace", help="Print a trace as a timeline")
    trace.add_argument("path")

    args = parser.parse_args(argv)
    try:
        if args.command == "talk":
            asyncio.run(_talk(args))
        elif args.command == "chat":
            asyncio.run(_chat(args))
        elif args.command == "voices":
            asyncio.run(_voices(args))
        elif args.command == "memory":
            asyncio.run(_memory(args))
        elif args.command == "trace":
            _trace(args)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
