"""A tool-using support agent: verified identity, permissions in code, confirmation before refunds.

    python examples/support_agent.py          # type
    python examples/support_agent.py --voice  # talk

The agent can look orders up for anyone, but only refunds an order after the
caller is verified (checked in code by `allow`) and has said yes to the
refund (`confirm=True`). The tool descriptions never authorize anything.
"""

import argparse
import asyncio
import sys

from eesi import NurClient, ToolContext

ORDERS = {
    "A1001": {
        "email": "sam@example.com",
        "item": "trail running shoes",
        "status": "delivered",
        "total": 89.0,
    },
    "A1002": {
        "email": "ana@example.com",
        "item": "rain jacket",
        "status": "in transit",
        "total": 120.0,
    },
}
verified: dict[str, str] = {}  # session player id -> verified email

nur = NurClient()
agent = nur.support_agent(
    name="Rowan",
    persona="Support for Northwind Outfitters, an outdoor gear shop.",
    voice="Orion",
    knowledge=[
        "Refunds are allowed within 30 days of delivery.",
        "Orders ship from Oslo.",
    ],
    turn_taking="app",  # fresh state before every reply: most reliable tool use
    thinking="deliberate",  # every request through the reasoning lane; slower, steadier with tools
    memory=False,
)


@agent.tool
def lookup_order(order_id: str) -> dict:
    """Look up an order by its id, like A1001."""
    order = ORDERS.get(order_id.upper())
    if order is None:
        return {"found": False}
    return {
        "found": True,
        "item": order["item"],
        "status": order["status"],
        "total": order["total"],
    }


@agent.tool
async def verify_customer(order_id: str, email: str, ctx: ToolContext) -> str:
    """Verify the caller owns an order by the email on it.

    Args:
        order_id: The order they are asking about.
        email: The email address they gave, spelled out.
    """
    order = ORDERS.get(order_id.upper())
    if order is None or order["email"] != email.strip().lower():
        return "Not verified: that email does not match the order."
    verified[ctx.player_id or ""] = order["email"]
    await ctx.session.context.update({"verified_email": order["email"]})
    return "Verified."


def must_be_verified(ctx: ToolContext) -> bool | str:
    order = ORDERS.get(str(ctx.arguments.get("order_id", "")).upper())
    if order is None:
        return "no such order"
    if verified.get(ctx.player_id or "") != order["email"]:
        return "the caller has not been verified for this order"
    return True


@agent.tool(allow=must_be_verified, confirm=True)
def refund_order(order_id: str) -> str:
    """Refund a delivered order in full."""
    order = ORDERS[order_id.upper()]
    if order["status"] != "delivered":
        return f"Cannot refund: the order is {order['status']}."
    order["status"] = "refunded"
    return f"Refunded {order['total']:.2f} to the original payment method."


async def main(voice: bool) -> None:
    session = agent.connect("caller-1", audio=nur.audio.local() if voice else None)

    @session.on("tool.completed")
    def done(event):
        print(f"  [{event.name}] {event.output}")

    @session.on("tool.failed")
    def refused(event):
        print(f"  [{event.name} refused: {event.reason}] {event.error}")

    async with session:
        if voice:
            await session.run()
            return

        @session.on("response.completed")
        def said(event):
            if event.text:
                print(f"Rowan: {event.text}")

        print(
            "Type to Rowan (empty line ends). Try: I'd like a refund for order A1001."
        )
        loop = asyncio.get_running_loop()
        while line := (await loop.run_in_executor(None, sys.stdin.readline)).strip():
            await session.ask(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", action="store_true")
    asyncio.run(main(parser.parse_args().voice))
