import asyncio

from orchestrator.a2a_client import A2AClient


async def main():
    client = await A2AClient.discover("http://localhost:8001")
    print([s.id for s in client.card.skills])
    print(await client.call("get_order", {"order_id": 1042}, "tok_acme_alice"))
    print(await client.call("get_order", {"order_id": 2001}, "tok_acme_alice"))
    print(await client.call("preview_action", {"action": "refund", "order_id": 1042}, "tok_acme_alice"))


asyncio.run(main())