from __future__ import annotations

import asyncio
import logging
import traceback

import websockets.asyncio.server
import websockets.frames

from diffusion_policy.sonic import msgpack_numpy

LOGGER = logging.getLogger(__name__)


class SonicWebsocketPolicyServer:
    def __init__(self, policy, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.policy = policy
        self.host = host
        self.port = port

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        async with websockets.asyncio.server.serve(
            self._handler, self.host, self.port, compression=None, max_size=None
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket) -> None:
        LOGGER.info("SONIC connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self.policy.metadata))
        while True:
            try:
                observation = msgpack_numpy.unpackb(await websocket.recv())
                await websocket.send(packer.pack(self.policy.infer(observation)))
            except websockets.ConnectionClosed:
                LOGGER.info("SONIC connection from %s closed", websocket.remote_address)
                return
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="SONIC policy inference failed",
                )
                raise
