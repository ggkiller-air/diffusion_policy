import argparse
import logging

import torch

from diffusion_policy.sonic.adapter import SonicPolicyAdapter
from diffusion_policy.sonic.checkpoint import load_policy
from diffusion_policy.sonic.websocket_server import SonicWebsocketPolicyServer

LOGGER = logging.getLogger(__name__)


def main(args) -> None:
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = load_policy(args.checkpoint, device)
    adapter = SonicPolicyAdapter(policy, device)
    LOGGER.info("SONIC metadata: %s", adapter.metadata)
    SonicWebsocketPolicyServer(adapter, host=args.host, port=args.port).serve_forever()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve a Diffusion Policy SONIC checkpoint"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    main(build_argparser().parse_args())
