# SONIC Diffusion Policy

This branch trains one Diffusion Policy backend with the same `sonic_vla_v1`
boundary as the VLA repositories. The model predicts 40 actions with 78 values
per step: `motion_token[0:64]`, `left_hand[64:71]`, and `right_hand[71:78]`.
SONIC, not this policy, decodes the 64D motion token into whole-body control.

## Modes

| Config | Current tactile | Future tactile target | Future state/vision targets |
| --- | --- | --- | --- |
| `sonic_notactile.json` | no | no | no |
| `sonic_htd.json` | yes | yes | no |
| `sonic_jepa.json` | yes | yes | yes |

HTD is short for *Humanoid Transformer with Touch Dreaming* (arXiv:2604.13015).
Here, HTD mode means its current-tactile fusion and future-tactile latent objective;
it is not a claim that this Diffusion Policy reproduces the paper's full system.
UniVLaT/JEPA additionally predicts future state and stereo-vision latents. Future
observations are teacher targets only; action conditioning and inference use current
observations only.

## Environment

```bash
cd /root/Projects/diffusion_policy
uv venv --python 3.11
uv pip install --python .venv/bin/python -e . -r requirements-sonic.txt
```

The SONIC path is independent of the repository's legacy robomimic environment.

## Train

Set one config and output directory:

```bash
CONFIG=configs/sonic_notactile.json  # or sonic_htd.json / sonic_jepa.json
OUTPUT=outputs/sonic_notactile
```

Single GPU:

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m diffusion_policy.sonic.train \
  --config "$CONFIG" \
  --dataset-path /root/Projects/data/carry-bucket-stereo \
  --output-dir "$OUTPUT" \
  --batch-size 8 \
  --max-steps 20000 \
  --save-every 1000 \
  --save-total-limit 1 \
  --bf16 \
  --use-wandb
```

Two GPUs, only after GPUs 2 and 3 are idle:

```bash
CUDA_VISIBLE_DEVICES=2,3 .venv/bin/torchrun --nproc_per_node=2 --master_port=29503 \
  -m diffusion_policy.sonic.train \
  --config "$CONFIG" \
  --dataset-path /root/Projects/data/carry-bucket-stereo \
  --output-dir "$OUTPUT" \
  --batch-size 4 \
  --gradient-accumulation-steps 4 \
  --max-steps 20000 \
  --bf16 \
  --use-wandb
```

`--batch-size` is per GPU. The second command has global batch size
`4 x 2 x 4 = 32`. The deployable checkpoint is `$OUTPUT/latest.pt`.

For a CPU dependency/data-path check, use `--device cpu --batch-size 1
--num-workers 0 --max-steps 1` with a config whose model fits the host.

## Serve And Bridge

Terminal 1, in this repository:

```bash
.venv/bin/python -m diffusion_policy.sonic.serve_policy \
  --checkpoint outputs/sonic_jepa/latest.pt \
  --device cuda:2 \
  --host 0.0.0.0 \
  --port 8000
```

Terminal 2, expose that websocket backend through the GR00T ZMQ PolicyServer:

```bash
cd /root/Projects/Isaac-GR00T
uv run --no-sync python gr00t/eval/run_sonic_bridge_server.py \
  --backend-host 127.0.0.1 \
  --backend-port 8000 \
  --host 0.0.0.0 \
  --port 5550
```

Terminal 3, launch the official SONIC control path:

```bash
cd /root/Projects/GR00T-WholeBodyControl
python gear_sonic/scripts/launch_inference.py \
  --policy-host 127.0.0.1 \
  --policy-port 5550 \
  --camera-host 192.168.123.164 \
  --tactile-zmq-host 192.168.123.164 \
  --prompt "move the bucket to the next stool"
```

For `sonic_notactile`, add `--no-use-tactile` to the launcher. HTD and JEPA
checkpoints require a fresh `uint8[256]` tactile frame. The bridge rejects any
backend whose state, image, action, horizon, tactile requirement, or protocol
metadata does not match `sonic_vla_v1`.

## Test

```bash
.venv/bin/python -m pytest -q tests/test_sonic_policy.py
```
