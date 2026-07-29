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

## Full training

The measured four-A800 setting uses batch 32 per GPU, global batch 128, four workers per
rank, BF16, 20k optimizer steps, and one retained checkpoint. Batch 64 per GPU is slower
because video decoding becomes the bottleneck.

```bash
cd /root/Projects/diffusion_policy
export CUDA_VISIBLE_DEVICES=0,1,2,3

.venv/bin/torchrun --nproc_per_node=4 --master_port=29511 \
  -m diffusion_policy.sonic.train \
  --config configs/sonic_notactile.json \
  --dataset-path /root/Projects/data/carry-bucket-stereo \
  --output-dir outputs/sonic_notactile \
  --batch-size 32 \
  --gradient-accumulation-steps 1 \
  --num-workers 4 \
  --max-steps 20000 \
  --save-every 1000 \
  --save-total-limit 1 \
  --bf16 \
  --use-wandb \
  --wandb-project univlat \
  --run-name dp-sonic-notactile

.venv/bin/torchrun --nproc_per_node=4 --master_port=29512 \
  -m diffusion_policy.sonic.train \
  --config configs/sonic_htd.json \
  --dataset-path /root/Projects/data/carry-bucket-stereo \
  --output-dir outputs/sonic_htd \
  --batch-size 32 \
  --gradient-accumulation-steps 1 \
  --num-workers 4 \
  --max-steps 20000 \
  --save-every 1000 \
  --save-total-limit 1 \
  --bf16 \
  --use-wandb \
  --wandb-project univlat \
  --run-name dp-sonic-htd

.venv/bin/torchrun --nproc_per_node=4 --master_port=29513 \
  -m diffusion_policy.sonic.train \
  --config configs/sonic_jepa.json \
  --dataset-path /root/Projects/data/carry-bucket-stereo \
  --output-dir outputs/sonic_jepa \
  --batch-size 32 \
  --gradient-accumulation-steps 1 \
  --num-workers 4 \
  --max-steps 20000 \
  --save-every 1000 \
  --save-total-limit 1 \
  --bf16 \
  --use-wandb \
  --wandb-project univlat \
  --run-name dp-sonic-jepa
```

`--batch-size` is per GPU. Each command has global batch size `32 x 4 = 128`; the
deployable checkpoint is `outputs/sonic_<mode>/latest.pt`.

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
