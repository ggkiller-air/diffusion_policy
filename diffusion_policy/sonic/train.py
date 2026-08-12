from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from diffusion_policy.sonic.checkpoint import load_checkpoint_payload, save_checkpoint
from diffusion_policy.sonic.config import SonicConfig, TactileMode
from diffusion_policy.sonic.dataset import (
    SonicLeRobotDataset,
    build_training_cache,
    compute_normalization_stats,
)
from diffusion_policy.sonic.policy import SonicDiffusionPolicy

LOGGER = logging.getLogger(__name__)


def split_episode_indices(dataset_path: str | Path, val_ratio: float, seed: int) -> tuple[set[int], set[int]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    with (Path(dataset_path) / "meta" / "episodes.jsonl").open(encoding="utf-8") as handle:
        episode_ids = [int(json.loads(line)["episode_index"]) for line in handle if line.strip()]
    if len(episode_ids) < 2:
        raise ValueError("best-model validation requires at least two episodes")
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(episode_ids, dtype=np.int64)
    rng.shuffle(shuffled)
    val_count = min(len(shuffled) - 1, max(1, round(len(shuffled) * val_ratio)))
    val_ids = set(shuffled[:val_count].tolist())
    return set(shuffled[val_count:].tolist()), val_ids


@torch.no_grad()
def evaluate_action_mse(model, loader, device, *, seed: int, max_batches: int, world_size: int) -> float:
    was_training = model.training
    model.eval()
    squared_error = torch.zeros((), dtype=torch.float64, device=device)
    count = torch.zeros((), dtype=torch.float64, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        batch = _move_batch(batch, device)
        predicted = model.predict_actions(batch, generator=generator)
        target = batch["actions"].float()
        predicted = model._normalize_action(predicted)
        target = model._normalize_action(target)
        squared_error += (predicted.double() - target.double()).square().sum()
        count += target.numel()
    if world_size > 1:
        dist.all_reduce(squared_error)
        dist.all_reduce(count)
    model.train(was_training)
    if count.item() == 0:
        raise RuntimeError("validation loader produced no actions")
    return (squared_error / count).item()


def _distributed_setup() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    return rank, local_rank, world_size


def _resolve_device(requested: str, local_rank: int) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        device = (
            torch.device("cuda", local_rank)
            if requested == "cuda"
            else torch.device(requested)
        )
        torch.cuda.set_device(device)
        return device
    return torch.device(requested)


def _move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _normalization_stats(dataset_path: str, rank: int, world_size: int) -> dict:
    stats = compute_normalization_stats(dataset_path) if rank == 0 else None
    if world_size > 1:
        values = [stats]
        dist.broadcast_object_list(values, src=0)
        stats = values[0]
    return stats


def _load_config(args) -> SonicConfig:
    config = SonicConfig.from_json(args.config)
    if args.mode is not None:
        config = dataclasses.replace(config, mode=TactileMode(args.mode))
    return config


def train(args) -> None:
    rank, local_rank, world_size = _distributed_setup()
    device = _resolve_device(args.device, local_rank)
    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config = _load_config(args)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("SONIC config: %s", json.dumps(config.to_dict(), indent=2))

    stats = _normalization_stats(args.dataset_path, rank, world_size)
    if rank == 0:
        cache_start = time.perf_counter()
        lowdim_cache_root, video_cache_root = build_training_cache(
            args.dataset_path, config, args.video_cache_dir, progress=True
        )
        LOGGER.info(
            "Training caches ready at %s and %s in %.1f seconds",
            lowdim_cache_root,
            video_cache_root,
            time.perf_counter() - cache_start,
        )
    if world_size > 1:
        dist.barrier()
    train_episode_ids, val_episode_ids = split_episode_indices(
        args.dataset_path, args.val_ratio, args.seed
    )
    dataset = SonicLeRobotDataset(
        args.dataset_path,
        config,
        video_cache_dir=args.video_cache_dir,
        episode_indices=train_episode_ids,
    )
    sampler = (
        DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed
        )
        if world_size > 1
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        drop_last=True,
    )
    if len(loader) == 0:
        raise ValueError("batch_size is larger than the available dataset")
    val_dataset = SonicLeRobotDataset(
        args.dataset_path,
        config,
        video_cache_dir=args.video_cache_dir,
        episode_indices=val_episode_ids,
    )
    val_sampler = (
        DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )
        if world_size > 1
        else None
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        drop_last=False,
    )

    model = SonicDiffusionPolicy(config).to(device)
    model.set_normalization_stats(stats)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    step = 0
    if args.resume:
        payload = load_checkpoint_payload(args.resume, device)
        checkpoint_config = SonicConfig.from_dict(payload["config"])
        if checkpoint_config != config:
            raise ValueError(
                "resume checkpoint config does not match the selected experiment config"
            )
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        step = int(payload["step"])
    training_model = (
        DistributedDataParallel(model, device_ids=[device.index])
        if world_size > 1 and device.type == "cuda"
        else DistributedDataParallel(model)
        if world_size > 1
        else model
    )
    best_metadata_path = output_dir / "best_model" / "metrics.json"
    best_mse = float("inf")
    if best_metadata_path.is_file():
        with best_metadata_path.open(encoding="utf-8") as handle:
            best_mse = float(json.load(handle)["val_action_mse"])

    wandb_run = None
    if args.use_wandb and rank == 0:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.run_name or f"dp-sonic-{config.mode.value}",
            config={**config.to_dict(), **vars(args)},
        )

    epoch = 0
    micro_step = 0
    interval_data_time = 0.0
    interval_step_time = 0.0
    interval_steps = 0
    data_start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    while step < args.max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            data_end = time.perf_counter()
            step_start = data_end
            interval_data_time += data_end - data_start
            batch = _move_batch(batch, device)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if args.bf16 and device.type == "cuda"
                else nullcontext()
            )
            with autocast:
                losses = training_model(batch)
                loss = losses["loss"] / args.gradient_accumulation_steps
            loss.backward()
            micro_step += 1
            if micro_step % args.gradient_accumulation_steps != 0:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            model.update_teacher()
            step += 1
            interval_step_time += time.perf_counter() - step_start
            interval_steps += 1

            if rank == 0 and (step == 1 or step % args.log_every == 0):
                metrics = {
                    name: float(value.detach()) for name, value in losses.items()
                }
                elapsed = interval_data_time + interval_step_time
                metrics.update(
                    {
                        "data_time": interval_data_time / interval_steps,
                        "step_time": interval_step_time / interval_steps,
                        "global_samples_per_second": (
                            interval_steps * args.batch_size * world_size / elapsed
                        ),
                    }
                )
                LOGGER.info(
                    "step=%d %s",
                    step,
                    " ".join(f"{key}={value:.6f}" for key, value in metrics.items()),
                )
                if wandb_run is not None:
                    wandb_run.log(metrics, step=step)
                interval_data_time = 0.0
                interval_step_time = 0.0
                interval_steps = 0
            if rank == 0 and args.save_every > 0 and step % args.save_every == 0:
                step_path = output_dir / f"checkpoint-{step:08d}.pt"
                save_checkpoint(step_path, model, step=step, optimizer=optimizer)
                save_checkpoint(
                    output_dir / "latest.pt", model, step=step, optimizer=optimizer
                )
                old = sorted(output_dir.glob("checkpoint-*.pt"))[
                    : -args.save_total_limit
                ]
                for path in old:
                    path.unlink()
            if args.eval_every > 0 and step % args.eval_every == 0:
                val_mse = evaluate_action_mse(
                    model,
                    val_loader,
                    device,
                    seed=args.seed + 10_000,
                    max_batches=args.val_batches,
                    world_size=world_size,
                )
                if rank == 0:
                    LOGGER.info("step=%d val_action_mse=%.8f", step, val_mse)
                    if wandb_run is not None:
                        wandb_run.log({"val/action_mse": val_mse}, step=step)
                    if val_mse < best_mse:
                        best_mse = val_mse
                        best_dir = output_dir / "best_model"
                        save_checkpoint(best_dir / "checkpoint.pt", model, step=step)
                        temporary = best_dir / "metrics.json.tmp"
                        with temporary.open("w", encoding="utf-8") as handle:
                            json.dump({"step": step, "val_action_mse": val_mse}, handle, indent=2)
                            handle.write("\n")
                        temporary.replace(best_metadata_path)
                        LOGGER.info("updated best_model at step=%d", step)
                if world_size > 1:
                    dist.barrier()
            if step >= args.max_steps:
                break
            data_start = time.perf_counter()
        epoch += 1

    if rank == 0:
        save_checkpoint(output_dir / "latest.pt", model, step=step, optimizer=optimizer)
        if wandb_run is not None:
            wandb_run.finish()
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Diffusion Policy on sonic_vla_v1 data"
    )
    parser.add_argument("--config", required=True, help="One of configs/sonic_*.json")
    parser.add_argument(
        "--mode", choices=[mode.value for mode in TactileMode], default=None
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument(
        "--video-cache-dir",
        default=None,
        help="Optional mmap video cache directory (default: <dataset>/.cache/diffusion_policy/...)",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Per-process batch size"
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=25000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--val-batches", type=int, default=8)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="univlat")
    parser.add_argument("--run-name", default=None)
    return parser


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    train(build_argparser().parse_args())
