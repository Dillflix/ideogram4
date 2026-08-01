from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from ideogram4 import PRESETS, Ideogram4Pipeline, Ideogram4PipelineConfig


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Run a local-caption Ideogram 4 split-device smoke test."
  )
  parser.add_argument(
    "--caption-file",
    type=Path,
    required=True,
    help="UTF-8 file containing an already-structured Ideogram JSON caption.",
  )
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--weights-repo", default="ideogram-ai/ideogram-4-nf4")
  parser.add_argument("--diffusion-device", default="cuda:0")
  parser.add_argument("--text-device", default="cuda:1")
  parser.add_argument("--height", type=int, default=256)
  parser.add_argument("--width", type=int, default=256)
  parser.add_argument(
    "--sampler-preset",
    choices=["custom", *sorted(PRESETS)],
    default="custom",
    help="Use an official preset, or custom for --num-steps/--guidance-scale.",
  )
  parser.add_argument("--num-steps", type=int, default=2)
  parser.add_argument("--guidance-scale", type=float, default=7.0)
  parser.add_argument("--seed", type=int, default=0)
  return parser.parse_args()


def _print_preflight(diffusion_device: str, text_device: str) -> None:
  print(f"torch={torch.__version__}")
  print(f"HIP={torch.version.hip}")
  print(
    "allocator_config="
    f"{os.environ.get('PYTORCH_ALLOC_CONF') or os.environ.get('PYTORCH_CUDA_ALLOC_CONF') or '<default>'}"
  )
  try:
    print(f"allocator_backend={torch.cuda.get_allocator_backend()}")
  except Exception as exc:  # noqa: BLE001 - diagnostics must not block validation
    print(f"allocator_backend=<unavailable: {exc}>")
  print(f"visible_device_count={torch.cuda.device_count()}")
  for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    print(
      f"cuda:{index}: {torch.cuda.get_device_name(index)}, "
      f"VRAM={properties.total_memory / 1024**3:.2f} GiB"
    )

  resolved_diffusion = torch.device(diffusion_device)
  resolved_text = torch.device(text_device)
  if resolved_diffusion != resolved_text:
    if resolved_diffusion.index is None or resolved_text.index is None:
      raise ValueError("Split-device smoke tests require explicit CUDA indexes")
    required_count = max(resolved_diffusion.index, resolved_text.index) + 1
    if torch.cuda.device_count() < required_count:
      raise RuntimeError(
        f"Requested {resolved_diffusion} and {resolved_text}, but only "
        f"{torch.cuda.device_count()} CUDA/ROCm device(s) are visible"
      )
    print(
      f"peer_access {resolved_diffusion}->{resolved_text}="
      f"{torch.cuda.can_device_access_peer(resolved_diffusion.index, resolved_text.index)}"
    )
    print(
      f"peer_access {resolved_text}->{resolved_diffusion}="
      f"{torch.cuda.can_device_access_peer(resolved_text.index, resolved_diffusion.index)}"
    )


def main() -> None:
  args = _parse_args()
  _print_preflight(args.diffusion_device, args.text_device)
  caption = args.caption_file.read_text(encoding="utf-8")

  pipeline = Ideogram4Pipeline.from_pretrained(
    config=Ideogram4PipelineConfig(weights_repo=args.weights_repo),
    device=args.diffusion_device,
    text_device=args.text_device,
    dtype=torch.bfloat16,
  )

  tracked_devices = {
    torch.device(args.diffusion_device),
    torch.device(args.text_device),
  }
  for device in tracked_devices:
    torch.cuda.reset_peak_memory_stats(device)

  generation_kwargs = {
    "num_steps": args.num_steps,
    "guidance_scale": args.guidance_scale,
  }
  if args.sampler_preset != "custom":
    preset = PRESETS[args.sampler_preset]
    generation_kwargs = {
      "num_steps": preset.num_steps,
      "guidance_schedule": preset.guidance_schedule,
      "mu": preset.mu,
      "std": preset.std,
    }
  print(f"generation={args.width}x{args.height}, sampler={args.sampler_preset}")
  images = pipeline(
    caption,
    height=args.height,
    width=args.width,
    seed=args.seed,
    **generation_kwargs,
  )
  args.output.parent.mkdir(parents=True, exist_ok=True)
  images[0].save(args.output)
  print(f"saved={args.output.resolve()}")

  for device in sorted(tracked_devices, key=str):
    peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    print(
      f"peak_memory device={device}, name={torch.cuda.get_device_name(device)}, "
      f"allocated={peak_allocated:.3f} GiB, reserved={peak_reserved:.3f} GiB"
    )


if __name__ == "__main__":
  main()
