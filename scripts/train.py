#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from _device import pick_best_device


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    root = repo_root()

    parser = argparse.ArgumentParser(
        description="YOLOX training launcher (unique run name by default)."
    )
    parser.add_argument(
        "--exp",
        default="exps/example/custom/yolox_s_basketball.py",
        help="Exp file path (relative to YOLOX root).",
    )

    parser.add_argument(
        "--base-name",
        default="yolox_s_basketball",
        help="Prefix for output folder under YOLOX_outputs/ (timestamp will be appended by default).",
    )

    parser.add_argument(
        "--name",
        default=None,
        help="Exact experiment name (no timestamp). If omitted, uses --base-name + timestamp.",
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "gpu", "cpu", "mps"],
        help="auto|gpu(CUDA)|mps(Apple)|cpu",
    )

    # Match your old default: -b 8
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size (default: 8).",
    )

    # Match your old default: -d 1 (CUDA only)
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of CUDA devices for training (CUDA only; default: 1).",
    )

    # Default checkpoint like your old command: -c pretrained/yolox_s.pth
    parser.add_argument(
        "--ckpt",
        default="pretrained/yolox_s.pth",
        help="Checkpoint path (.pth). Default: pretrained/yolox_s.pth",
    )

    # FP16: enabled by default for CUDA, off otherwise.
    # Provide --no-fp16 to disable even on CUDA.
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable fp16 even when using CUDA.",
    )

    parser.add_argument("--resume", action="store_true", help="Resume training.")

    parser.add_argument(
        "--cache",
        default=None,
        choices=[None, "ram", "disk"],
        help="Cache images: ram|disk (matches tools/train.py --cache behavior).",
    )

    parser.add_argument("--occupy", action="store_true", help="Occupy GPU memory first.")

    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Extra YOLOX opts after '--' (passed through). Example: -- -l tensorboard",
    )
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        args.device = pick_best_device()

    # Create unique run name unless user explicitly set --name
    if args.name is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = f"{args.base_name}_{stamp}"
    else:
        run_name = args.name

    train_py = root / "tools" / "train.py"
    exp_file = root / args.exp

    if not train_py.exists():
        print(f"ERROR: missing {train_py}")
        return 2
    if not exp_file.exists():
        print(f"ERROR: missing {exp_file}")
        return 2

    cmd = [
        sys.executable, str(train_py),
        "-f", str(exp_file),
        "-n", run_name,
        "--device", args.device,
        "-b", str(args.batch),
    ]

    # CUDA-only flags
    if args.device == "gpu":
        cmd += ["-d", str(args.devices)]
        if not args.no_fp16:
            cmd += ["--fp16"]
    # Non-CUDA: don't pass fp16; your tools/train.py disables it anyway if passed.

    if args.cache is not None:
        cmd += ["--cache", args.cache]

    if args.occupy:
        cmd += ["--occupy"]

    if args.resume:
        cmd += ["--resume"]

    if args.ckpt:
        # tools/train.py uses -c / --ckpt
        cmd += ["-c", str(root / args.ckpt if not Path(args.ckpt).is_absolute() else args.ckpt)]

    # Pass through extra args after "--"
    extra = args.extra
    if extra and extra[0] == "--":
        extra = extra[1:]
    cmd += extra

    print(f"\nRun name: {run_name}")
    print("Running:\n  " + " ".join(cmd) + "\n")
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
