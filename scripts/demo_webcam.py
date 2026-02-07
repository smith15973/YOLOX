#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _device import pick_best_device


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    root = repo_root()

    parser = argparse.ArgumentParser(description="YOLOX webcam demo launcher (auto device).")
    parser.add_argument("--camid", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--nms", type=float, default=0.30)
    parser.add_argument("--tsize", type=int, default=640)

    # If user doesn't pass --device, auto-select
    parser.add_argument("--device", default=None, help="cpu | gpu | mps (default: auto)")

    parser.add_argument("--exp", default="exps/example/custom/yolox_s_basketball.py")
    parser.add_argument("--ckpt", default="YOLOX_outputs/yolox_s_basketball/best_ckpt.pth")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--fuse", action="store_true")
    args = parser.parse_args()

    if args.device is None:
        args.device = pick_best_device()

    demo_py = root / "tools" / "demo.py"
    exp_file = root / args.exp
    ckpt_file = root / args.ckpt

    for p in (demo_py, exp_file, ckpt_file):
        if not p.exists():
            print(f"ERROR: missing file: {p}")
            return 2

    cmd = [
        sys.executable, str(demo_py), "webcam",
        "-f", str(exp_file),
        "-c", str(ckpt_file),
        "--camid", str(args.camid),
        "--conf", str(args.conf),
        "--nms", str(args.nms),
        "--device", str(args.device),
        "--tsize", str(args.tsize),
    ]
    if args.fp16:
        cmd.append("--fp16")
    if args.fuse:
        cmd.append("--fuse")

    print("\nRunning:\n  " + " ".join(cmd) + "\n")
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
