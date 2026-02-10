#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

"""
============================= YOLOX TRAIN USAGE ==============================

TRAIN (AUTO DEVICE — GPU > MPS > CPU)
python -m tools.train -f exps/example/custom/yolox_nano_basketball.py -b 32 --device auto

TRAIN (CPU ONLY)
python -m tools.train -f exps/example/custom/yolox_nano_basketball.py -b 16 --device cpu

TRAIN (APPLE SILICON / MPS)
python -m tools.train -f exps/example/custom/yolox_nano_basketball.py -b 32 --device mps

TRAIN (NVIDIA CUDA GPU, FP16)
python -m tools.train -f exps/example/custom/yolox_nano_basketball.py -b 64 --device gpu --fp16

RESUME TRAINING FROM CHECKPOINT
python -m tools.train -f exps/example/custom/yolox_nano_basketball.py -b 32 --device auto --resume -c YOLOX_outputs/yolox_nano_basketball/latest_ckpt.pth

SPECIFY EXPERIMENT NAME
python -m tools.train -f exps/example/custom/yolox_nano_basketball.py -b 32 --device auto -expn my_run_name

CACHE DATASET IN RAM (FASTER)
python -m tools.train -f exps/example/custom/yolox_nano_basketball.py -b 32 --device auto --cache ram

MULTI-GPU (CUDA ONLY)
python -m tools.train -f exps/example/custom/yolox_nano_basketball.py -b 64 --device gpu -d 2

NOTES
- device: auto | cpu | mps | gpu
- gpu = NVIDIA CUDA only
- mps = Apple Silicon acceleration
- fp16 only works on CUDA
- batch size depends heavily on VRAM / RAM

===============================================================================
"""


import argparse
import os
import random
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn

from yolox.core import launch
from yolox.exp import Exp, check_exp_value, get_exp
from yolox.utils import configure_module, configure_nccl, configure_omp, get_num_devices


def _auto_device() -> str:
    """Return best available device string: gpu (CUDA) -> mps -> cpu"""
    if torch.cuda.is_available():
        return "gpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # NEW: device selection
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "gpu", "cpu", "mps"],
        help="Training device: auto|gpu(CUDA)|mps(Apple)|cpu",
    )

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="number of devices for training (CUDA only)"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="plz input your experiment description file",
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume training"
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument(
        "-e",
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision training (CUDA only).",
    )
    parser.add_argument(
        "--cache",
        type=str,
        nargs="?",
        const="ram",
        help="Caching imgs to ram/disk for fast training.",
    )
    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=False,
        action="store_true",
        help="occupy GPU memory first for training.",
    )
    parser.add_argument(
        "-l",
        "--logger",
        type=str,
        help="Logger to be used for metrics. Implemented: tensorboard, mlflow, wandb",
        default="tensorboard"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


@logger.catch
def main(exp: Exp, args):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # Configure distributed env vars (safe to call even if not using NCCL)
    configure_nccl()
    configure_omp()

    # Only makes sense for CUDA
    cudnn.benchmark = True

    trainer = exp.get_trainer(args)
    trainer.train()


if __name__ == "__main__":
    configure_module()
    args = make_parser().parse_args()

    # Resolve device choice
    device = args.device.lower()
    if device == "auto":
        device = _auto_device()

    # If user asked for GPU but CUDA isn't available, fall back cleanly
    if device == "gpu" and not torch.cuda.is_available():
        logger.warning("CUDA not available; falling back to CPU.")
        device = "cpu"

    # If user asked for MPS but it's not available, fall back cleanly
    if device == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            logger.warning("MPS not available; falling back to CPU.")
            device = "cpu"

    # Store resolved device back into args so exp/trainer can read it
    args.device = device

    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)
    check_exp_value(exp)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    if args.cache is not None:
        exp.dataset = exp.get_dataset(cache=True, cache_type=args.cache)

    # ---- CUDA path: keep original distributed launch behavior ----
    if args.device == "gpu":
        num_gpu = get_num_devices() if args.devices is None else args.devices
        assert num_gpu <= get_num_devices()

        dist_url = "auto" if args.dist_url is None else args.dist_url
        launch(
            main,
            num_gpu,
            args.num_machines,
            args.machine_rank,
            backend=args.dist_backend,
            dist_url=dist_url,
            args=(exp, args),
        )

    # ---- Non-CUDA path: run single-process (no torch.cuda assumptions) ----
    else:
        # Disable CUDA visibility just in case someone has a weird setup
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        # fp16 in this repo is CUDA-focused; disable for cpu/mps
        if args.fp16:
            logger.warning("FP16 flag is CUDA-only in this training setup; disabling fp16.")
            args.fp16 = False

        # Use gloo if anything in trainer tries to init distributed
        if args.dist_backend == "nccl":
            args.dist_backend = "gloo"

        logger.info(f"Running single-process training on device='{args.device}' (no distributed launch).")
        main(exp, args)
