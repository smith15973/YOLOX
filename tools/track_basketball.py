#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

"""
=========================== YOLOX + BYTE TRACK DEMO ===========================

WEBCAM TRACKING (CPU)
python -m tools.demo_track webcam \
  -f exps/example/custom/yolox_nano_basketball.py \
  -c models/best_nano_ckpt.pth \
  --camid 0 --conf 0.3 --nms 0.3 --tsize 640 \
  --device cpu \
  --track_class ball

WEBCAM TRACKING (MPS – Apple Silicon)
python -m tools.demo_track webcam \
  -f exps/example/custom/yolox_nano_basketball.py \
  -c models/best_nano_ckpt.pth \
  --camid 0 --conf 0.3 --nms 0.3 --tsize 640 \
  --device mps \
  --track_class ball

WEBCAM TRACKING (NVIDIA CUDA GPU)
python -m tools.demo_track webcam \
  -f exps/example/custom/yolox_nano_basketball.py \
  -c models/best_nano_ckpt.pth \
  --camid 0 --conf 0.3 --nms 0.3 --tsize 640 \
  --device gpu --fp16 \
  --track_class ball

VIDEO FILE TRACKING
python -m tools.demo_track video \
  -f exps/example/custom/yolox_nano_basketball.py \
  -c models/best_nano_ckpt.pth \
  --path ./video.mp4 \
  --conf 0.3 --nms 0.3 --tsize 640 \
  --track_class ball \
  --save_result

TRACKING OPTIONS
--track_class     ball | human | rim | all
--track_thresh    Tracking confidence threshold (default 0.5)
--match_thresh    Association matching threshold (default 0.8)
--track_buffer    Frames to keep lost tracks alive (default 30)
--mot20           Enable MOT20 association mode (usually False)

NOTES
- device: cpu | gpu | mps
- gpu = NVIDIA CUDA
- tsize 640 = good speed/accuracy balance
- For sports tracking, --track_class ball is recommended
- Lower --track_thresh (0.2–0.4) if small objects disappear
- Tracking IDs persist across frames while visible

===============================================================================
"""


import argparse
import os
import time
from loguru import logger

import cv2
import torch
from numpy.ma.core import filled

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
import numpy as np
from yolox.tracker.byte_tracker import BYTETracker

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

def calc_velocity(initial_pos: float, final_pos: float, dt: float) -> float:
    if dt == 0.0 or dt == 0:
        return 0.0
    return (float(final_pos) - float(initial_pos)) / float(dt)


@dataclass
class Detection:
    frame: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0


class Basketball:
    def __init__(self, track_id):
        self.track_id: int = track_id
        self.detections: List[Detection] = []
        self.last_possession: Tuple[int, int] = (0, 0)
        self.x: float = 0.0
        self.y: float = 0.0
        self.last_seen: int = 0

    def add_detection(self, frame: int, x: float, y: float, vx: float = 0.0, vy: float = 0.0):
        if frame < self.last_seen:
            return
        self.detections.append(Detection(frame, x, y, vx, vy))
        self.last_seen = frame
        self.x = x
        self.y = y

    def get_detection(self, frame: int) -> Optional[Detection]:
        for detection in self.detections:
            if detection.frame == frame:
                return detection
        return None

    def get_velocity(self, frame: int) -> Tuple[float, float]:
        """Get velocity at given frame"""
        curr_detection = self.get_detection(frame)
        if curr_detection is None:
            return 0.0, 0.0
        
        curr_detection_idx = self.detections.index(curr_detection)
        if curr_detection_idx == 0:
            return 0.0, 0.0
        
        prev_detection = self.detections[curr_detection_idx-1]
        
        frame_gap = curr_detection.frame - prev_detection.frame
        if frame_gap == 0:
            return 0.0, 0.0

        vx = calc_velocity(prev_detection.x, prev_detection.x, float(frame_gap))
        vy = calc_velocity(prev_detection.y, prev_detection.y, float(frame_gap))
        
        return vx, vy

    def get_velocity_at_index(self, detection_idx: int) -> Tuple[float, float]:
        """Get velocity at given detection index"""
        if detection_idx <= 0 or detection_idx >= len(self.detections):
            return 0.0, 0.0

        curr_detection = self.detections[detection_idx]
        prev_detection = self.detections[detection_idx - 1]

        frame_gap = curr_detection.frame - prev_detection.frame
        if frame_gap == 0:
            return 0.0, 0.0
        vx = calc_velocity(prev_detection.x, prev_detection.x, float(frame_gap))
        vy = calc_velocity(prev_detection.y, prev_detection.y, float(frame_gap))

        return vx, vy

    def get_last_seen_frame(self) -> Optional[int]:
        return self.detections[-1].frame if self.detections else None





IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument("demo", default="image", help="demo type, eg. image, video and webcam")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument("--path", default="./assets/dog.jpg", help="path to images or video")
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run model: cpu | gpu (CUDA) | mps (Apple Silicon)",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating (CUDA only).",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing (CUDA only).",
    )
    
    parser.add_argument("--track_thresh", type=float, default=0.5)
    parser.add_argument("--track_buffer", type=int, default=30)
    parser.add_argument("--match_thresh", type=float, default=0.8)
    parser.add_argument("--mot20", action="store_true")
    parser.add_argument("--track_class", type=str, default="all", help="ball | human | rim | all")

    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def _move_model(model, device: str, fp16: bool):
    device = device.lower()
    if device in ("gpu", "cuda"):
        model.cuda()
        if fp16:
            model.half()
        return model

    if device == "mps":
        # FP16 on MPS is not consistently beneficial; keep fp32
        model.to("mps")
        return model

    # cpu
    model.cpu()
    return model


def _move_tensor(x: torch.Tensor, device: str, fp16: bool) -> torch.Tensor:
    device = device.lower()
    if device in ("gpu", "cuda"):
        x = x.cuda()
        if fp16:
            x = x.half()
        return x
    if device == "mps":
        return x.to("mps")
    return x


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)

        if trt_file is not None:
            # TRT is CUDA-only; keep the old behavior
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0).float()

        img = _move_tensor(img, self.device, self.fp16)

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res


def image_demo(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if save_result:
            save_folder = os.path.join(vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time))
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
        ch = cv2.waitKey(0)
        if ch == 27 or ch in (ord("q"), ord("Q")):
            break


def imageflow_track_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1:
        fps = 30

    ball_tracker = BYTETracker(args, frame_rate=int(fps))
    human_tracker = BYTETracker(args, frame_rate=int(fps))
    rim_tracker = BYTETracker(args, frame_rate=int(fps))

    vid_writer = None
    if args.save_result:
        save_folder = os.path.join(vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time))
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, "track.mp4" if args.demo != "video" else os.path.basename(args.path))
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )

    # helper: map class name -> class id
    def get_track_class_id():
        if args.track_class.lower() == "all":
            return None
        if args.track_class in predictor.cls_names:
            return predictor.cls_names.index(args.track_class)
        return None

    track_cls_id = get_track_class_id()
    if args.track_class.lower() != "all" and track_cls_id is None:
        logger.warning(f"track_class='{args.track_class}' not found in cls_names; tracking nothing.")

    frame_num = 0
    classes = {'ball': 0, 'human': 1, 'rim': 2}
    basketballs: Dict[int, Basketball] = {}
    while True:
        ret_val, frame = cap.read()
        if not ret_val:
            break

        outputs, img_info = predictor.inference(frame)

        # base image to draw on
        result_frame = img_info["raw_img"].copy()
        ratio = img_info["ratio"]

        ball_targets = []
        human_targets = []
        rim_targets = []
        if outputs[0] is not None:
            out = outputs[0].cpu().numpy()

            # YOLOX postprocess output: [x1, y1, x2, y2, obj_conf, cls_conf, cls_id]
            bboxes = out[:, 0:4]
            scores = out[:, 4] * out[:, 5]
            clses = out[:, 6].astype(int)

            for cls_id, cls_name in [(0, 'ball'), (1, 'human'), (2, 'rim')]:
                keep = (clses == cls_id)
                if keep.sum() > 0:
                    cls_bboxes = bboxes[keep]
                    cls_scores = scores[keep]
                    dets = np.concatenate([cls_bboxes, cls_scores.reshape(-1, 1)], axis=1).astype(np.float32)



                    if cls_name == 'ball':
                        ball_targets = ball_tracker.update(
                            dets,
                            [img_info["height"], img_info["width"]],
                            predictor.test_size,
                        )
                    elif cls_name == 'human':
                        human_targets = human_tracker.update(
                            dets,
                            [img_info["height"], img_info["width"]],
                            predictor.test_size,
                        )
                    elif cls_name == 'rim':
                        rim_targets = rim_tracker.update(
                            dets,
                            [img_info["height"], img_info["width"]],
                            predictor.test_size,
                        )



        # draw tracks
        for t in ball_targets + human_targets + rim_targets:
            x, y, w, h = t.tlwh
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            cx, cy = int(x + (w / 2)), int(y + (h / 2))
            tid = t.track_id

            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                result_frame, f"ID {tid}", (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )


        for t in ball_targets:
            x, y, w, h = t.tlwh
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            cx, cy = float(x + (w / 2.0)), float(y + (h / 2.0))
            if t.track_id not in basketballs:
                basketballs[t.track_id] = Basketball(t.track_id)

            bball = basketballs[t.track_id]

            prev_x, prev_y = bball.x, bball.y
            dt = frame_num - bball.last_seen
            vx = calc_velocity(prev_x, cx, dt)
            vy = calc_velocity(prev_y, cy, dt)

            basketballs[t.track_id].add_detection(frame_num, cx, cy, vx, vy)  # mean = [x, y, a, h, vx, vy, va, vh]

            print(f"Raw: {vx:.2f}, {vy:.2f}\n")
            cv2.circle(result_frame, (int(cx), int(cy)), 5, (0, 255, 0), -1)
            cv2.putText(
                result_frame, f"({vx}, {vy})", (x1, max(0, y2 + 30)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )


        # for track_id, bball in basketballs.items():
        #     vx, vy = bball.get_velocity(frame_num)
        #     print(f"Basketball {track_id} at frame {frame_num}: velocity = ({vx:.2f}, {vy:.2f})")




        if args.save_result:
            vid_writer.write(result_frame)
        else:
            cv2.namedWindow("yolox_track", cv2.WINDOW_NORMAL)
            cv2.imshow("yolox_track", result_frame)

        ch = cv2.waitKey(1)
        frame_num += 1
        if ch == 27 or ch in (ord("q"), ord("Q")):
            break
    print(basketballs)



def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    # TRT implies CUDA
    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    model = _move_model(model, args.device, args.fp16)
    model.eval()

    if not args.trt:
        ckpt_file = args.ckpt if args.ckpt is not None else os.path.join(file_name, "best_ckpt.pth")
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model does not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(trt_file), "TensorRT model is not found! Run python tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    # ✅ Use custom class names if your Exp provides them; otherwise COCO
    cls_names = getattr(exp, "class_names", COCO_CLASSES)
    logger.info(f"Using class names: {cls_names}")

    predictor = Predictor(
        model, exp, cls_names, trt_file, decoder,
        args.device, args.fp16, args.legacy,
    )

    current_time = time.localtime()
    if args.demo in ("video", "webcam"):
        imageflow_track_demo(predictor, vis_folder, current_time, args)
    else:
        raise ValueError("demo_track.py supports only video/webcam for now")



if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)
