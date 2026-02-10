#!/usr/bin/env python3
"""
Real-time Basketball Tracking with YOLOX + ByteTrack
Tracks ball, human, and rim objects from webcam feed
"""

import argparse
import cv2
import torch
import numpy as np
from collections import defaultdict

# YOLOX imports
from yolox.data.data_augment import ValTransform
from yolox.utils import postprocess
from yolox.tracker.byte_tracker import BYTETracker

# For loading your custom exp
import sys
import os


class BasketballTracker:
    def __init__(self, exp_file, weights_path, conf_thresh=0.5, track_thresh=0.5, 
                 track_buffer=30, match_thresh=0.8, device="cuda"):
        """
        Args:
            exp_file: Path to your yolox_nano_basketball.py
            weights_path: Path to your .pth weights file
            conf_thresh: Detection confidence threshold
            track_thresh: Tracking confidence threshold
            track_buffer: Number of frames to buffer for tracking
            match_thresh: IOU threshold for matching tracks
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.conf_thresh = conf_thresh
        
        # Load experiment config
        self.exp = self._load_exp(exp_file)
        self.class_names = self.exp.class_names
        self.num_classes = self.exp.num_classes
        
        # Load model
        self.model = self._load_model(weights_path)
        
        # Setup preprocessor
        self.preproc = ValTransform(legacy=False)
        self.test_size = self.exp.test_size
        
        # Setup tracker
        self.tracker = BYTETracker(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            frame_rate=30  # Adjust based on your camera
        )
        
        # For visualization
        self.colors = self._get_colors()
        self.track_history = defaultdict(lambda: [])
        
    def _load_exp(self, exp_file):
        """Load experiment configuration from .py file"""
        # Add directory to path
        exp_dir = os.path.dirname(os.path.abspath(exp_file))
        sys.path.insert(0, exp_dir)
        
        # Import the Exp class
        module_name = os.path.splitext(os.path.basename(exp_file))[0]
        exp_module = __import__(module_name)
        exp = exp_module.Exp()
        
        return exp
    
    def _load_model(self, weights_path):
        """Load YOLOX model with weights"""
        model = self.exp.get_model()
        model.eval()
        
        # Load weights
        ckpt = torch.load(weights_path, map_location=self.device)
        model.load_state_dict(ckpt["model"])
        model.to(self.device)
        
        print(f"Model loaded from {weights_path}")
        print(f"Using device: {self.device}")
        
        return model
    
    def _get_colors(self):
        """Generate distinct colors for each class"""
        colors = {
            "ball": (0, 255, 255),    # Yellow
            "human": (0, 255, 0),     # Green
            "rim": (255, 0, 0)        # Blue (BGR format)
        }
        return colors
    
    def preprocess(self, img):
        """Preprocess image for model input"""
        img_info = {"height": img.shape[0], "width": img.shape[1]}
        
        # Apply preprocessing
        img, ratio = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        
        img_info["ratio"] = ratio
        return img, img_info
    
    def detect(self, img):
        """Run detection on preprocessed image"""
        with torch.no_grad():
            outputs = self.model(img)
            outputs = postprocess(
                outputs, self.num_classes, self.conf_thresh, nms_thres=0.45
            )
        return outputs
    
    def track_and_visualize(self, frame, detections, img_info):
        """Update tracker and draw results"""
        if detections[0] is None:
            return frame
        
        # Scale detections back to original image
        detections = detections[0].cpu().numpy()
        detections[:, :4] /= img_info["ratio"]
        
        # Update tracker
        online_targets = self.tracker.update(
            detections[:, :5],  # bbox + score
            [img_info["height"], img_info["width"]],
            self.test_size
        )
        
        # Draw results
        for track in online_targets:
            tlbr = track.tlbr  # top-left, bottom-right
            track_id = track.track_id
            cls_id = int(track.cls) if hasattr(track, 'cls') else 0
            
            # Get class info
            if cls_id < len(self.class_names):
                class_name = self.class_names[cls_id]
                color = self.colors.get(class_name, (255, 255, 255))
            else:
                class_name = f"class_{cls_id}"
                color = (255, 255, 255)
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, tlbr)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name} ID:{track_id}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Store trajectory for ball
            if class_name == "ball":
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                self.track_history[track_id].append(center)
                
                # Keep last 30 points
                if len(self.track_history[track_id]) > 30:
                    self.track_history[track_id].pop(0)
                
                # Draw trajectory
                points = self.track_history[track_id]
                for i in range(1, len(points)):
                    cv2.line(frame, points[i-1], points[i], color, 2)
        
        return frame
    
    def run_webcam(self, camera_id=0, display_fps=True):
        """Run tracking on webcam feed"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        # Set camera properties (optional, adjust as needed)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("Starting basketball tracking...")
        print("Press 'q' to quit")
        print("Press 's' to save current frame")
        
        frame_count = 0
        
        # For FPS calculation
        import time
        fps_time = time.time()
        fps_counter = 0
        current_fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Preprocess
            img, img_info = self.preprocess(frame)
            
            # Detect
            detections = self.detect(img)
            
            # Track and visualize
            frame = self.track_and_visualize(frame, detections, img_info)
            
            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_time > 1.0:
                current_fps = fps_counter / (time.time() - fps_time)
                fps_counter = 0
                fps_time = time.time()
            
            # Display FPS
            if display_fps:
                cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow("Basketball Tracker", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"basketball_frame_{frame_count:04d}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved {filename}")
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"Processed {frame_count} frames")


def main():
    parser = argparse.ArgumentParser(description="Basketball Tracking with YOLOX")
    parser.add_argument("--exp", type=str, required=True, 
                       help="Path to experiment file (yolox_nano_basketball.py)")
    parser.add_argument("--weights", type=str, required=True,
                       help="Path to .pth weights file")
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera ID (default: 0)")
    parser.add_argument("--conf", type=float, default=0.5,
                       help="Detection confidence threshold (default: 0.5)")
    parser.add_argument("--track-thresh", type=float, default=0.5,
                       help="Tracking confidence threshold (default: 0.5)")
    parser.add_argument("--track-buffer", type=int, default=30,
                       help="Tracking buffer size (default: 30)")
    parser.add_argument("--match-thresh", type=float, default=0.8,
                       help="Matching threshold for tracking (default: 0.8)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use: cuda or cpu (default: cuda)")
    parser.add_argument("--no-fps", action="store_true",
                       help="Don't display FPS counter")
    
    args = parser.parse_args()
    
    # Create tracker
    tracker = BasketballTracker(
        exp_file=args.exp,
        weights_path=args.weights,
        conf_thresh=args.conf,
        track_thresh=args.track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        device=args.device
    )
    
    # Run on webcam
    tracker.run_webcam(
        camera_id=args.camera,
        display_fps=not args.no_fps
    )


if __name__ == "__main__":
    main()