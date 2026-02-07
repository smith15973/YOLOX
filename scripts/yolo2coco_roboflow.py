import os
import json
import yaml
from pathlib import Path
from PIL import Image

# ---- EDIT THESE 2 PATHS ONLY ----
SRC = Path(r"datasets\basketball_yolov5_pt")   # Roboflow YOLOv5 PyTorch export
DST = Path(r"datasets\basketball2")            # YOLOX COCO dataset folder
# ---------------------------------

# Your class order (Roboflow YOLOv5 uses index order)
CLASS_NAMES = ["ball", "human", "rim"]

def yolo_line_to_xywh(line, w, h):
    # YOLO: cls xc yc bw bh, normalized [0..1]
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    cls = int(float(parts[0]))
    xc = float(parts[1]) * w
    yc = float(parts[2]) * h
    bw = float(parts[3]) * w
    bh = float(parts[4]) * h

    x = xc - bw / 2.0
    y = yc - bh / 2.0

    # COCO bbox: [x, y, width, height]
    return cls, [x, y, bw, bh]

def convert_split(split_name, img_dir, label_dir, out_img_dir, out_ann_path, starting_img_id=0, starting_ann_id=0):
    images = []
    annotations = []
    img_id = starting_img_id
    ann_id = starting_ann_id

    img_paths = sorted(list(Path(img_dir).glob("*.*")))
    img_paths = [p for p in img_paths if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]]

    for p in img_paths:
        # Copy image into COCO folder
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_img_dir / p.name
        if not out_path.exists():
            out_path.write_bytes(p.read_bytes())

        # Get image size
        with Image.open(p) as im:
            w, h = im.size

        images.append({
            "id": img_id,
            "file_name": p.name,
            "width": w,
            "height": h
        })

        # Labels file
        label_path = Path(label_dir) / (p.stem + ".txt")
        if label_path.exists():
            lines = label_path.read_text(encoding="utf-8").strip().splitlines()
            for line in lines:
                if not line.strip():
                    continue
                parsed = yolo_line_to_xywh(line, w, h)
                if parsed is None:
                    continue
                cls, bbox = parsed

                # Guard: class id must be 0..num_classes-1
                if cls < 0 or cls >= len(CLASS_NAMES):
                    raise ValueError(f"[{split_name}] Bad class id {cls} in {label_path} (expected 0..{len(CLASS_NAMES)-1})")

                x, y, bw, bh = bbox

                # Clamp bbox to image bounds (prevents weird negatives)
                x = max(0.0, min(x, w - 1.0))
                y = max(0.0, min(y, h - 1.0))
                bw = max(0.0, min(bw, w - x))
                bh = max(0.0, min(bh, h - y))

                area = bw * bh

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls,     # 0,1,2 (IMPORTANT)
                    "bbox": [x, y, bw, bh],
                    "area": area,
                    "iscrowd": 0
                })
                ann_id += 1

        img_id += 1

    coco = {
        "info": {"description": "Converted from Roboflow YOLOv5 export"},
        "licenses": [],
        "categories": [{"id": i, "name": n, "supercategory": "none"} for i, n in enumerate(CLASS_NAMES)],
        "images": images,
        "annotations": annotations
    }

    out_ann_path.parent.mkdir(parents=True, exist_ok=True)
    out_ann_path.write_text(json.dumps(coco), encoding="utf-8")

    return img_id, ann_id

def find_split_dirs(src: Path, split: str):
    # Handles both:
    # train/images + train/labels
    # or train/ + labels in same folder (rare)
    split_dir = src / split
    if not split_dir.exists():
        return None, None

    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"

    if img_dir.exists() and lbl_dir.exists():
        return img_dir, lbl_dir

    # fallback: images directly in split dir, labels in split dir
    return split_dir, split_dir

def main():
    # quick check
    if not SRC.exists():
        raise FileNotFoundError(f"SRC not found: {SRC.resolve()}")

    # Convert train/valid/test -> train2017/val2017/test2017
    splits = [
        ("train", "train2017", "instances_train2017.json"),
        ("valid", "val2017", "instances_val2017.json"),
        ("test",  "test2017", "instances_test2017.json"),
    ]

    img_id = 0
    ann_id = 0

    for s, out_folder, out_json in splits:
        img_dir, lbl_dir = find_split_dirs(SRC, s)
        if img_dir is None:
            print(f"Skipping split {s}: not found")
            continue

        out_img_dir = DST / out_folder
        out_ann_path = DST / "annotations" / out_json

        print(f"Converting {s}:")
        print(f"  images: {img_dir}")
        print(f"  labels: {lbl_dir}")
        print(f"  -> {out_img_dir}")
        print(f"  -> {out_ann_path}")

        img_id, ann_id = convert_split(
            s, img_dir, lbl_dir, out_img_dir, out_ann_path, starting_img_id=img_id, starting_ann_id=ann_id
        )

    print("\nDone.")
    print(f"COCO dataset created at: {DST.resolve()}")
    print("Categories:")
    for i, n in enumerate(CLASS_NAMES):
        print(f"  {i}: {n}")

if __name__ == "__main__":
    main()
