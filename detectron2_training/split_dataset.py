"""
Split annotations_human.json -> train.json + val.json (80/20)
"""
import json
import random
from pathlib import Path

SEED = 42
VAL_RATIO = 0.2

DATA_DIR = Path(r"D:\Object Detection & OCR System for Engineering Drawings\labeled_export")
ANN_FILE = DATA_DIR / "annotations_human.json"
OUT_DIR = Path(r"D:\Object Detection & OCR System for Engineering Drawings\detectron2_training")

with open(ANN_FILE) as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]
categories = coco["categories"]

random.seed(SEED)
random.shuffle(images)

n_val = max(1, int(len(images) * VAL_RATIO))
val_images = images[:n_val]
train_images = images[n_val:]

val_ids = {img["id"] for img in val_images}
train_ids = {img["id"] for img in train_images}

train_anns = [a for a in annotations if a["image_id"] in train_ids]
val_anns   = [a for a in annotations if a["image_id"] in val_ids]

def make_coco(imgs, anns, cats):
    return {"images": imgs, "annotations": anns, "categories": cats}

with open(OUT_DIR / "train.json", "w") as f:
    json.dump(make_coco(train_images, train_anns, categories), f, indent=2)

with open(OUT_DIR / "val.json", "w") as f:
    json.dump(make_coco(val_images, val_anns, categories), f, indent=2)

print(f"Train: {len(train_images)} images, {len(train_anns)} annotations")
print(f"Val:   {len(val_images)} images, {len(val_anns)} annotations")
print("Saved train.json and val.json")
