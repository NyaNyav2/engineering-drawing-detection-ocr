"""
Test frcnn_x101 (Faster R-CNN X-101-32x8d FPN 3x) on a custom folder of images.
Model from: output_v4_ensemble/frcnn_x101/model_final.pth
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

# ---------------------------------------------------------------------------
BASE_DIR    = Path(r"D:\Object Detection & OCR System for Engineering Drawings")
INPUT_DIR   = BASE_DIR / "blueprint_yolo" / "images" / "custom_test"
OUTPUT_DIR  = BASE_DIR / "detectron2_training" / "output_v4_ensemble" / "frcnn_x101" / "test_results_custom"
MODEL_PATH  = BASE_DIR / "detectron2_training" / "output_v4_ensemble" / "frcnn_x101" / "model_final.pth"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES  = ["PartDrawing", "Note", "Table"]
# Per-class threshold: predictor dùng min, sau đó filter lại từng class
CLASS_THRESH = {
    "PartDrawing": 0.95,   # 95%
    "Note"       : 0.50,   # 50%
    "Table"      : 0.99,   # 99%
}
SCORE_THRESH  = min(CLASS_THRESH.values())   # 0.90 — predictor cutoff
MIN_SIZE_TEST = 960
MAX_SIZE_TEST = 1600
# ---------------------------------------------------------------------------

assert MODEL_PATH.exists(), f"[ERROR] model not found: {MODEL_PATH}"

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES       = len(CLASS_NAMES)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH
cfg.INPUT.MIN_SIZE_TEST               = MIN_SIZE_TEST
cfg.INPUT.MAX_SIZE_TEST               = MAX_SIZE_TEST
cfg.MODEL.WEIGHTS                     = str(MODEL_PATH)
cfg.MODEL.DEVICE                      = "cuda" if torch.cuda.is_available() else "cpu"

META_NAME = "frcnn_x101_meta"
if META_NAME not in MetadataCatalog:
    MetadataCatalog.get(META_NAME).set(thing_classes=CLASS_NAMES)
metadata  = MetadataCatalog.get(META_NAME)
predictor = DefaultPredictor(cfg)

print(f"[OK] Model loaded")
print(f"     device      : {cfg.MODEL.DEVICE}")
print(f"     score_thresh: {SCORE_THRESH}")
print(f"     weights     : {MODEL_PATH}")

# -- Collect images ---------------------------------------------------------
EXTS = ["*.jpg", "*.jpeg", "*.png", "*.PNG", "*.JPG", "*.JPEG", "*.webp"]
all_imgs = []
for ext in EXTS:
    all_imgs.extend(INPUT_DIR.glob(ext))
all_imgs = sorted({str(p).lower(): p for p in all_imgs}.values(), key=lambda p: p.name.lower())
print(f"[DIR] {len(all_imgs)} images in {INPUT_DIR}\n")

# -- Inference loop ---------------------------------------------------------
results      = []
class_counts = {c: 0 for c in CLASS_NAMES}

for i, img_path in enumerate(all_imgs):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"  [{i+1:02d}] SKIP (unreadable): {img_path.name}")
        continue

    outputs   = predictor(img_bgr)
    instances = outputs["instances"].to("cpu")
    scores    = instances.scores.numpy()            if instances.has("scores")       else []
    classes   = instances.pred_classes.numpy()      if instances.has("pred_classes") else []
    boxes_t   = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes")   else []
    # n sẽ được tính lại sau khi filter per-class

    det_by_class = {}
    boxes_list   = []
    keep_mask    = []   # True nếu pass per-class threshold
    for j, (c, s) in enumerate(zip(classes, scores)):
        cname    = CLASS_NAMES[int(c)]
        thr      = CLASS_THRESH[cname]
        if float(s) < thr:
            keep_mask.append(False)
            continue
        keep_mask.append(True)
        det_by_class.setdefault(cname, []).append(round(float(s), 3))
        class_counts[cname] += 1
        if len(boxes_t) > j:
            b = boxes_t[j]
            boxes_list.append({
                "class"    : cname,
                "score"    : round(float(s), 3),
                "bbox_xyxy": [round(float(x), 1) for x in b],
            })

    # Lọc instances để visualizer chỉ vẽ những cái pass per-class threshold
    import torch as _torch
    keep_idx  = [j for j, (c, s) in enumerate(zip(classes, scores))
                 if float(s) >= CLASS_THRESH[CLASS_NAMES[int(c)]]]
    instances = instances[_torch.tensor(keep_idx, dtype=_torch.long)] if keep_idx else instances[:0]
    n         = len(keep_idx)

    # Visualise
    v   = Visualizer(img_bgr[:, :, ::-1], metadata=metadata,
                     scale=1.0, instance_mode=ColorMode.SEGMENTATION)
    out = v.draw_instance_predictions(instances)
    vis = out.get_image()[:, :, ::-1]

    # Header bar
    h, w = vis.shape[:2]
    bar  = np.zeros((36, w, 3), dtype=np.uint8)
    det_str = "  ".join(f"{k}={len(v_)}" for k, v_ in det_by_class.items()) or "no detections"
    label   = f"{img_path.name}  |  {n} det  |  {det_str}"
    cv2.putText(bar, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 200), 2)
    vis = np.vstack([bar, vis])

    out_path = OUTPUT_DIR / f"{img_path.stem}_result.jpg"
    cv2.imwrite(str(out_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 95])

    tag = "  ".join(f"{k}:{len(v_)}" for k, v_ in det_by_class.items()) or "--"
    print(f"  [{i+1:02d}/{len(all_imgs):02d}] {img_path.name:<20}  dets={n:<3}  [{tag}]")

    results.append({
        "image"         : img_path.name,
        "num_detections": int(n),
        "detections"    : det_by_class,
        "boxes"         : boxes_list,
        "saved_to"      : out_path.name,
    })

# -- Summary ----------------------------------------------------------------
summary = {
    "model"                   : "Faster R-CNN X-101-32x8d FPN 3x (frcnn_x101)",
    "weights"                 : str(MODEL_PATH),
    "input_folder"            : str(INPUT_DIR),
    "score_threshold"         : SCORE_THRESH,
    "classes"                 : CLASS_NAMES,
    "num_images_tested"       : len(results),
    "total_detections"        : sum(r["num_detections"] for r in results),
    "detections_per_class"    : class_counts,
    "images_with_detections"  : sum(1 for r in results if r["num_detections"] > 0),
    "results"                 : results,
}
summary_path = OUTPUT_DIR / "results_summary.json"
summary_path.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
)

print()
print("=" * 58)
print(f"[DONE] {len(results)}/{len(all_imgs)} images processed")
print(f"[STATS] Total detections: {summary['total_detections']}")
for cls, cnt in class_counts.items():
    print(f"   {cls:<15}: {cnt}")
print(f"[OUT]   {OUTPUT_DIR}")
print(f"[JSON]  {summary_path}")
print("=" * 58)
