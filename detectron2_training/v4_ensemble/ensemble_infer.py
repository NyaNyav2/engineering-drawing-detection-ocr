"""
BBox ensemble inference for engineering drawings.
- Loads multiple Detectron2 models
- Merges predictions with simple weighted box fusion
- Saves visualizations + JSON
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode

BASE_DIR = Path(r"D:\Object Detection & OCR System for Engineering Drawings")
INPUT_DIR = BASE_DIR / "blueprint_yolo" / "images" / "custom_test"
OUT_DIR = BASE_DIR / "detectron2_training" / "output_v4_ensemble" / "ensemble_infer"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CLASS_NAMES = ["PartDrawing", "Note", "Table"]

MODELS = [
    {
        "name": "frcnn_x101",
        "zoo": "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
        "weights": BASE_DIR / "detectron2_training" / "output_v4_ensemble" / "frcnn_x101" / "model_final.pth",
        "score_thresh": 0.30,
        "weight": {0: 1.3, 1: 1.0, 2: 1.2},
    },
    {
        "name": "retinanet_r101",
        "zoo": "COCO-Detection/retinanet_R_101_FPN_3x.yaml",
        "weights": BASE_DIR / "detectron2_training" / "output_v4_ensemble" / "retinanet_r101" / "model_final.pth",
        "score_thresh": 0.20,
        "weight": {0: 0.9, 1: 1.2, 2: 1.0},
    },
    {
        "name": "frcnn_r101",
        "zoo": "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
        "weights": BASE_DIR / "detectron2_training" / "output_v4_ensemble" / "frcnn_r101" / "model_final.pth",
        "score_thresh": 0.25,
        "weight": {0: 1.0, 1: 1.0, 2: 1.0},
    },
]

IOU_BY_CLASS = {0: 0.55, 1: 0.45, 2: 0.60}


def iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(1e-6, area_a + area_b - inter)


def simple_wbf(dets: List[Dict[str, Any]], iou_thr: float):
    dets = sorted(dets, key=lambda d: d['score'], reverse=True)
    fused = []
    used = [False] * len(dets)
    for i, det in enumerate(dets):
        if used[i]:
            continue
        cluster = [det]
        used[i] = True
        for j in range(i + 1, len(dets)):
            if used[j]:
                continue
            if iou_xyxy(det['bbox'], dets[j]['bbox']) >= iou_thr:
                cluster.append(dets[j])
                used[j] = True
        score_sum = sum(c['score'] * c.get('model_weight', 1.0) for c in cluster)
        weight_sum = sum(c.get('model_weight', 1.0) for c in cluster)
        box = [
            sum(c['bbox'][k] * c['score'] * c.get('model_weight', 1.0) for c in cluster) / max(1e-6, score_sum)
            for k in range(4)
        ]
        fused.append({
            'class_id': det['class_id'],
            'class_name': det['class_name'],
            'score': round(float(sum(c['score'] for c in cluster) / len(cluster)), 4),
            'bbox': [round(float(x), 1) for x in box],
            'votes': len(cluster),
            'sources': [c['model'] for c in cluster],
            'ensemble_weight': round(float(weight_sum), 3),
        })
    return fused


def build_predictor(zoo_cfg: str, weights: Path, score_thresh: float):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(zoo_cfg))
    if 'retinanet' in zoo_cfg.lower():
        cfg.MODEL.RETINANET.NUM_CLASSES = len(CLASS_NAMES)
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = score_thresh
    else:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASS_NAMES)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.WEIGHTS = str(weights)
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    return DefaultPredictor(cfg)


def collect_predictions(predictor, model_spec, img_bgr):
    outputs = predictor(img_bgr)
    inst = outputs['instances'].to('cpu')
    if not inst.has('scores'):
        return []
    scores = inst.scores.numpy()
    classes = inst.pred_classes.numpy()
    boxes = inst.pred_boxes.tensor.numpy()
    out = []
    for box, score, cls_id in zip(boxes, scores, classes):
        out.append({
            'model': model_spec['name'],
            'model_weight': model_spec['weight'][int(cls_id)],
            'class_id': int(cls_id),
            'class_name': CLASS_NAMES[int(cls_id)],
            'score': float(score),
            'bbox': [float(v) for v in box],
        })
    return out


def main():
    MetadataCatalog.get('ensemble_meta').set(thing_classes=CLASS_NAMES)
    metadata = MetadataCatalog.get('ensemble_meta')

    predictors = []
    for spec in MODELS:
        if not spec['weights'].exists():
            print(f"[WARN] skip missing weights: {spec['weights']}")
            continue
        print(f"[LOAD] {spec['name']} -> {spec['weights']}")
        predictors.append((spec, build_predictor(spec['zoo'], spec['weights'], spec['score_thresh'])))

    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.JPG', '*.PNG', '*.JPEG']:
        image_paths.extend(INPUT_DIR.glob(ext))
    image_paths = sorted({str(p).lower(): p for p in image_paths}.values(), key=lambda p: p.name.lower())

    summary = []
    for img_path in image_paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        all_dets = []
        for spec, predictor in predictors:
            all_dets.extend(collect_predictions(predictor, spec, img_bgr))

        fused = []
        for cls_id in range(len(CLASS_NAMES)):
            cls_dets = [d for d in all_dets if d['class_id'] == cls_id]
            fused.extend(simple_wbf(cls_dets, IOU_BY_CLASS[cls_id]))

        vis_instances = {
            'boxes': [f['bbox'] for f in fused],
            'classes': [f['class_id'] for f in fused],
            'scores': [f['score'] for f in fused],
        }
        vis = img_bgr.copy()
        for f in fused:
            x1, y1, x2, y2 = [int(round(v)) for v in f['bbox']]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"{f['class_name']} {f['score']:.2f}", (x1, max(16, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
        out_img = OUT_DIR / f"{img_path.stem}_ensemble.jpg"
        cv2.imwrite(str(out_img), vis)
        out_json = OUT_DIR / f"{img_path.stem}_ensemble.json"
        out_json.write_text(json.dumps({'image': img_path.name, 'detections': fused}, indent=2, ensure_ascii=False), encoding='utf-8')
        summary.append({'image': img_path.name, 'num_detections': len(fused), 'output_image': str(out_img), 'output_json': str(out_json)})
        print(f"[DONE] {img_path.name} -> {len(fused)} fused dets")

    (OUT_DIR / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"[OUT] {OUT_DIR}")


if __name__ == '__main__':
    main()
