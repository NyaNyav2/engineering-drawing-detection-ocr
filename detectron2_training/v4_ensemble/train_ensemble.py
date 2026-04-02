"""
Train Detectron2 models for v4 bbox ensemble.
Supports:
- faster_rcnn_X_101_32x8d_FPN_3x
- retinanet_R_101_FPN_3x
- faster_rcnn_R_101_FPN_3x
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from detectron2.utils.logger import setup_logger
setup_logger()

import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
import detectron2.data.transforms as T

BASE_DIR = Path(r"D:\Object Detection & OCR System for Engineering Drawings")
DATA_DIR = BASE_DIR / "labeled_export" / "backup_images"
TRAIN_JSON = BASE_DIR / "detectron2_training" / "train.json"
VAL_JSON = BASE_DIR / "detectron2_training" / "val.json"
ROOT_OUT = BASE_DIR / "detectron2_training" / "output_v4_ensemble"
CLASS_NAMES = ["PartDrawing", "Note", "Table"]

MODEL_SPECS = {
    "frcnn_x101": {
        "zoo": "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
        "base_lr": 0.00020,
        "max_iter": 6000,
        "steps": (4000, 5200),
        "warmup": 300,
        "ims_per_batch": 2,
        "batch_size_per_image": 128,
        "short_edge": (640, 800, 960),
        "max_size": 1600,
        "output": "frcnn_x101",
    },
    "retinanet_r101": {
        "zoo": "COCO-Detection/retinanet_R_101_FPN_3x.yaml",
        "base_lr": 0.00015,
        "max_iter": 7000,
        "steps": (4500, 6000),
        "warmup": 500,
        "ims_per_batch": 2,
        "short_edge": (640, 768, 896, 1024),
        "max_size": 1600,
        "output": "retinanet_r101",
    },
    "frcnn_r101": {
        "zoo": "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
        "base_lr": 0.00025,
        "max_iter": 5000,
        "steps": (3200, 4200),
        "warmup": 200,
        "ims_per_batch": 2,
        "batch_size_per_image": 128,
        "short_edge": (800, 960, 1024),
        "max_size": 1700,
        "output": "frcnn_r101",
    },
}


def ensure_datasets_registered():
    reg = {
        "engineering_train_v4": (TRAIN_JSON, DATA_DIR),
        "engineering_val_v4": (VAL_JSON, DATA_DIR),
    }
    for name, (ann, img_root) in reg.items():
        try:
            register_coco_instances(name, {}, str(ann), str(img_root))
        except AssertionError:
            pass


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        output_folder = output_folder or os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        augs = [
            T.ResizeShortestEdge(
                short_edge_length=cfg.INPUT.MIN_SIZE_TRAIN,
                max_size=cfg.INPUT.MAX_SIZE_TRAIN,
                sample_style="choice",
            ),
            T.RandomBrightness(0.9, 1.1),
            T.RandomContrast(0.9, 1.1),
        ]
        mapper = DatasetMapper(cfg, is_train=True, augmentations=augs)
        return build_detection_train_loader(cfg, mapper=mapper)


def build_cfg(model_key: str, gpu_id: int | None = None):
    spec = MODEL_SPECS[model_key]
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(spec["zoo"]))
    cfg.DATASETS.TRAIN = ("engineering_train_v4",)
    cfg.DATASETS.TEST = ("engineering_val_v4",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(spec["zoo"])
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if gpu_id is not None and cfg.MODEL.DEVICE == "cuda":
        cfg.MODEL.DEVICE = f"cuda:{gpu_id}"

    if "faster_rcnn" in spec["zoo"]:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASS_NAMES)
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = spec.get("batch_size_per_image", 128)
    elif "retinanet" in spec["zoo"]:
        cfg.MODEL.RETINANET.NUM_CLASSES = len(CLASS_NAMES)

    cfg.INPUT.MIN_SIZE_TRAIN = spec["short_edge"]
    cfg.INPUT.MAX_SIZE_TRAIN = spec["max_size"]
    cfg.INPUT.MIN_SIZE_TEST = max(spec["short_edge"])
    cfg.INPUT.MAX_SIZE_TEST = spec["max_size"]

    cfg.SOLVER.IMS_PER_BATCH = spec["ims_per_batch"]
    cfg.SOLVER.BASE_LR = spec["base_lr"]
    cfg.SOLVER.MAX_ITER = spec["max_iter"]
    cfg.SOLVER.STEPS = spec["steps"]
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.WARMUP_ITERS = spec["warmup"]
    cfg.SOLVER.CHECKPOINT_PERIOD = 500
    cfg.TEST.EVAL_PERIOD = 500

    out_dir = ROOT_OUT / spec["output"]
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUT_DIR = str(out_dir)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_SPECS.keys()), required=True)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    ensure_datasets_registered()
    cfg = build_cfg(args.model, args.gpu)

    print("=" * 72)
    print(f"Model key:   {args.model}")
    print(f"Zoo config:  {MODEL_SPECS[args.model]['zoo']}")
    print(f"Device:      {cfg.MODEL.DEVICE}")
    print(f"Train JSON:  {TRAIN_JSON}")
    print(f"Val JSON:    {VAL_JSON}")
    print(f"Image root:  {DATA_DIR}")
    print(f"Output dir:  {cfg.OUTPUT_DIR}")
    print(f"Max iter:    {cfg.SOLVER.MAX_ITER}")
    print(f"Base LR:     {cfg.SOLVER.BASE_LR}")
    print(f"Batch:       {cfg.SOLVER.IMS_PER_BATCH}")
    print("=" * 72)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
