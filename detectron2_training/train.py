"""
Detectron2 Training Script
Dataset: Engineering Drawings — PartDrawing / Note / Table
Model:   Faster R-CNN R-50-FPN 3x
"""
import os
import json
import logging
from pathlib import Path

from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetMapper, build_detection_train_loader
import detectron2.data.transforms as T

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR  = Path(r"D:\Object Detection & OCR System for Engineering Drawings")
DATA_DIR  = BASE_DIR / "labeled_export" / "backup_images"
TRAIN_DIR = BASE_DIR / "detectron2_training"
OUTPUT_DIR = TRAIN_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Register Datasets ───────────────────────────────────────────────────────
register_coco_instances(
    "engineering_train", {},
    str(TRAIN_DIR / "train.json"),
    str(DATA_DIR)
)
register_coco_instances(
    "engineering_val", {},
    str(TRAIN_DIR / "val.json"),
    str(DATA_DIR)
)

# ─── Trainer with Evaluator ──────────────────────────────────────────────────
class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        # Light augmentations suitable for engineering drawings
        augs = [
            T.ResizeShortestEdge(
                short_edge_length=(640, 800),
                max_size=1333,
                sample_style="choice"
            ),
            T.RandomFlip(horizontal=True, vertical=False),
            T.RandomBrightness(0.8, 1.2),
            T.RandomContrast(0.8, 1.2),
        ]
        mapper = DatasetMapper(cfg, is_train=True, augmentations=augs)
        return build_detection_train_loader(cfg, mapper=mapper)


# ─── Config ──────────────────────────────────────────────────────────────────
def build_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.DATASETS.TRAIN = ("engineering_train",)
    cfg.DATASETS.TEST  = ("engineering_val",)

    cfg.DATALOADER.NUM_WORKERS = 2

    # Pretrained weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    )

    # Model head
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # PartDrawing, Note, Table
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

    # Solver — small dataset, conservative LR
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 3000
    cfg.SOLVER.STEPS = (2000, 2500)   # LR decay points
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.WARMUP_ITERS = 200
    cfg.SOLVER.CHECKPOINT_PERIOD = 500

    # Eval every N iters
    cfg.TEST.EVAL_PERIOD = 500

    cfg.OUTPUT_DIR = str(OUTPUT_DIR)
    return cfg


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = build_cfg()

    print("=" * 60)
    print(f"Model:      Faster R-CNN R-50-FPN 3x")
    print(f"Classes:    PartDrawing, Note, Table")
    print(f"Train set:  engineering_train")
    print(f"Val set:    engineering_val")
    print(f"Max iters:  {cfg.SOLVER.MAX_ITER}")
    print(f"LR:         {cfg.SOLVER.BASE_LR}")
    print(f"Batch:      {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"Output dir: {cfg.OUTPUT_DIR}")
    print("=" * 60)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
