---
title: Engineering Drawing Detection & OCR
emoji: 🔧
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: "4.44.0"
app_file: app_hf.py
pinned: false
---

# 🔧 Engineering Drawing Detection & OCR

Object detection + OCR pipeline for engineering drawings.

## Classes
- **PartDrawing** — technical drawing region
- **Note** — annotation / note region  
- **Table** — data table region

## Model
- Detector: Faster R-CNN X-101-32x8d FPN 3x (Detectron2)
- OCR: PaddleOCR + PPStructure
- Per-class threshold: PartDrawing ≥ 99%, Note/Table ≥ 90%
