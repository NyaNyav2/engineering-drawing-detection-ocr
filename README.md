# 🔧 Object Detection & OCR System for Engineering Drawings
## 📞 Liên hệ

- **SĐT:** 0368820203
- **Email:** vansykien03@gmail.com
**Sotatek Computer Vision Assessment**

Hệ thống tự động phát hiện và OCR 3 loại đối tượng trong bản vẽ kỹ thuật:
- **PartDrawing** — vùng bản vẽ chi tiết kỹ thuật
- **Note** — vùng ghi chú / chú thích
- **Table** — vùng bảng dữ liệu kỹ thuật

🌐 **Web Demo:** https://senators-richmond-montana-hist.trycloudflare.com

📦 **Model Weights:** https://huggingface.co/siucapditbu/engineering-drawing-frcnn-x101

---

## 📐 Approach & Methodology

### Detection
- **Model:** Faster R-CNN X-101-32x8d FPN 3x (Detectron2)
- **Backbone:** ResNeXt-101 32x8d — mạnh hơn ResNet-50/101 với bản vẽ kỹ thuật (chi tiết nhỏ, layout phức tạp)
- **Per-class confidence threshold:**
  - PartDrawing ≥ 95%
  - Note ≥ 50%
  - Table ≥ 99%
- **Data augmentation:** RandomBrightness, RandomContrast, multi-scale resize (640–960px)

### OCR
- **Note regions:** PaddleOCR (angle classification + text grouping theo hàng)
- **Table regions:** PPStructure (cell detection) → PaddleOCR (text per cell) → Markdown table
- **Preprocessing:** 2x upscale + bilateral filter + adaptive threshold để tăng độ rõ

### Experiments
| Model | mAP@50 (val) | Notes |
|-------|-------------|-------|
| Faster R-CNN R-50 FPN 3x | ~0.72 | Baseline |
| Faster R-CNN R-101 FPN 3x | ~0.79 | Better |
| **Faster R-CNN X-101 FPN 3x** | **~0.85** | Best — dùng cho production |
| RetinaNet R-101 FPN 3x | ~0.74 | Worse on small objects |

### Hướng cải thiện
- Thử **DINO / DINOv2** (ViT backbone) làm feature extractor thay cho ResNeXt — ViT học được global context tốt hơn CNN trên layout phức tạp của bản vẽ kỹ thuật
- **ViTDet** (plain ViT backbone + Detectron2) — loại bỏ FPN, dùng window attention để xử lý ảnh độ phân giải cao mà không tốn quá nhiều VRAM
- Fine-tune **Grounding DINO** (ViT + text-conditioned detection) để detect open-vocabulary object theo tên class tùy biến — phù hợp khi mở rộng sang loại bản vẽ mới
- Thử **RT-DETR-L/X** (ViT encoder hybrid) — nhanh hơn Faster R-CNN X-101 ở inference trong khi vẫn giữ được accuracy cao
- Ensemble **X-101 + ViTDet + Grounding DINO** với WBF để tăng recall trên các vùng Note nhỏ và bị che khuất
- Tích hợp SAM của Meta để tự động trích xuất segmentation mask chi tiết cho các vùng object, đặc biệt hữu ích khi muốn phân tích chính xác biên dạng các chi tiết cơ khí phức tạp trong PartDrawing thay vì chỉ khoanh vùng bằng bounding box.

---

## 🛠️ Installation

### Requirements
- Python 3.8+
- CUDA 11.8+ (recommended) hoặc CPU
- Conda (recommended)

### Setup môi trường

```bash
# 1. Tạo conda environment
conda create -n cv-engineering python=3.8 -y
conda activate cv-engineering

# 2. Cài PyTorch (CUDA 12.1)
pip install torch==2.0.1+cu121 torchvision==0.15.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# 3. Cài Detectron2 từ source
pip install git+https://github.com/facebookresearch/detectron2.git

# 4. Cài PaddleOCR
pip install paddlepaddle-gpu==2.5.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install paddleocr>=2.7.0

# 5. Cài các deps còn lại
pip install opencv-python numpy Pillow gradio huggingface_hub
```

### Download model weights

```bash
# Option 1: HuggingFace Hub (recommended)
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='siucapditbu/engineering-drawing-frcnn-x101',
    filename='model_final.pth',
    local_dir='detectron2_training/output_v4_ensemble/frcnn_x101/'
)
"

# Option 2: Manual download
# https://huggingface.co/siucapditbu/engineering-drawing-frcnn-x101/blob/main/model_final.pth
# Đặt vào: detectron2_training/output_v4_ensemble/frcnn_x101/model_final.pth
```

---

## 🚀 Running Inference

### Pipeline hoàn chỉnh (detection + OCR)

```bash
cd detectron2_training
python pipeline_final.py
```

Output sẽ lưu tại:
```
detectron2_training/output_v4_ensemble/frcnn_x101/ocr_pipeline_final/
├── crops/              ← ảnh crop từng object
├── json/               ← JSON output từng ảnh
├── markdown/           ← OCR text dạng .md
├── visualizations/     ← ảnh detection visualized
├── preprocessed/       ← ảnh preprocess OCR
├── table_structure_vis/ ← cell structure visualization
└── summary.json        ← tổng hợp toàn bộ
```

### Chỉ test detection (không OCR)

```bash
cd detectron2_training
python test_frcnn_x101_custom.py
```

### Web Demo local

```bash
cd web_demo
python app.py
# Mở http://localhost:7860
```

---

## 🏋️ Training

```bash
cd detectron2_training/v4_ensemble

# Train frcnn_x101
python train_frcnn_x101.py

# Hoặc chọn model cụ thể
python train_ensemble.py --model frcnn_x101
python train_ensemble.py --model frcnn_r101
python train_ensemble.py --model retinanet_r101

# Resume training
python train_ensemble.py --model frcnn_x101 --resume
```

**Cấu hình training frcnn_x101:**
- Max iterations: 6000
- Base LR: 0.0002 (warmup 300 iters)
- LR decay: 0.1 tại iter 4000, 5200
- Batch: 2 images/iter
- Multi-scale: 640–960px short edge, max 1600px
- Augmentation: RandomBrightness(0.9–1.1), RandomContrast(0.9–1.1)
- Eval period: 500 iters

---

## 📁 Project Structure

```
├── detectron2_training/
│   ├── v4_ensemble/
│   │   ├── train_ensemble.py       ← main training script
│   │   ├── train_frcnn_x101.py     ← shortcut script
│   │   ├── train_frcnn_r101.py
│   │   ├── train_retinanet_r101.py
│   │   └── ensemble_infer.py       ← ensemble inference (WBF)
│   ├── pipeline_final.py           ← ★ main pipeline (detect + OCR)
│   ├── test_frcnn_x101_custom.py   ← detection-only test
│   ├── train.json                  ← COCO format train annotations
│   └── val.json                    ← COCO format val annotations
├── web_demo/
│   ├── app.py                      ← Gradio local demo
│   ├── app_final_hf.py             ← HuggingFace Spaces version
│   └── requirements.txt
└── README.md
```

---

## 📊 JSON Output Format

```json
{
  "image": "drawing_001.jpg",
  "objects": [
    {
      "id": 1,
      "class": "Table",
      "confidence": 0.9987,
      "bbox": { "x1": 120, "y1": 340, "x2": 680, "y2": 520 },
      "ocr_content": "| Col1 | Col2 |\n|------|------|\n| val1 | val2 |"
    },
    {
      "id": 2,
      "class": "Note",
      "confidence": 0.9231,
      "bbox": { "x1": 50, "y1": 600, "x2": 400, "y2": 650 },
      "ocr_content": "GENERAL TOLERANCES AS PER ISO 2768"
    },
    {
      "id": 3,
      "class": "PartDrawing",
      "confidence": 0.9999,
      "bbox": { "x1": 10, "y1": 10, "x2": 500, "y2": 450 },
      "ocr_content": ""
    }
  ]
}
```
