"""
Web Demo - Engineering Drawing Detection & OCR
Gradio app using frcnn_x101 + PaddleOCR + PPStructure
"""
from __future__ import annotations
import json
import sys
import tempfile
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import gradio as gr

# ── detectron2 / paddle imports (lazy to keep startup fast) ──────────────────
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from paddleocr import PaddleOCR, PPStructure

# ============================================================
# PATHS  (edit MODEL_PATH if needed)
# ============================================================
BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_PATH = (BASE_DIR / "detectron2_training" / "output_v4_ensemble"
              / "frcnn_x101" / "model_final.pth")

CLASS_NAMES  = ["PartDrawing", "Note", "Table"]
CLASS_COLORS = {          # BGR for cv2, RGB string for legend
    "PartDrawing": ((255, 100,  30), "#1e64ff"),
    "Note"       : (( 30, 200, 100), "#28c864"),
    "Table"      : (( 30,  30, 220), "#dc1e1e"),
}
CLASS_THRESH = {"PartDrawing": 0.95, "Note": 0.50, "Table": 0.99}
SCORE_THRESH  = min(CLASS_THRESH.values())
TABLE_PAD_RATIO = 0.08

# ============================================================
# MODEL INIT (global, load once)
# ============================================================
print("[INIT] Loading models…")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES       = len(CLASS_NAMES)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH
cfg.INPUT.MIN_SIZE_TEST               = 960
cfg.INPUT.MAX_SIZE_TEST               = 1600
cfg.MODEL.WEIGHTS                     = str(MODEL_PATH)
cfg.MODEL.DEVICE                      = "cuda" if torch.cuda.is_available() else "cpu"

_META = "demo_meta"
if _META not in MetadataCatalog:
    MetadataCatalog.get(_META).set(thing_classes=CLASS_NAMES)
METADATA  = MetadataCatalog.get(_META)
PREDICTOR = DefaultPredictor(cfg)

OCR_ENGINE   = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
TABLE_ENGINE = PPStructure(show_log=False, layout=False)

print(f"[INIT] Done — device={cfg.MODEL.DEVICE}")

# ============================================================
# HELPER FUNCTIONS  (same logic as pipeline_final.py)
# ============================================================
def preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    th   = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 31, 8)
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)


def add_padding(x1, y1, x2, y2, W, H, r=TABLE_PAD_RATIO):
    px = int((x2-x1)*r); py = int((y2-y1)*r)
    return max(0,x1-px), max(0,y1-py), min(W,x2+px), min(H,y2+py)


def paddle_lines(img: np.ndarray):
    res = OCR_ENGINE.ocr(img, cls=True)
    out = []
    if not res or res[0] is None:
        return out
    for item in res[0]:
        if len(item) != 2:
            continue
        box, (text, score) = item
        if not str(text).strip():
            continue
        xs=[p[0] for p in box]; ys=[p[1] for p in box]
        out.append({"text": str(text).strip(), "score": float(score),
                    "x1":min(xs),"y1":min(ys),"x2":max(xs),"y2":max(ys),
                    "cx":(min(xs)+max(xs))/2,"cy":(min(ys)+max(ys))/2,
                    "h":max(ys)-min(ys)})
    return out


def group_rows(items):
    if not items: return []
    items = sorted(items, key=lambda x: (x["cy"], x["x1"]))
    tol = max(12.0, np.median([max(8.0,i["h"]) for i in items])*0.8)
    rows, cur = [], [items[0]]
    for it in items[1:]:
        if abs(it["cy"]-cur[-1]["cy"]) <= tol: cur.append(it)
        else: rows.append(sorted(cur,key=lambda z:z["x1"])); cur=[it]
    rows.append(sorted(cur, key=lambda z: z["x1"]))
    return rows


def note_to_text(items) -> str:
    lines = [" ".join(i["text"] for i in row) for row in group_rows(items)]
    return "\n".join(l for l in lines if l.strip()).strip()


def ppstructure_cells(img: np.ndarray):
    res = TABLE_ENGINE(img)
    if not res or res[0].get("type") != "table": return []
    cells = []
    for idx, poly in enumerate(res[0].get("res",{}).get("cell_bbox",[])):
        xs=[poly[0],poly[2],poly[4],poly[6]]; ys=[poly[1],poly[3],poly[5],poly[7]]
        cells.append({"id":idx,"x1":min(xs),"y1":min(ys),"x2":max(xs),"y2":max(ys),
                      "cx":(min(xs)+max(xs))/2,"cy":(min(ys)+max(ys))/2,"text":""})
    return cells


def assign_cells(cells, items):
    for it in items:
        best, dist = None, 1e18
        for i,c in enumerate(cells):
            if c["x1"]-4<=it["cx"]<=c["x2"]+4 and c["y1"]-4<=it["cy"]<=c["y2"]+4:
                d=(it["cx"]-c["cx"])**2+(it["cy"]-c["cy"])**2
                if d<dist: dist=d; best=i
        if best is not None:
            cells[best]["text"]=(cells[best]["text"]+" "+it["text"]).strip()
    return cells


def cluster(vals, tol):
    if not vals: return []
    vals=sorted(vals); g=[[vals[0]]]
    for v in vals[1:]:
        if abs(v-np.mean(g[-1]))<=tol: g[-1].append(v)
        else: g.append([v])
    return [float(np.mean(x)) for x in g]


def cells_to_md(cells) -> str:
    if not cells: return ""
    rk=cluster([c["cy"] for c in cells],18)
    ck=cluster([c["cx"] for c in cells],30)
    mat=[[""]*len(ck) for _ in rk]
    for c in cells:
        r=min(range(len(rk)),key=lambda i:abs(c["cy"]-rk[i]))
        col=min(range(len(ck)),key=lambda i:abs(c["cx"]-ck[i]))
        mat[r][col]=(mat[r][col]+" "+c["text"]).strip() if mat[r][col] else c["text"]
    nc=max(len(r) for r in mat)
    mat=[r+[""]*(nc-len(r)) for r in mat]
    hdr=mat[0]; body=mat[1:]
    sep=["---"]*nc
    rows=["|"+" | ".join(c or " " for c in hdr)+" |",
          "|"+" | ".join(sep)+" |"]
    for row in body: rows.append("|"+" | ".join(c or " " for c in row)+" |")
    return "\n".join(rows)


def run_table_ocr(crop: np.ndarray) -> str:
    proc = preprocess_for_ocr(crop)
    c1 = ppstructure_cells(crop); c2 = ppstructure_cells(proc)
    img = proc if len(c2)>len(c1) else crop
    cells = c2 if len(c2)>len(c1) else c1
    if not cells:
        return note_to_text(paddle_lines(img))
    cells = assign_cells(cells, paddle_lines(img))
    return cells_to_md(cells)


# ============================================================
# DRAW BOUNDING BOXES (custom, color-per-class)
# ============================================================
def draw_boxes(img_bgr: np.ndarray, objects: list) -> np.ndarray:
    vis = img_bgr.copy()
    for obj in objects:
        b = obj["bbox"]
        x1,y1,x2,y2 = b["x1"],b["y1"],b["x2"],b["y2"]
        color_bgr = CLASS_COLORS[obj["class"]][0]
        cv2.rectangle(vis,(x1,y1),(x2,y2),color_bgr,2)
        label = f"{obj['class']} {obj['confidence']:.2f}"
        (tw,th),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.55,1)
        cv2.rectangle(vis,(x1,max(0,y1-th-6)),(x1+tw+4,y1),color_bgr,-1)
        cv2.putText(vis,label,(x1+2,y1-4),cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,(255,255,255),1,cv2.LINE_AA)
    return vis


# ============================================================
# MAIN INFERENCE FUNCTION
# ============================================================
def run_pipeline(img_np: np.ndarray) -> Tuple[np.ndarray, str, str]:
    """
    Returns:
        vis_rgb    : annotated image (RGB)
        json_str   : pretty JSON
        ocr_md     : OCR markdown text
    """
    if img_np is None:
        return None, "", ""

    # Gradio passes RGB; convert to BGR for OpenCV/detectron2
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    H, W    = img_bgr.shape[:2]

    # --- Detection ---
    outputs   = PREDICTOR(img_bgr)
    instances = outputs["instances"].to("cpu")
    scores_a  = instances.scores.numpy()            if instances.has("scores")       else []
    classes_a = instances.pred_classes.numpy()      if instances.has("pred_classes") else []
    boxes_a   = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes")   else []

    keep = [j for j,(c,s) in enumerate(zip(classes_a,scores_a))
            if float(s) >= CLASS_THRESH[CLASS_NAMES[int(c)]]]
    filtered = instances[torch.tensor(keep,dtype=torch.long)] if keep else instances[:0]

    scores_f  = filtered.scores.numpy()            if filtered.has("scores")       else []
    classes_f = filtered.pred_classes.numpy()      if filtered.has("pred_classes") else []
    boxes_f   = filtered.pred_boxes.tensor.numpy() if filtered.has("pred_boxes")   else []

    objects   = []
    ocr_parts = []

    for i,(box,score,cls_id) in enumerate(zip(boxes_f,scores_f,classes_f),start=1):
        cls_name = CLASS_NAMES[int(cls_id)]
        x1,y1,x2,y2 = [int(round(v)) for v in box]
        if cls_name == "Table":
            x1,y1,x2,y2 = add_padding(x1,y1,x2,y2,W,H)
        x1=max(0,x1); y1=max(0,y1); x2=min(W,x2); y2=min(H,y2)
        crop = img_bgr[y1:y2,x1:x2]
        if crop.size == 0: continue

        ocr_content = ""
        if cls_name == "Note":
            proc = preprocess_for_ocr(crop)
            i1   = paddle_lines(crop); i2 = paddle_lines(proc)
            items = i2 if len(i2)>=len(i1) else i1
            ocr_content = note_to_text(items)
            if ocr_content:
                ocr_parts.append(f"### 📝 Note #{i}\n\n{ocr_content}\n")

        elif cls_name == "Table":
            ocr_content = run_table_ocr(crop)
            if ocr_content:
                ocr_parts.append(f"### 📊 Table #{i}\n\n{ocr_content}\n")

        objects.append({
            "id"         : i,
            "class"      : cls_name,
            "confidence" : round(float(score),4),
            "bbox"       : {"x1":x1,"y1":y1,"x2":x2,"y2":y2},
            "ocr_content": ocr_content,
        })

    # --- Visualization ---
    vis_bgr = draw_boxes(img_bgr, objects)
    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)

    # --- JSON ---
    result  = {"image": "uploaded_image", "objects": objects}
    json_str = json.dumps(result, indent=2, ensure_ascii=False)

    # --- OCR panel ---
    ocr_text = "\n".join(ocr_parts) if ocr_parts else "_No Note or Table detected_"

    return vis_rgb, json_str, ocr_text


# ============================================================
# GRADIO UI
# ============================================================
CSS = """
#title { text-align: center; }
#subtitle { text-align: center; color: #666; margin-top: -10px; }
.legend span { display:inline-block; width:14px; height:14px;
               border-radius:3px; margin-right:6px; vertical-align:middle; }
"""

EXAMPLES_DIR = Path(__file__).parent / "examples"

with gr.Blocks(css=CSS, title="Engineering Drawing Detection & OCR") as demo:

    gr.Markdown("# 🔧 Engineering Drawing Detection & OCR", elem_id="title")
    gr.Markdown(
        "Upload an engineering drawing → detect **PartDrawing / Note / Table** "
        "with **Faster R-CNN X-101** → OCR with **PaddleOCR + PPStructure**",
        elem_id="subtitle"
    )

    # Legend
    gr.HTML("""
    <div class='legend' style='text-align:center;margin:8px 0 16px'>
      <span style='background:#1e64ff'></span><b>PartDrawing</b>&nbsp;&nbsp;
      <span style='background:#28c864'></span><b>Note</b>&nbsp;&nbsp;
      <span style='background:#dc1e1e'></span><b>Table</b>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            inp_img = gr.Image(type="numpy", label="📂 Upload Engineering Drawing")
            run_btn = gr.Button("🚀 Detect & OCR", variant="primary", size="lg")

        with gr.Column(scale=2):
            out_vis = gr.Image(type="numpy", label="🎯 Detection Result")

    with gr.Row():
        with gr.Column():
            out_json = gr.Code(language="json", label="📋 JSON Output",
                               lines=20)
        with gr.Column():
            out_ocr = gr.Markdown(label="📄 OCR Content (Note & Table)")

    run_btn.click(
        fn=run_pipeline,
        inputs=[inp_img],
        outputs=[out_vis, out_json, out_ocr],
    )

    # Also trigger on image upload
    inp_img.upload(
        fn=run_pipeline,
        inputs=[inp_img],
        outputs=[out_vis, out_json, out_ocr],
    )

    gr.Markdown(
        "---\n"
        "**Model:** Faster R-CNN X-101-32x8d FPN 3x (Detectron2)  |  "
        "**OCR:** PaddleOCR + PPStructure  |  "
        "**Thresholds:** PartDrawing ≥ 95%, Note ≥ 50%, Table ≥ 99%"
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
