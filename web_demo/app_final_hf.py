"""
Engineering Drawing Detection & OCR - HuggingFace Spaces
Model: Faster R-CNN X-101-32x8d FPN 3x (Detectron2)
OCR: PaddleOCR + PPStructure
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import gradio as gr
from huggingface_hub import hf_hub_download

# ── Install detectron2 from source if not present ───────────────────────────
try:
    import detectron2
except ImportError:
    print("[SETUP] Installing detectron2...")
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "git+https://github.com/facebookresearch/detectron2.git",
        "--quiet"
    ], check=True)
    import detectron2

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor

from paddleocr import PaddleOCR, PPStructure

# ============================================================
# CONFIG
# ============================================================
MODEL_REPO   = "siucapditbu/engineering-drawing-frcnn-x101"
MODEL_FILE   = "model_final.pth"
WEIGHTS_DIR  = Path("weights")
WEIGHTS_DIR.mkdir(exist_ok=True)
MODEL_PATH   = WEIGHTS_DIR / MODEL_FILE

CLASS_NAMES  = ["PartDrawing", "Note", "Table"]
CLASS_COLORS = {
    "PartDrawing": (( 30, 100, 255), "#1e64ff"),
    "Note"       : (( 30, 200,  80), "#28c864"),
    "Table"      : ((  0,  30, 220), "#dc1e1e"),
}
CLASS_THRESH    = {"PartDrawing": 0.95, "Note": 0.50, "Table": 0.99}
SCORE_THRESH    = min(CLASS_THRESH.values())
TABLE_PAD_RATIO = 0.08

# ============================================================
# DOWNLOAD WEIGHTS
# ============================================================
def ensure_weights():
    if not MODEL_PATH.exists():
        print(f"[DOWNLOAD] {MODEL_REPO}/{MODEL_FILE} ...")
        hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            local_dir=str(WEIGHTS_DIR),
        )
        print("[DOWNLOAD] Done")

# ============================================================
# INIT MODELS
# ============================================================
print("[INIT] Loading models...")
ensure_weights()

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES       = len(CLASS_NAMES)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH
cfg.INPUT.MIN_SIZE_TEST               = 960
cfg.INPUT.MAX_SIZE_TEST               = 1600
cfg.MODEL.WEIGHTS                     = str(MODEL_PATH)
cfg.MODEL.DEVICE                      = "cuda" if torch.cuda.is_available() else "cpu"

_META = "hf_meta"
if _META not in MetadataCatalog:
    MetadataCatalog.get(_META).set(thing_classes=CLASS_NAMES)

PREDICTOR    = DefaultPredictor(cfg)
OCR_ENGINE   = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
TABLE_ENGINE = PPStructure(show_log=False, layout=False)
print(f"[INIT] Done — device={cfg.MODEL.DEVICE}")

# ============================================================
# HELPERS
# ============================================================
def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    th   = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 31, 8)
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

def add_padding(x1,y1,x2,y2,W,H,r=TABLE_PAD_RATIO):
    px=int((x2-x1)*r); py=int((y2-y1)*r)
    return max(0,x1-px),max(0,y1-py),min(W,x2+px),min(H,y2+py)

def paddle_lines(img):
    res=OCR_ENGINE.ocr(img,cls=True); out=[]
    if not res or res[0] is None: return out
    for item in res[0]:
        if len(item)!=2: continue
        box,(text,score)=item
        if not str(text).strip(): continue
        xs=[p[0] for p in box]; ys=[p[1] for p in box]
        out.append({"text":str(text).strip(),"score":float(score),
                    "x1":min(xs),"y1":min(ys),"x2":max(xs),"y2":max(ys),
                    "cx":(min(xs)+max(xs))/2,"cy":(min(ys)+max(ys))/2,"h":max(ys)-min(ys)})
    return out

def group_rows(items):
    if not items: return []
    items=sorted(items,key=lambda x:(x["cy"],x["x1"]))
    tol=max(12.0,np.median([max(8.0,i["h"]) for i in items])*0.8)
    rows,cur=[],[items[0]]
    for it in items[1:]:
        if abs(it["cy"]-cur[-1]["cy"])<=tol: cur.append(it)
        else: rows.append(sorted(cur,key=lambda z:z["x1"])); cur=[it]
    rows.append(sorted(cur,key=lambda z:z["x1"]))
    return rows

def note_to_text(items):
    return "\n".join(" ".join(i["text"] for i in r) for r in group_rows(items)).strip()

def ppstructure_cells(img):
    res=TABLE_ENGINE(img)
    if not res or res[0].get("type")!="table": return []
    cells=[]
    for idx,poly in enumerate(res[0].get("res",{}).get("cell_bbox",[])):
        xs=[poly[0],poly[2],poly[4],poly[6]]; ys=[poly[1],poly[3],poly[5],poly[7]]
        cells.append({"id":idx,"x1":min(xs),"y1":min(ys),"x2":max(xs),"y2":max(ys),
                      "cx":(min(xs)+max(xs))/2,"cy":(min(ys)+max(ys))/2,"text":""})
    return cells

def assign_cells(cells,items):
    for it in items:
        best,dist=None,1e18
        for i,c in enumerate(cells):
            if c["x1"]-4<=it["cx"]<=c["x2"]+4 and c["y1"]-4<=it["cy"]<=c["y2"]+4:
                d=(it["cx"]-c["cx"])**2+(it["cy"]-c["cy"])**2
                if d<dist: dist=d; best=i
        if best is not None:
            cells[best]["text"]=(cells[best]["text"]+" "+it["text"]).strip()
    return cells

def cluster(vals,tol):
    if not vals: return []
    vals=sorted(vals); g=[[vals[0]]]
    for v in vals[1:]:
        if abs(v-np.mean(g[-1]))<=tol: g[-1].append(v)
        else: g.append([v])
    return [float(np.mean(x)) for x in g]

def cells_to_md(cells):
    if not cells: return ""
    rk=cluster([c["cy"] for c in cells],18); ck=cluster([c["cx"] for c in cells],30)
    mat=[[""]*len(ck) for _ in rk]
    for c in cells:
        r=min(range(len(rk)),key=lambda i:abs(c["cy"]-rk[i]))
        col=min(range(len(ck)),key=lambda i:abs(c["cx"]-ck[i]))
        mat[r][col]=(mat[r][col]+" "+c["text"]).strip() if mat[r][col] else c["text"]
    nc=max(len(r) for r in mat); mat=[r+[""]*(nc-len(r)) for r in mat]
    rows=["|"+" | ".join(c or " " for c in mat[0])+" |",
          "|"+" | ".join(["---"]*nc)+" |"]
    for row in mat[1:]: rows.append("|"+" | ".join(c or " " for c in row)+" |")
    return "\n".join(rows)

def run_table_ocr(crop):
    proc=preprocess_for_ocr(crop)
    c1=ppstructure_cells(crop); c2=ppstructure_cells(proc)
    img=proc if len(c2)>len(c1) else crop; cells=c2 if len(c2)>len(c1) else c1
    if not cells: return note_to_text(paddle_lines(img))
    return cells_to_md(assign_cells(cells,paddle_lines(img)))

def draw_boxes(img_bgr, objects):
    vis=img_bgr.copy()
    for obj in objects:
        b=obj["bbox"]; x1,y1,x2,y2=b["x1"],b["y1"],b["x2"],b["y2"]
        color=CLASS_COLORS[obj["class"]][0]
        cv2.rectangle(vis,(x1,y1),(x2,y2),color,2)
        label=f"{obj['class']} {obj['confidence']:.2f}"
        (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.55,1)
        cv2.rectangle(vis,(x1,max(0,y1-th-6)),(x1+tw+4,y1),color,-1)
        cv2.putText(vis,label,(x1+2,y1-4),cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,(255,255,255),1,cv2.LINE_AA)
    return vis

# ============================================================
# INFERENCE
# ============================================================
def run_pipeline(img_np):
    if img_np is None:
        return None, "{}", "_No image uploaded_"
    img_bgr=cv2.cvtColor(img_np,cv2.COLOR_RGB2BGR); H,W=img_bgr.shape[:2]

    outputs=PREDICTOR(img_bgr); inst=outputs["instances"].to("cpu")
    scores_a =inst.scores.numpy()            if inst.has("scores")       else []
    classes_a=inst.pred_classes.numpy()      if inst.has("pred_classes") else []
    boxes_a  =inst.pred_boxes.tensor.numpy() if inst.has("pred_boxes")   else []

    keep=[j for j,(c,s) in enumerate(zip(classes_a,scores_a))
          if float(s)>=CLASS_THRESH[CLASS_NAMES[int(c)]]]
    filtered=inst[torch.tensor(keep,dtype=torch.long)] if keep else inst[:0]
    scores_f =filtered.scores.numpy()            if filtered.has("scores")       else []
    classes_f=filtered.pred_classes.numpy()      if filtered.has("pred_classes") else []
    boxes_f  =filtered.pred_boxes.tensor.numpy() if filtered.has("pred_boxes")   else []

    objects=[]; ocr_parts=[]
    for i,(box,score,cls_id) in enumerate(zip(boxes_f,scores_f,classes_f),start=1):
        cls_name=CLASS_NAMES[int(cls_id)]
        x1,y1,x2,y2=[int(round(v)) for v in box]
        if cls_name=="Table": x1,y1,x2,y2=add_padding(x1,y1,x2,y2,W,H)
        x1=max(0,x1);y1=max(0,y1);x2=min(W,x2);y2=min(H,y2)
        crop=img_bgr[y1:y2,x1:x2]
        if crop.size==0: continue

        ocr_content=""
        if cls_name=="Note":
            proc=preprocess_for_ocr(crop)
            i1=paddle_lines(crop); i2=paddle_lines(proc)
            ocr_content=note_to_text(i2 if len(i2)>=len(i1) else i1)
            if ocr_content: ocr_parts.append(f"### 📝 Note #{i}\n\n{ocr_content}\n")
        elif cls_name=="Table":
            ocr_content=run_table_ocr(crop)
            if ocr_content: ocr_parts.append(f"### 📊 Table #{i}\n\n{ocr_content}\n")

        objects.append({"id":i,"class":cls_name,"confidence":round(float(score),4),
                        "bbox":{"x1":x1,"y1":y1,"x2":x2,"y2":y2},
                        "ocr_content":ocr_content})

    vis_rgb=cv2.cvtColor(draw_boxes(img_bgr,objects),cv2.COLOR_BGR2RGB)
    json_str=json.dumps({"image":"uploaded_image","objects":objects},
                        indent=2,ensure_ascii=False)
    ocr_text="\n".join(ocr_parts) if ocr_parts else "_No Note or Table detected_"
    return vis_rgb, json_str, ocr_text

# ============================================================
# GRADIO UI
# ============================================================
CSS="#title{text-align:center}#sub{text-align:center;color:#666;margin-top:-10px}"

with gr.Blocks(css=CSS, title="Engineering Drawing Detection & OCR") as demo:
    gr.Markdown("# 🔧 Engineering Drawing — Detection & OCR", elem_id="title")
    gr.Markdown(
        "Upload a drawing → detect **PartDrawing / Note / Table** "
        "(**Faster R-CNN X-101**) → OCR (**PaddleOCR + PPStructure**)",
        elem_id="sub"
    )
    gr.HTML("""
    <div style='text-align:center;margin:6px 0 14px'>
      <span style='background:#1e64ff;display:inline-block;width:13px;height:13px;
        border-radius:3px;margin-right:5px;vertical-align:middle'></span><b>PartDrawing</b>&nbsp;&nbsp;
      <span style='background:#28c864;display:inline-block;width:13px;height:13px;
        border-radius:3px;margin-right:5px;vertical-align:middle'></span><b>Note</b>&nbsp;&nbsp;
      <span style='background:#dc1e1e;display:inline-block;width:13px;height:13px;
        border-radius:3px;margin-right:5px;vertical-align:middle'></span><b>Table</b>
    </div>""")

    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type="numpy", label="📂 Upload Engineering Drawing")
            btn = gr.Button("🚀 Detect & OCR", variant="primary", size="lg")
        with gr.Column(scale=2):
            out_vis = gr.Image(type="numpy", label="🎯 Detection Result")

    with gr.Row():
        out_json = gr.Code(language="json", label="📋 JSON Output", lines=20)
        out_ocr  = gr.Markdown(label="📄 OCR Content (Note & Table)")

    btn.click(run_pipeline, inputs=[inp], outputs=[out_vis, out_json, out_ocr])
    inp.upload(run_pipeline, inputs=[inp], outputs=[out_vis, out_json, out_ocr])

    gr.Markdown(
        "---\n**Model:** Faster R-CNN X-101-32x8d FPN 3x &nbsp;|&nbsp; "
        "**OCR:** PaddleOCR + PPStructure &nbsp;|&nbsp; "
        "**Thresholds:** PartDrawing ≥ 95%, Note ≥ 50%, Table ≥ 99%"
    )

if __name__ == "__main__":
    demo.launch()
