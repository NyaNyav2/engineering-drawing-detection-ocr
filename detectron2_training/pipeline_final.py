"""
Pipeline hoàn chỉnh - Final Version
- Detection : Faster R-CNN X-101-32x8d FPN 3x (frcnn_x101)
- Per-class threshold : PartDrawing >= 99%, Note >= 90%, Table >= 90%
- OCR Note  : PaddleOCR (text lines)
- OCR Table : PPStructure (cell structure) + PaddleOCR -> Markdown
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np
import torch
from paddleocr import PaddleOCR, PPStructure

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

# ============================================================
# CONFIG
# ============================================================
BASE_DIR   = Path(r"D:\Object Detection & OCR System for Engineering Drawings")
INPUT_DIR  = BASE_DIR / "blueprint_yolo" / "images" / "custom_test"
MODEL_PATH = BASE_DIR / "detectron2_training" / "output_v4_ensemble" / "frcnn_x101" / "model_final.pth"
OUT_DIR    = BASE_DIR / "detectron2_training" / "output_v4_ensemble" / "frcnn_x101" / "ocr_pipeline_final"

CLASS_NAMES = ["PartDrawing", "Note", "Table"]

# Per-class confidence threshold
CLASS_THRESH = {
    "PartDrawing": 0.95,
    "Note"       : 0.50,
    "Table"      : 0.99,
}
SCORE_THRESH    = min(CLASS_THRESH.values())   # predictor cutoff = 0.50
MIN_SIZE_TEST   = 960
MAX_SIZE_TEST   = 1600
TABLE_PAD_RATIO = 0.08

# Output sub-folders
CROPS_DIR = OUT_DIR / "crops"
JSON_DIR  = OUT_DIR / "json"
MD_DIR    = OUT_DIR / "markdown"
VIS_DIR   = OUT_DIR / "visualizations"
PRE_DIR   = OUT_DIR / "preprocessed"
TABLE_VIS = OUT_DIR / "table_structure_vis"
for _p in [OUT_DIR, CROPS_DIR, JSON_DIR, MD_DIR, VIS_DIR, PRE_DIR, TABLE_VIS]:
    _p.mkdir(parents=True, exist_ok=True)

# ============================================================
# BUILD MODELS
# ============================================================
def build_predictor():
    assert MODEL_PATH.exists(), f"[ERROR] model not found: {MODEL_PATH}"
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES       = len(CLASS_NAMES)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH
    cfg.INPUT.MIN_SIZE_TEST               = MIN_SIZE_TEST
    cfg.INPUT.MAX_SIZE_TEST               = MAX_SIZE_TEST
    cfg.MODEL.WEIGHTS                     = str(MODEL_PATH)
    cfg.MODEL.DEVICE                      = "cuda" if torch.cuda.is_available() else "cpu"
    meta_name = "pipeline_final_meta"
    if meta_name not in MetadataCatalog:
        MetadataCatalog.get(meta_name).set(thing_classes=CLASS_NAMES)
    metadata  = MetadataCatalog.get(meta_name)
    predictor = DefaultPredictor(cfg)
    return predictor, metadata, cfg


def build_ocr_engine():
    return PaddleOCR(use_angle_cls=True, lang='en', show_log=False)


def build_table_engine():
    return PPStructure(show_log=False, layout=False)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================
def preprocess_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    th   = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 31, 8)
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)


def add_padding(x1, y1, x2, y2, w, h, ratio=TABLE_PAD_RATIO):
    pad_x = int((x2 - x1) * ratio)
    pad_y = int((y2 - y1) * ratio)
    return max(0, x1 - pad_x), max(0, y1 - pad_y), min(w, x2 + pad_x), min(h, y2 + pad_y)


# ============================================================
# OCR HELPERS
# ============================================================
def clean_text(txt: str) -> str:
    lines = [ln.rstrip() for ln in txt.replace('\r', '').split('\n')]
    return '\n'.join(ln for ln in lines if ln.strip()).strip()


def paddle_lines(ocr_engine, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
    result = ocr_engine.ocr(img_bgr, cls=True)
    out = []
    if not result or result[0] is None:
        return out
    for item in result[0]:
        if len(item) != 2:
            continue
        box, (text, score) = item
        if not str(text).strip():
            continue
        xs = [pt[0] for pt in box]; ys = [pt[1] for pt in box]
        out.append({
            'text' : str(text).strip(),
            'score': float(score),
            'x1': float(min(xs)), 'y1': float(min(ys)),
            'x2': float(max(xs)), 'y2': float(max(ys)),
            'cx': float((min(xs) + max(xs)) / 2),
            'cy': float((min(ys) + max(ys)) / 2),
            'h' : float(max(ys) - min(ys)),
        })
    return out


def group_text_rows(items: List[Dict]) -> List[List[Dict]]:
    if not items:
        return []
    items = sorted(items, key=lambda x: (x['cy'], x['x1']))
    med_h = np.median([max(8.0, it['h']) for it in items])
    tol   = max(12.0, med_h * 0.8)
    rows, cur = [], [items[0]]
    for item in items[1:]:
        if abs(item['cy'] - cur[-1]['cy']) <= tol:
            cur.append(item)
        else:
            rows.append(sorted(cur, key=lambda z: z['x1']))
            cur = [item]
    rows.append(sorted(cur, key=lambda z: z['x1']))
    return rows


def note_to_text(items: List[Dict]) -> str:
    rows = group_text_rows(items)
    return clean_text('\n'.join(' '.join(it['text'] for it in row) for row in rows))


# ============================================================
# TABLE HELPERS
# ============================================================
def ppstructure_cells(table_engine, img_bgr: np.ndarray):
    res = table_engine(img_bgr)
    if not res:
        return []
    item = res[0]
    if item.get('type') != 'table':
        return []
    cells = []
    for idx, poly in enumerate(item.get('res', {}).get('cell_bbox', [])):
        xs = [poly[0], poly[2], poly[4], poly[6]]
        ys = [poly[1], poly[3], poly[5], poly[7]]
        cells.append({
            'id': idx, 'poly': poly,
            'x1': float(min(xs)), 'y1': float(min(ys)),
            'x2': float(max(xs)), 'y2': float(max(ys)),
            'cx': float((min(xs)+max(xs))/2),
            'cy': float((min(ys)+max(ys))/2),
            'text': '',
        })
    return cells


def assign_text_to_cells(cells, ocr_items):
    for it in ocr_items:
        best_idx, best_dist = None, 1e18
        for idx, cell in enumerate(cells):
            if (cell['x1']-4 <= it['cx'] <= cell['x2']+4 and
                    cell['y1']-4 <= it['cy'] <= cell['y2']+4):
                dx = it['cx'] - cell['cx']; dy = it['cy'] - cell['cy']
                dist = dx*dx + dy*dy
                if dist < best_dist:
                    best_dist = dist; best_idx = idx
        if best_idx is not None:
            cur = cells[best_idx]['text']
            cells[best_idx]['text'] = (cur + ' ' + it['text']).strip()
    return cells


def cluster_positions(values: List[float], tol: float) -> List[float]:
    if not values:
        return []
    values = sorted(values)
    groups = [[values[0]]]
    for v in values[1:]:
        if abs(v - np.mean(groups[-1])) <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [float(np.mean(g)) for g in groups]


def cells_to_matrix(cells):
    if not cells:
        return [], []
    row_keys = cluster_positions([c['cy'] for c in cells], tol=18)
    col_keys = cluster_positions([c['cx'] for c in cells], tol=30)
    matrix   = [['' for _ in col_keys] for _ in row_keys]
    meta     = []
    for c in cells:
        r   = min(range(len(row_keys)), key=lambda i: abs(c['cy']-row_keys[i]))
        col = min(range(len(col_keys)), key=lambda i: abs(c['cx']-col_keys[i]))
        matrix[r][col] = ((matrix[r][col]+' '+c['text']).strip() if matrix[r][col] else c['text'])
        meta.append({'row': r, 'col': col,
                     'bbox': [round(c['x1'],1), round(c['y1'],1), round(c['x2'],1), round(c['y2'],1)],
                     'text': c['text']})
    return matrix, meta


def matrix_to_markdown(matrix) -> str:
    if not matrix:
        return ''
    max_cols = max(len(r) for r in matrix)
    matrix   = [r + ['']*(max_cols-len(r)) for r in matrix]
    header, body = matrix[0], matrix[1:]
    md = ['| ' + ' | '.join(c or ' ' for c in header) + ' |',
          '| ' + ' | '.join(['---']*max_cols) + ' |']
    for row in body:
        md.append('| ' + ' | '.join(c or ' ' for c in row) + ' |')
    return '\n'.join(md)


def draw_table_cells(img_bgr: np.ndarray, cells, out_path: Path):
    vis = img_bgr.copy()
    for c in cells:
        x1,y1,x2,y2 = map(int,[c['x1'],c['y1'],c['x2'],c['y2']])
        cv2.rectangle(vis,(x1,y1),(x2,y2),(0,180,255),2)
        cv2.putText(vis, str(c['id']), (x1,max(12,y1+12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    cv2.imwrite(str(out_path), vis)


def run_table(table_engine, ocr_engine, crop_bgr: np.ndarray, crop_name: str):
    proc       = preprocess_for_ocr(crop_bgr)
    cells_orig = ppstructure_cells(table_engine, crop_bgr)
    cells_pre  = ppstructure_cells(table_engine, proc)
    use_pre    = len(cells_pre) > len(cells_orig)
    base_img   = proc if use_pre else crop_bgr
    cells      = cells_pre if use_pre else cells_orig

    if not cells:
        # fallback: plain OCR
        items = paddle_lines(ocr_engine, base_img)
        return note_to_text(items), [], [], ''

    ocr_items = paddle_lines(ocr_engine, base_img)
    cells     = assign_text_to_cells(cells, ocr_items)
    matrix, cell_meta = cells_to_matrix(cells)
    markdown  = matrix_to_markdown(matrix)

    vis_path  = TABLE_VIS / f"{crop_name}_cells.jpg"
    draw_table_cells(base_img, cells, vis_path)
    return markdown, cells, cell_meta, str(vis_path)


# ============================================================
# MAIN
# ============================================================
def main():
    predictor, metadata, cfg = build_predictor()
    ocr_engine   = build_ocr_engine()
    table_engine = build_table_engine()

    print("[OK] frcnn_x101 loaded")
    print(f"     device       : {cfg.MODEL.DEVICE}")
    print(f"     predictor thr: {SCORE_THRESH}")
    print(f"     per-class thr: PartDrawing={CLASS_THRESH['PartDrawing']:.2f} "
          f"Note={CLASS_THRESH['Note']:.2f} Table={CLASS_THRESH['Table']:.2f}")
    print("[OK] PaddleOCR loaded")
    print("[OK] PPStructure loaded")

    # Collect images (dedup case-insensitive)
    all_imgs = []
    for ext in ['*.jpg','*.jpeg','*.png','*.webp','*.JPG','*.PNG','*.JPEG']:
        all_imgs.extend(INPUT_DIR.glob(ext))
    all_imgs = sorted({str(p).lower(): p for p in all_imgs}.values(),
                      key=lambda p: p.name.lower())
    print(f"[DIR] {len(all_imgs)} images in {INPUT_DIR}\n")

    summary = []

    for img_idx, img_path in enumerate(all_imgs, start=1):
        print(f"[{img_idx:02d}/{len(all_imgs):02d}] {img_path.name}")
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print("  SKIP: unreadable"); continue

        # --- Detection ---
        outputs   = predictor(img_bgr)
        instances = outputs['instances'].to('cpu')
        scores_all  = instances.scores.numpy()            if instances.has('scores')       else []
        classes_all = instances.pred_classes.numpy()      if instances.has('pred_classes') else []
        boxes_all   = instances.pred_boxes.tensor.numpy() if instances.has('pred_boxes')   else []

        # Per-class filter
        keep_idx = [j for j, (c, s) in enumerate(zip(classes_all, scores_all))
                    if float(s) >= CLASS_THRESH[CLASS_NAMES[int(c)]]]
        filtered = instances[torch.tensor(keep_idx, dtype=torch.long)] if keep_idx else instances[:0]

        # Visualization (only kept instances)
        vis_img = (Visualizer(img_bgr[:, :, ::-1], metadata=metadata,
                              scale=1.0, instance_mode=ColorMode.SEGMENTATION)
                   .draw_instance_predictions(filtered)
                   .get_image()[:, :, ::-1])
        cv2.imwrite(str(VIS_DIR / f"{img_path.stem}_det.jpg"), vis_img)

        scores_f  = filtered.scores.numpy()            if filtered.has('scores')       else []
        classes_f = filtered.pred_classes.numpy()      if filtered.has('pred_classes') else []
        boxes_f   = filtered.pred_boxes.tensor.numpy() if filtered.has('pred_boxes')   else []

        objects  = []
        md_parts = [f"# Pipeline Final — {img_path.name}"]

        for i, (box, score, cls_id) in enumerate(zip(boxes_f, scores_f, classes_f), start=1):
            cls_name = CLASS_NAMES[int(cls_id)]
            x1,y1,x2,y2 = [int(round(v)) for v in box]

            if cls_name == 'Table':
                x1,y1,x2,y2 = add_padding(x1,y1,x2,y2,
                                           img_bgr.shape[1], img_bgr.shape[0])
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(img_bgr.shape[1], x2); y2 = min(img_bgr.shape[0], y2)
            crop = img_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_name = f"{img_path.stem}_{i:02d}_{cls_name}"
            crop_path = CROPS_DIR / f"{crop_name}.png"
            cv2.imwrite(str(crop_path), crop)

            ocr_content = ''; md_file = ''; pre_path = ''
            table_cells = []; table_vis_path = ''

            if cls_name == 'Note':
                proc = preprocess_for_ocr(crop)
                pre_p = PRE_DIR / f"{crop_name}_pre.png"
                cv2.imwrite(str(pre_p), proc)
                pre_path = str(pre_p)
                items_orig = paddle_lines(ocr_engine, crop)
                items_pre  = paddle_lines(ocr_engine, proc)
                ocr_items  = items_pre if len(items_pre) >= len(items_orig) else items_orig
                ocr_content = note_to_text(ocr_items)
                print(f"  - {cls_name:<12} conf={float(score):.3f}  "
                      f"lines={len(ocr_items)}  chars={len(ocr_content)}")

            elif cls_name == 'Table':
                proc = preprocess_for_ocr(crop)
                pre_p = PRE_DIR / f"{crop_name}_pre.png"
                cv2.imwrite(str(pre_p), proc)
                pre_path = str(pre_p)
                ocr_content, _, table_cells, table_vis_path = run_table(
                    table_engine, ocr_engine, crop, crop_name)
                print(f"  - {cls_name:<12} conf={float(score):.3f}  "
                      f"cells={len(table_cells)}  chars={len(ocr_content)}")

            else:  # PartDrawing — no OCR
                print(f"  - {cls_name:<12} conf={float(score):.3f}")

            if cls_name in ('Note', 'Table'):
                md_path = MD_DIR / f"{crop_name}.md"
                md_text = f"## {cls_name} {i}\n\n" + (ocr_content or '_empty_') + '\n'
                md_path.write_text(md_text, encoding='utf-8')
                md_file  = str(md_path)
                md_parts.append(md_text)

            objects.append({
                'id'               : i,
                'class'            : cls_name,
                'confidence'       : round(float(score), 4),
                'bbox'             : {'x1':x1,'y1':y1,'x2':x2,'y2':y2},
                'crop_path'        : str(crop_path),
                'preprocessed_path': pre_path,
                'ocr_content'      : ocr_content,
                'markdown_path'    : md_file,
                'table_cells'      : table_cells,
                'table_vis_path'   : table_vis_path,
            })

        # Save per-image markdown
        img_md_path = MD_DIR / f"{img_path.stem}.md"
        img_md_path.write_text('\n'.join(md_parts).strip() + '\n', encoding='utf-8')

        # Save per-image JSON
        out_json = {
            'image'           : img_path.name,
            'visualization'   : str(VIS_DIR / f"{img_path.stem}_det.jpg"),
            'num_detections'  : len(objects),
            'objects'         : objects,
            'markdown_summary': str(img_md_path),
        }
        json_path = JSON_DIR / f"{img_path.stem}.json"
        json_path.write_text(json.dumps(out_json, indent=2, ensure_ascii=False),
                              encoding='utf-8')

        summary.append({
            'image'           : img_path.name,
            'num_objects'     : len(objects),
            'num_note_table'  : sum(1 for o in objects if o['class'] in ('Note','Table')),
            'json_path'       : str(json_path),
            'markdown_summary': str(img_md_path),
        })
        print(f"  => {len(objects)} detections saved")

    # Global summary
    summary_path = OUT_DIR / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                             encoding='utf-8')
    total_dets = sum(s['num_objects'] for s in summary)
    print()
    print("=" * 60)
    print(f"[DONE] {len(summary)}/{len(all_imgs)} images processed")
    print(f"[TOTAL] {total_dets} detections across all images")
    print(f"[OUT]   {OUT_DIR}")
    print(f"[SUM]   {summary_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
