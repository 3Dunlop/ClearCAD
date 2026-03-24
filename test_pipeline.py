"""
test_pipeline.py -- Standalone end-to-end test for the CAD Legend Processor.

PRIMARY path: PDF vector analysis (PyMuPDF get_drawings + get_text).
FALLBACK path: Florence-2 detection + OCR + pixel swatch sampling.

Usage:
  python test_pipeline.py "D:/CAD/your_file.pdf" [page_index]

Outputs saved to: D:/CAD/output/
"""

import sys
import os
import json
import numpy as np
import cv2
import torch
import fitz  # PyMuPDF
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PDF_PATH        = sys.argv[1] if len(sys.argv) > 1 else r"D:/CAD/BSIP_Newhaven_Informal_TRO_plans_v3.pdf"
PAGE_INDEX      = int(sys.argv[2]) if len(sys.argv) > 2 else 0
DPI             = 600
OUTPUT_DIR      = r"D:/CAD/output"
FLORENCE2_NODES = r"C:/Users/miked/Documents/ComfyUI/custom_nodes/ComfyUI-Florence2"
MODEL_STORE     = r"C:/Users/miked/Documents/ComfyUI/models/LLM/Florence-2-large"
MODEL_ID        = "microsoft/Florence-2-large"

# Ensure our package and kijai's Florence-2 files are importable
_this_dir = os.path.dirname(os.path.abspath(__file__))
_custom_nodes_dir = os.path.dirname(_this_dir)
for p in [_custom_nodes_dir, FLORENCE2_NODES]:
    if p not in sys.path:
        sys.path.insert(0, p)

from comfyui_cad_legend.color_utils import (
    rgb_to_hex,
    find_swatch_left_of_text,
    find_swatch_right_of_text,
    create_mask_lab,
    apply_morphology,
    draw_legend_annotations,
    parse_florence2_ocr_with_region,
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32
print("Device : %s" % DEVICE)
print("DType  : %s" % DTYPE)


# ---------------------------------------------------------------------------
# Step 1: PDF render
# ---------------------------------------------------------------------------
def render_pdf_page(path, page_idx, dpi):
    print("\n[1] Rendering PDF: %s  page %d  @ %d DPI" % (os.path.basename(path), page_idx, dpi))
    doc  = fitz.open(path)
    print("    Pages: %d" % doc.page_count)
    page = doc[page_idx]
    mat  = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
    doc.close()
    arr  = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3).copy()
    print("    Size: %d x %d px" % (arr.shape[1], arr.shape[0]))
    return arr


# ---------------------------------------------------------------------------
# Step 2 (primary): PDF vector legend extraction
# ---------------------------------------------------------------------------
def extract_legend_via_pdf(pdf_path, page_idx, dpi, image_rgb=None):
    """
    Hybrid legend extraction:
      - PDF text extraction for label positions (reliable)
      - Raster pixel sampling at the KEY swatch position for colours (accurate)
    The vector 'fill/color' values in get_drawings() are PDF metadata colours
    and do NOT match the rendered raster; sampling the raster is more reliable.

    Returns (entries, legend_bbox_px).
    """
    print("\n[2] PDF+raster legend extraction...")

    doc  = fitz.open(pdf_path)
    page = doc[page_idx]
    scale = dpi / 72.0          # PDF points -> pixels

    # ---- Collect text lines (union spans per line) ----------------------
    raw_lines = []
    for block in page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]:
        if block.get("type") != 0:   # skip image blocks
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            txt   = " ".join(s["text"] for s in spans if s["text"].strip()).strip()
            if not txt:
                continue
            xs = [s["bbox"][0] for s in spans] + [s["bbox"][2] for s in spans]
            ys = [s["bbox"][1] for s in spans] + [s["bbox"][3] for s in spans]
            raw_lines.append({"text": txt, "bbox": (min(xs), min(ys), max(xs), max(ys))})

    print("    %d text lines on page." % len(raw_lines))

    # ---- Locate the KEY / LEGEND heading --------------------------------
    key_anchor = None
    for line in raw_lines:
        t = line["text"].strip().upper()
        if t in ("KEY", "LEGEND", "KEY:", "LEGEND:") or t.startswith("KEY ") or t.startswith("LEGEND "):
            key_anchor = line
            print("    KEY heading: '%s'  bbox %s" % (line["text"], [round(v,1) for v in line["bbox"]]))
            break

    if key_anchor is None:
        # Fallback: find a cluster of lines containing "PROPOSED"
        proposed = sorted(
            [l for l in raw_lines if "PROPOSED" in l["text"].upper()],
            key=lambda l: l["bbox"][1]
        )
        if proposed:
            synthetic_bbox = (
                min(l["bbox"][0] for l in proposed),
                proposed[0]["bbox"][1] - 5,
                max(l["bbox"][2] for l in proposed),
                proposed[-1]["bbox"][3] + 5,
            )
            key_anchor = {"text": "INFERRED_KEY", "bbox": synthetic_bbox}
            print("    KEY inferred from %d PROPOSED lines." % len(proposed))
        else:
            print("    ERROR: Cannot find KEY section.")
            doc.close()
            return [], None

    kx0, ky0, kx1, ky1 = key_anchor["bbox"]

    # ---- Narrow to KEY section: same X region as heading, ~130 pts tall ----
    # The KEY heading is typically a title box; labels start just below it at
    # similar x. Using a tight X filter avoids picking up drawing-body text.
    KEY_HEIGHT   = 130   # PDF points below heading to search
    LABEL_X_TOL  = 20    # how far left of heading x a label can start
    SWATCH_X_GAP = 130   # how far LEFT of heading x a swatch can be
    SWATCH_X_RIGHT = 200 # how far RIGHT of heading x

    label_lines = [
        l for l in raw_lines
        if l["bbox"][1] > ky0                       # below heading
        and l["bbox"][1] <= ky0 + KEY_HEIGHT
        and l["bbox"][0] >= kx0 - LABEL_X_TOL       # near heading x
        and l["text"].strip().upper() not in ("KEY", "LEGEND", "KEY:", "LEGEND:")
        and len(l["text"].strip()) > 4
    ]

    if not label_lines:
        print("    ERROR: No label lines found near KEY heading.")
        doc.close()
        return [], None

    ky1_actual = max(l["bbox"][3] for l in label_lines)
    sec_x_min  = kx0 - SWATCH_X_GAP
    sec_x_max  = kx0 + SWATCH_X_RIGHT

    ky1_actual = max(l["bbox"][3] for l in label_lines)
    # KEY swatch zone: just to the LEFT of the label text.
    # Each swatch is ~35-40 pts wide, ending a few pts before the text start.
    # We detect swatch_x relative to each label's individual tx0 (not the KEY heading).
    SWATCH_WIDTH = 45   # pts to the left of label text
    SWATCH_GAP   = 3    # pts of gap between swatch right and text left

    print("    KEY section: y=[%.1f, %.1f]  swatch: [tx0-%d, tx0-%d]  (%d labels)" % (
        ky0, ky1_actual, SWATCH_WIDTH, SWATCH_GAP, len(label_lines)))
    for l in label_lines:
        print("      y=%.1f  '%s'" % (l["bbox"][1], l["text"]))

    # Legend bbox in pixels for visualisation.
    # tx0 is consistent across all label_lines (~690.7 for this PDF).
    # Swatch starts SWATCH_WIDTH pts left of tx0.
    _tx0_ref = label_lines[0]["bbox"][0]
    legend_bbox_px = (
        int((_tx0_ref - SWATCH_WIDTH - 5) * scale),
        int(ky0 * scale),
        int((_tx0_ref + 200) * scale),
        int(ky1_actual * scale),
    )

    # ---- Pixel-sample the KEY swatch for each label ---------------------
    # The PDF vector colours (get_drawings) are metadata and don't match the
    # rendered raster. Sampling the actual pixels at the KEY swatch position
    # gives the colour as it appears in the image — which is what we need for
    # threshold-based masks.

    entries    = []
    seen_labs  = []

    if image_rgb is not None:
        H_img, W_img = image_rgb.shape[:2]
        px_per_pt = dpi / 72.0

        for span in label_lines:
            label = span["text"].strip()
            tx0_pt, ty0_pt, tx1_pt, ty1_pt = span["bbox"]

            # Swatch box in pixels
            sx1 = max(0, int((tx0_pt - SWATCH_WIDTH) * px_per_pt))
            sx2 = min(W_img, int((tx0_pt - SWATCH_GAP) * px_per_pt))
            sy1 = max(0, int(ty0_pt * px_per_pt))
            sy2 = min(H_img, int(ty1_pt * px_per_pt))

            if sx2 <= sx1 or sy2 <= sy1:
                print("    Skip (invalid swatch region): '%s'" % label)
                continue

            region = image_rgb[sy1:sy2, sx1:sx2]

            # Sample pixels: exclude near-white paper and unsaturated bright grey.
            flat = region.reshape(-1, 3).astype(float)
            flat_sum = flat[:, 0] + flat[:, 1] + flat[:, 2]
            flat_sat = np.max(flat, axis=1) - np.min(flat, axis=1)

            # Primary: coloured pixels (saturation > 10 and not too bright)
            primary = flat[(flat_sum < 690) & (flat_sat > 10)]
            # Secondary fallback: any non-white pixel (catches dark grey swatches)
            secondary = flat[(flat_sum < 640)]

            if len(primary) >= 3:
                filtered = primary
                src = "raster_sample"
            elif len(secondary) >= 3:
                filtered = secondary
                src = "raster_sample_grey"
            else:
                filtered = None

            if filtered is None or len(filtered) < 3:
                # Swatch zone is all white/near-white → no swatch here
                r, g, b = 152, 152, 152
                src = "grey_no_swatch"
            else:
                r = int(np.median(filtered[:, 0]))
                g = int(np.median(filtered[:, 1]))
                b = int(np.median(filtered[:, 2]))
                src = "raster_sample"

            hex_colour = rgb_to_hex(r, g, b)

            # CIE76 deduplication (skip colours already well-covered)
            c_arr = np.uint8([[[r, g, b]]])
            lab1  = cv2.cvtColor(c_arr, cv2.COLOR_RGB2LAB).astype(float)[0, 0]
            if any(np.sqrt(np.sum((lab1 - sl)**2)) < 12.0 for sl in seen_labs):
                # Allow grey duplicates (multiple grey features get same mask)
                sat = max(r, g, b) - min(r, g, b)
                if sat > 15:
                    print("    [dup] '%s'  %s  (skipping duplicate colour)" % (label, hex_colour))
                    # Still add to entries so label appears in legend, but with a note
                    entries.append({"label": label, "rgb": [r, g, b],
                                     "hex": hex_colour, "source": src + "_dup"})
                    continue
            seen_labs.append(lab1)

            entries.append({"label": label, "rgb": [r, g, b],
                             "hex": hex_colour, "source": src})
            print("    '%s'  ->  %s  RGB(%d,%d,%d)  [%s]" % (label, hex_colour, r, g, b, src))
    else:
        # No raster available: use vector drawings as fallback
        print("    (no raster provided; using vector colours only)")
        def colour_tuple_to_rgb(c):
            return (int(round(c[0]*255)), int(round(c[1]*255)), int(round(c[2]*255)))
        all_drawings = page.get_drawings()
        for span in label_lines:
            label = span["text"].strip()
            entries.append({"label": label, "rgb": [152, 152, 152],
                             "hex": "#989898", "source": "no_raster"})

    doc.close()
    print("\n    Total: %d legend entries." % len(entries))
    return entries, legend_bbox_px


# ---------------------------------------------------------------------------
# Florence-2 loading (used only if vector extraction fails)
# ---------------------------------------------------------------------------
def download_model_if_needed():
    if os.path.isdir(MODEL_STORE) and any(
        f.endswith((".safetensors", ".bin")) for f in os.listdir(MODEL_STORE)
    ):
        print("    Model found at: %s" % MODEL_STORE)
        return
    print("    Downloading Florence-2-large to: %s" % MODEL_STORE)
    from huggingface_hub import snapshot_download
    os.makedirs(MODEL_STORE, exist_ok=True)
    snapshot_download(repo_id=MODEL_ID, local_dir=MODEL_STORE, local_dir_use_symlinks=False)
    print("    Download complete.")


def _register_florence2_package():
    import types
    import importlib.util

    pkg_name = "_comfyui_florence2"
    if pkg_name in sys.modules:
        return

    pkg             = types.ModuleType(pkg_name)
    pkg.__path__    = [FLORENCE2_NODES]
    pkg.__file__    = os.path.join(FLORENCE2_NODES, "__init__.py")
    pkg.__package__ = pkg_name
    sys.modules[pkg_name] = pkg

    sub_mods = ["configuration_florence2", "modeling_florence2", "processing_florence2"]
    for mod_name in sub_mods:
        spec = importlib.util.spec_from_file_location(
            "%s.%s" % (pkg_name, mod_name),
            os.path.join(FLORENCE2_NODES, "%s.py" % mod_name),
        )
        mod             = importlib.util.module_from_spec(spec)
        mod.__package__ = pkg_name
        sys.modules["%s.%s" % (pkg_name, mod_name)] = mod

    for mod_name in sub_mods:
        sys.modules["%s.%s" % (pkg_name, mod_name)].__spec__.loader.exec_module(
            sys.modules["%s.%s" % (pkg_name, mod_name)]
        )


def load_florence2():
    print("\n[F] Loading Florence-2-large (fallback)...")
    download_model_if_needed()
    _register_florence2_package()

    from _comfyui_florence2.modeling_florence2   import Florence2ForConditionalGeneration, Florence2Config
    from _comfyui_florence2.processing_florence2 import Florence2Processor
    from transformers       import CLIPImageProcessor, BartTokenizerFast
    from safetensors.torch  import load_file as load_safetensors
    from accelerate         import init_empty_weights
    from accelerate.utils   import set_module_tensor_to_device

    config = Florence2Config.from_pretrained(MODEL_STORE)
    config._attn_implementation = "sdpa"

    with init_empty_weights():
        model = Florence2ForConditionalGeneration(config)

    ckpt = os.path.join(MODEL_STORE, "model.safetensors")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(MODEL_STORE, "pytorch_model.bin")
    if not os.path.exists(ckpt):
        raise FileNotFoundError("No model weights found in %s" % MODEL_STORE)

    print("    Loading weights: %s" % os.path.basename(ckpt))
    if ckpt.endswith(".safetensors"):
        state_dict = load_safetensors(ckpt, device="cpu")
    else:
        state_dict = torch.load(ckpt, map_location="cpu")

    key_mapping = {}
    if "language_model.model.shared.weight" in state_dict:
        key_mapping["language_model.model.encoder.embed_tokens.weight"] = "language_model.model.shared.weight"
        key_mapping["language_model.model.decoder.embed_tokens.weight"] = "language_model.model.shared.weight"

    for name, _ in model.named_parameters():
        actual_key = key_mapping.get(name, name)
        if actual_key in state_dict:
            set_module_tensor_to_device(model, name, DEVICE, value=state_dict[actual_key].to(DTYPE))

    model.language_model.tie_weights()
    model = model.eval().to(DTYPE).to(DEVICE)

    image_processor = CLIPImageProcessor(
        do_resize=True,
        size={"height": 768, "width": 768},
        resample=3,
        do_center_crop=False,
        do_rescale=True,
        rescale_factor=1/255.0,
        do_normalize=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )
    image_processor.image_seq_length = 577
    tokenizer = BartTokenizerFast.from_pretrained(MODEL_STORE)
    processor = Florence2Processor(image_processor=image_processor, tokenizer=tokenizer)

    print("    Florence-2-large ready.")
    return model, processor


def run_florence2(model, processor, image_pil, task_prompt, text_input=""):
    prompt = task_prompt if not text_input else task_prompt + " " + text_input
    inputs = processor(text=prompt, images=image_pil, return_tensors="pt")
    inputs = {k: v.to(DEVICE, dtype=DTYPE if v.dtype.is_floating_point else v.dtype)
              for k, v in inputs.items()}
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
            use_cache=True,
        )
    text_out = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return processor.post_process_generation(
        text_out, task=task_prompt,
        image_size=(image_pil.width, image_pil.height)
    )


LEGEND_KEYWORDS = {"legend", "key", "schedule", "notation", "symbols", "notes",
                   "marking", "colour code", "color code", "proposed", "existing"}

def _label_is_legend(label):
    lbl = label.lower()
    return any(k in lbl for k in LEGEND_KEYWORDS)


def detect_legend_florence(model, processor, image_rgb):
    H, W = image_rgb.shape[:2]
    max_side = 1024
    scale    = min(max_side / W, max_side / H)
    tw, th   = int(W * scale), int(H * scale)
    thumb    = cv2.resize(image_rgb, (tw, th), interpolation=cv2.INTER_AREA)
    pil_th   = Image.fromarray(thumb)
    print("[F3a] Florence-2 dense_region_caption on %dx%d thumbnail..." % (tw, th))

    result = run_florence2(model, processor, pil_th, "<DENSE_REGION_CAPTION>")
    key    = "<DENSE_REGION_CAPTION>"
    b_list = result.get(key, {}).get("bboxes", [])
    l_list = result.get(key, {}).get("labels", [])
    print("     %d regions." % len(b_list))

    matched  = [(b, l) for b, l in zip(b_list, l_list) if _label_is_legend(l)]
    bbox_raw = None
    if matched:
        matched.sort(key=lambda x: (x[0][2]-x[0][0])*(x[0][3]-x[0][1]), reverse=True)
        b, lbl = matched[0]
        print("     Matched: '%s'" % lbl)
        sx, sy   = W / tw, H / th
        bbox_raw = [int(b[0]*sx), int(b[1]*sy), int(b[2]*sx), int(b[3]*sy)]

    if bbox_raw:
        x1, y1, x2, y2 = bbox_raw
    else:
        print("     Heuristic: bottom-right 38%%.")
        x1, y1, x2, y2 = int(W*0.62), int(H*0.62), W, H

    pad = 30
    x1, y1 = max(0, x1-pad), max(0, y1-pad)
    x2, y2 = min(W, x2+pad), min(H, y2+pad)
    print("     Legend crop: (%d,%d) -> (%d,%d)" % (x1, y1, x2, y2))
    return (x1, y1, x2, y2), image_rgb[y1:y2, x1:x2]


def ocr_legend_florence(model, processor, legend_rgb):
    print("\n[F4] Florence-2: OCR_WITH_REGION...")
    H, W   = legend_rgb.shape[:2]
    pil    = Image.fromarray(legend_rgb)
    result = run_florence2(model, processor, pil, "<OCR_WITH_REGION>")
    print("    Raw: %s" % json.dumps(result)[:600])
    ocr_entries = parse_florence2_ocr_with_region(json.dumps(result), (W, H))
    print("    %d text region(s) found." % len(ocr_entries))
    for e in ocr_entries:
        print("      '%s'  @  %s" % (e["text"], e["bbox"]))
    return ocr_entries


def extract_swatches_from_image(legend_rgb, ocr_entries):
    print("\n[F5] Pixel swatch extraction...")
    entries   = []
    seen_labs = []
    for ocr in ocr_entries:
        label     = ocr["text"].strip()
        text_bbox = ocr["bbox"]
        if not label:
            continue
        result = find_swatch_left_of_text(legend_rgb, text_bbox, search_width=120, gap=4, variance_threshold=22.0)
        if result is None:
            result = find_swatch_right_of_text(legend_rgb, text_bbox, search_width=120, gap=4, variance_threshold=22.0)
        if result is None:
            print("    No swatch: '%s'" % label)
            continue
        swatch_bbox, (r, g, b) = result
        c_arr = np.uint8([[[r, g, b]]])
        lab1  = cv2.cvtColor(c_arr, cv2.COLOR_RGB2LAB).astype(float)[0, 0]
        if any(np.sqrt(np.sum((lab1 - sl)**2)) < 8.0 for sl in seen_labs):
            print("    Duplicate for '%s'." % label)
            continue
        seen_labs.append(lab1)
        entries.append({
            "label":       label,
            "rgb":         [r, g, b],
            "hex":         rgb_to_hex(r, g, b),
            "swatch_bbox": list(swatch_bbox),
            "text_bbox":   list(text_bbox),
        })
        print("    -> '%s'  %s  RGB(%d,%d,%d)" % (label, rgb_to_hex(r,g,b), r, g, b))
    print("    %d entries." % len(entries))
    return entries


# ---------------------------------------------------------------------------
# Mask generation
# ---------------------------------------------------------------------------
def generate_masks(image_rgb, entries, tolerance=25.0):
    print("\n[6] Generating masks (LAB CIE76, tolerance=%.1f)..." % tolerance)
    masks = {}
    for entry in entries:
        r, g, b = entry["rgb"]
        mask    = create_mask_lab(image_rgb, (r, g, b), tolerance=tolerance)
        mask    = apply_morphology(mask, kernel_size=3)
        pct     = float(np.sum(mask > 0)) / mask.size * 100
        masks[entry["label"]] = mask
        print("    '%s'  %s  -- %.2f%%" % (entry["label"], entry["hex"], pct))
    return masks


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------
def save_outputs(image_rgb, entries, masks, legend_bbox_px, page_idx):
    print("\n[7] Saving to: %s" % OUTPUT_DIR)
    pfx = "page%02d" % page_idx

    # Raw full-resolution render
    cv2.imwrite(os.path.join(OUTPUT_DIR, "%s_raw.png" % pfx),
                cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    print("  %s_raw.png" % pfx)

    # Full drawing with KEY region highlighted
    if legend_bbox_px:
        ann = image_rgb.copy()
        x1, y1, x2, y2 = legend_bbox_px
        cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 200, 0), 6)
        cv2.putText(ann, "KEY", (x1, max(0, y1-12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,0,0), 6)
        cv2.putText(ann, "KEY", (x1, max(0, y1-12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,200,0), 3)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "%s_key_region.png" % pfx),
                    cv2.cvtColor(ann, cv2.COLOR_RGB2BGR))
        print("  %s_key_region.png" % pfx)

        # Crop KEY area for close inspection
        key_crop = image_rgb[y1:y2, x1:x2]
        cv2.imwrite(os.path.join(OUTPUT_DIR, "%s_key_crop.png" % pfx),
                    cv2.cvtColor(key_crop, cv2.COLOR_RGB2BGR))
        print("  %s_key_crop.png" % pfx)

    # Legend JSON
    with open(os.path.join(OUTPUT_DIR, "%s_legend.json" % pfx), "w") as f:
        json.dump(entries, f, indent=2)
    print("  %s_legend.json" % pfx)

    # Individual masks
    mask_dir = os.path.join(OUTPUT_DIR, "%s_masks" % pfx)
    os.makedirs(mask_dir, exist_ok=True)
    for i, (label, mask) in enumerate(masks.items()):
        safe = label.replace(" ", "_").replace("/", "-")[:60]
        cv2.imwrite(os.path.join(mask_dir, "%03d_%s.png" % (i, safe)), mask)
    print("  %d mask(s) -> %s/" % (len(masks), mask_dir))

    # Colour-tinted composite overlay
    comp = image_rgb.copy().astype(float)
    for entry in entries:
        lbl = entry["label"]
        if lbl not in masks:
            continue
        r, g, b = entry["rgb"]
        blend = (masks[lbl] > 0).astype(float)[:, :, np.newaxis]
        tint  = np.full_like(comp, [r, g, b], dtype=float)
        comp  = comp * (1 - blend * 0.45) + tint * (blend * 0.45)
    comp = np.clip(comp, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "%s_composite.png" % pfx),
                cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
    print("  %s_composite.png" % pfx)

    # Summary table
    print("\n  Legend table:")
    print("  " + "="*60)
    for i, e in enumerate(entries):
        r, g, b = e["rgb"]
        src = e.get("source", "?")
        pct_str = ""
        if e["label"] in masks:
            pct = float(np.sum(masks[e["label"]] > 0)) / masks[e["label"]].size * 100
            pct_str = "  %.2f%%" % pct
        print("  %2d. %s  RGB(%3d,%3d,%3d)  %-8s  %s%s" % (
            i+1, e["hex"], r, g, b, "[%s]" % src, e["label"], pct_str))
    print("  " + "="*60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("PDF   : %s" % PDF_PATH)
    print("Page  : %d" % PAGE_INDEX)
    print("Output: %s\n" % OUTPUT_DIR)

    image_rgb = render_pdf_page(PDF_PATH, PAGE_INDEX, DPI)

    # --- Primary path: PDF + raster extraction ---
    entries, legend_bbox_px = extract_legend_via_pdf(PDF_PATH, PAGE_INDEX, DPI, image_rgb)

    if entries:
        print("\n[3] Vector extraction succeeded: %d entries." % len(entries))
        tolerance = 25.0
    else:
        # --- Fallback: Florence-2 OCR + pixel swatch sampling ---
        print("\n[3] Falling back to Florence-2 pipeline...")
        model, processor = load_florence2()

        legend_bbox_px, legend_rgb = detect_legend_florence(model, processor, image_rgb)

        cv2.imwrite(
            os.path.join(OUTPUT_DIR, "page%02d_legend_crop.png" % PAGE_INDEX),
            cv2.cvtColor(legend_rgb, cv2.COLOR_RGB2BGR),
        )
        print("    Legend crop saved.")

        ocr_entries = ocr_legend_florence(model, processor, legend_rgb)
        if not ocr_entries:
            print("\nWARNING: Florence-2 returned no OCR results. Exiting.")
            sys.exit(1)

        entries = extract_swatches_from_image(legend_rgb, ocr_entries)
        if not entries:
            print("\nWARNING: No swatches found. Exiting.")
            sys.exit(1)

        tolerance = 20.0  # pixel-sampled colours need slightly tighter tolerance

    masks = generate_masks(image_rgb, entries, tolerance=tolerance)
    save_outputs(image_rgb, entries, masks, legend_bbox_px, PAGE_INDEX)

    print("\nPipeline complete.")
    print("All outputs in: %s" % OUTPUT_DIR)
