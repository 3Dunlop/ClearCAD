"""
nodes_legend.py — Legend region detection, swatch extraction, and display nodes.

Nodes:
  CAD_LegendCropFromFlorence2  — Crop legend area using Florence-2 detection JSON
  CAD_ManualLegendCrop         — Crop legend area using manual pixel coordinates
  CAD_SwatchExtractor          — Extract colour+label pairs from cropped legend + OCR JSON
  CAD_PDFSwatchExtractor       — Direct PDF+raster extraction (no Florence-2 required)
  CAD_LegendDisplay            — Format and display legend data as text
"""

import json
import numpy as np
import cv2
from typing import Any, Dict, List, Optional, Tuple

from .color_utils import (
    tensor_to_numpy,
    numpy_to_tensor,
    rgb_to_hex,
    parse_florence2_detection,
    parse_florence2_ocr_with_region,
    find_swatch_left_of_text,
    find_swatch_right_of_text,
    draw_legend_annotations,
)

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


# ---------------------------------------------------------------------------
# CAD_LegendCropFromFlorence2
# ---------------------------------------------------------------------------

class CAD_LegendCropFromFlorence2:
    """
    Crops the legend / key region from a CAD drawing image using the
    bounding box returned by Florence-2 OPEN_VOCABULARY_DETECTION.

    Connect the Florence2Run result_json output here.
    If Florence-2 finds no match, the full image is passed through with a warning.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "florence2_detection_json": ("STRING", {"multiline": True, "default": "{}"}),
                "detection_label": ("STRING", {
                    "default": "legend",
                    "placeholder": "e.g. legend, key, symbol table"
                }),
                "padding_px": ("INT", {
                    "default": 12,
                    "min": 0,
                    "max": 200,
                    "step": 4,
                    "display": "number"
                }),
                "pick_largest": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("legend_image", "bbox_json", "x1", "y1", "x2", "y2")
    FUNCTION = "crop_legend"
    CATEGORY = "CAD Legend Processor"

    def crop_legend(self, image, florence2_detection_json: str,
                    detection_label: str, padding_px: int, pick_largest: bool):

        img_np = tensor_to_numpy(image)
        H, W = img_np.shape[:2]

        bboxes = parse_florence2_detection(florence2_detection_json, detection_label)

        if not bboxes:
            print(f"[CAD Legend] WARNING: No '{detection_label}' detected — using full image.")
            return (image, json.dumps({"x1": 0, "y1": 0, "x2": W, "y2": H}), 0, 0, W, H)

        if pick_largest:
            bboxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)

        x1, y1, x2, y2 = bboxes[0]

        # Apply padding, clamped to image bounds
        x1 = max(0, x1 - padding_px)
        y1 = max(0, y1 - padding_px)
        x2 = min(W, x2 + padding_px)
        y2 = min(H, y2 + padding_px)

        crop = img_np[y1:y2, x1:x2]
        crop_tensor = numpy_to_tensor(crop)
        bbox_json = json.dumps({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

        return (crop_tensor, bbox_json, x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# CAD_ManualLegendCrop
# ---------------------------------------------------------------------------

class CAD_ManualLegendCrop:
    """
    Crop the legend area using manually specified pixel coordinates.
    Use this when drawings follow a consistent template — it is faster and
    100% reliable for known formats.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "x1": ("INT", {"default": 0,    "min": 0, "max": 99999}),
                "y1": ("INT", {"default": 0,    "min": 0, "max": 99999}),
                "x2": ("INT", {"default": 512,  "min": 1, "max": 99999}),
                "y2": ("INT", {"default": 512,  "min": 1, "max": 99999}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("legend_image", "bbox_json")
    FUNCTION = "crop"
    CATEGORY = "CAD Legend Processor"

    def crop(self, image, x1: int, y1: int, x2: int, y2: int):
        img_np = tensor_to_numpy(image)
        H, W = img_np.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        crop = img_np[y1:y2, x1:x2]
        return (numpy_to_tensor(crop), json.dumps({"x1": x1, "y1": y1, "x2": x2, "y2": y2}))


# ---------------------------------------------------------------------------
# CAD_SwatchExtractor
# ---------------------------------------------------------------------------

class CAD_SwatchExtractor:
    """
    Core legend parser.

    Given:
      - The cropped legend image
      - Florence-2 OCR_WITH_REGION JSON (text + bounding boxes)

    This node:
      1. Iterates over each OCR'd text label
      2. Searches left (or right) of the text for a solid-colour swatch
      3. Samples the dominant colour from that swatch
      4. Outputs a JSON array: [{label, rgb, hex, swatch_bbox, text_bbox}, ...]
      5. Outputs an annotated preview image

    Adjust `swatch_search_width` if swatches aren't found.
    Adjust `variance_threshold` upward for hatched/gradient fills (triggers a warning).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "legend_image": ("IMAGE",),
                "florence2_ocr_json": ("STRING", {"multiline": True, "default": "{}"}),
                "swatch_side": (["LEFT", "RIGHT", "BOTH"], {"default": "LEFT"}),
                "swatch_search_width": ("INT", {
                    "default": 90,
                    "min": 10,
                    "max": 400,
                    "step": 5,
                    "display": "number"
                }),
                "variance_threshold": ("FLOAT", {
                    "default": 18.0,
                    "min": 1.0,
                    "max": 80.0,
                    "step": 0.5,
                    "display": "number"
                }),
                "gap_px": ("INT", {
                    "default": 4,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "display": "number"
                }),
                "deduplicate_colors": ("BOOLEAN", {"default": True}),
                "dedup_cie76_threshold": ("FLOAT", {
                    "default": 8.0,
                    "min": 1.0,
                    "max": 50.0,
                    "step": 0.5,
                    "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", "INT")
    RETURN_NAMES = ("legend_json", "preview_image", "entry_count")
    FUNCTION = "extract_swatches"
    CATEGORY = "CAD Legend Processor"

    def extract_swatches(self, legend_image, florence2_ocr_json: str,
                         swatch_side: str, swatch_search_width: int,
                         variance_threshold: float, gap_px: int,
                         deduplicate_colors: bool, dedup_cie76_threshold: float):

        img_np = tensor_to_numpy(legend_image)
        H, W = img_np.shape[:2]

        # Parse OCR results
        ocr_entries = parse_florence2_ocr_with_region(florence2_ocr_json, (W, H))

        if not ocr_entries:
            print("[CAD Legend] WARNING: No OCR results found. Check Florence-2 OCR_WITH_REGION output.")
            empty_json = json.dumps([])
            return (empty_json, legend_image, 0)

        entries = []
        seen_colors = []  # for deduplication

        for ocr in ocr_entries:
            label = ocr["text"]
            text_bbox = ocr["bbox"]

            result = None
            if swatch_side in ("LEFT", "BOTH"):
                result = find_swatch_left_of_text(
                    img_np, text_bbox,
                    search_width=swatch_search_width,
                    gap=gap_px,
                    variance_threshold=variance_threshold,
                )
            if result is None and swatch_side in ("RIGHT", "BOTH"):
                result = find_swatch_right_of_text(
                    img_np, text_bbox,
                    search_width=swatch_search_width,
                    gap=gap_px,
                    variance_threshold=variance_threshold,
                )

            if result is None:
                print(f"[CAD Legend] No swatch found for label: '{label}' — skipping.")
                continue

            swatch_bbox, (r, g, b) = result

            # Deduplication: skip if a very similar colour was already found
            if deduplicate_colors:
                import cv2 as _cv2
                c1 = np.uint8([[[r, g, b]]])
                lab1 = _cv2.cvtColor(c1, _cv2.COLOR_RGB2LAB).astype(float)[0, 0]
                is_dup = False
                for sc in seen_colors:
                    c2 = np.uint8([[[sc[0], sc[1], sc[2]]]])
                    lab2 = _cv2.cvtColor(c2, _cv2.COLOR_RGB2LAB).astype(float)[0, 0]
                    dist = float(np.sqrt(np.sum((lab1 - lab2) ** 2)))
                    if dist < dedup_cie76_threshold:
                        print(f"[CAD Legend] Duplicate colour for '{label}' "
                              f"(ΔE={dist:.1f} < {dedup_cie76_threshold}) — skipping.")
                        is_dup = True
                        break
                if is_dup:
                    continue
                seen_colors.append((r, g, b))

            entries.append({
                "label": label,
                "rgb": [r, g, b],
                "hex": rgb_to_hex(r, g, b),
                "swatch_bbox": list(swatch_bbox),
                "text_bbox": list(text_bbox),
            })

        if not entries:
            print("[CAD Legend] WARNING: No swatches extracted. "
                  "Try increasing swatch_search_width or variance_threshold.")

        legend_json = json.dumps(entries, indent=2)
        preview_np = draw_legend_annotations(img_np, entries)
        preview_tensor = numpy_to_tensor(preview_np)

        print(f"[CAD Legend] Extracted {len(entries)} legend entries.")
        return (legend_json, preview_tensor, len(entries))


# ---------------------------------------------------------------------------
# CAD_PDFSwatchExtractor
# ---------------------------------------------------------------------------

class CAD_PDFSwatchExtractor:
    """
    Direct PDF + raster legend extraction.  No Florence-2 required.

    Uses PyMuPDF to locate the KEY/LEGEND heading and label text positions,
    then samples the rendered raster at each swatch region to get the actual
    pixel colour (more reliable than PDF vector metadata colours).

    Connect the same pdf_path, page_index, and dpi as CAD_PDFToImage.
    The dpi must match what was used to render the image.

    Returns legend_json in the same format as CAD_SwatchExtractor so it can
    be wired directly into CAD_BatchMaskFromLegend.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "pdf_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Path to PDF file (e.g. D:/CAD/plan.pdf)",
                }),
                "page_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                    "step": 1,
                    "display": "number",
                }),
                "dpi": ("INT", {
                    "default": 600,
                    "min": 72,
                    "max": 1200,
                    "step": 50,
                    "display": "number",
                }),
                "key_height_pts": ("INT", {
                    "default": 130,
                    "min": 20,
                    "max": 500,
                    "step": 5,
                    "display": "number",
                }),
                "swatch_width_pts": ("INT", {
                    "default": 45,
                    "min": 5,
                    "max": 150,
                    "step": 5,
                    "display": "number",
                }),
                "dedup_threshold": ("FLOAT", {
                    "default": 12.0,
                    "min": 1.0,
                    "max": 50.0,
                    "step": 0.5,
                    "display": "number",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE", "INT")
    RETURN_NAMES = ("legend_json", "preview_image", "entry_count")
    FUNCTION = "extract"
    CATEGORY = "CAD Legend Processor"

    def extract(self, image, pdf_path: str = "", page_index: int = 0,
                dpi: int = 600, key_height_pts: int = 130,
                swatch_width_pts: int = 45, dedup_threshold: float = 12.0):

        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF is not installed. Run: pip install PyMuPDF")

        import os
        pdf_path = pdf_path.strip().strip('"').strip("'")
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF not found: '{pdf_path}'")

        img_np = tensor_to_numpy(image)
        H_img, W_img = img_np.shape[:2]
        scale = dpi / 72.0          # PDF points → pixels

        # ---- Open PDF and collect text lines --------------------------------
        doc  = fitz.open(pdf_path)
        page = doc[page_index]

        raw_lines = []
        for block in page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                txt   = " ".join(s["text"] for s in spans if s["text"].strip()).strip()
                if not txt:
                    continue
                xs = [s["bbox"][0] for s in spans] + [s["bbox"][2] for s in spans]
                ys = [s["bbox"][1] for s in spans] + [s["bbox"][3] for s in spans]
                raw_lines.append({"text": txt,
                                  "bbox": (min(xs), min(ys), max(xs), max(ys))})
        doc.close()

        # ---- Locate KEY / LEGEND heading ------------------------------------
        key_anchor = None
        for line in raw_lines:
            t = line["text"].strip().upper()
            if (t in ("KEY", "LEGEND", "KEY:", "LEGEND:")
                    or t.startswith("KEY ") or t.startswith("LEGEND ")):
                key_anchor = line
                print(f"[CAD PDF] KEY heading: '{line['text']}'  "
                      f"bbox {[round(v,1) for v in line['bbox']]}")
                break

        if key_anchor is None:
            # Fallback: infer from PROPOSED lines
            proposed = sorted(
                [l for l in raw_lines if "PROPOSED" in l["text"].upper()],
                key=lambda l: l["bbox"][1],
            )
            if proposed:
                key_anchor = {
                    "text": "INFERRED_KEY",
                    "bbox": (
                        min(l["bbox"][0] for l in proposed),
                        proposed[0]["bbox"][1] - 5,
                        max(l["bbox"][2] for l in proposed),
                        proposed[-1]["bbox"][3] + 5,
                    ),
                }
                print(f"[CAD PDF] KEY inferred from {len(proposed)} PROPOSED lines.")
            else:
                print("[CAD PDF] ERROR: Cannot find KEY section. "
                      "Returning empty legend.")
                return (json.dumps([]), image, 0)

        kx0, ky0, kx1, ky1 = key_anchor["bbox"]
        LABEL_X_TOL = 20

        # Phrases that appear near the KEY area but are not legend labels
        _NON_LEGEND = {
            "KEY", "LEGEND", "KEY:", "LEGEND:",
        }
        _NON_LEGEND_CONTAINS = [
            "CONTINUATION", "VIEWPORT", "SEE SHEET", "REFER TO",
            "DRAWING NO", "SCALE", "DATE", "REV ", "NOTES",
        ]

        label_lines = [
            l for l in raw_lines
            if l["bbox"][1] > ky0
            and l["bbox"][1] <= ky0 + key_height_pts
            and l["bbox"][0] >= kx0 - LABEL_X_TOL
            and l["text"].strip().upper() not in _NON_LEGEND
            and len(l["text"].strip()) > 4
            and not any(phrase in l["text"].upper()
                        for phrase in _NON_LEGEND_CONTAINS)
        ]

        if not label_lines:
            print("[CAD PDF] ERROR: No label lines found near KEY heading.")
            return (json.dumps([]), image, 0)

        ky1_actual = max(l["bbox"][3] for l in label_lines)
        print(f"[CAD PDF] KEY section: y=[{ky0:.1f}, {ky1_actual:.1f}]  "
              f"{len(label_lines)} labels  swatch=[tx0-{swatch_width_pts}, tx0-3]")

        SWATCH_GAP = 3   # pts gap between swatch right edge and text left

        # ---- Pixel-sample each swatch ---------------------------------------
        entries   = []
        seen_labs = []

        for span in label_lines:
            label = span["text"].strip()
            tx0_pt, ty0_pt, tx1_pt, ty1_pt = span["bbox"]

            sx1 = max(0, int((tx0_pt - swatch_width_pts) * scale))
            sx2 = min(W_img, int((tx0_pt - SWATCH_GAP) * scale))
            sy1 = max(0, int(ty0_pt * scale))
            sy2 = min(H_img, int(ty1_pt * scale))

            if sx2 <= sx1 or sy2 <= sy1:
                continue

            region = img_np[sy1:sy2, sx1:sx2]
            flat   = region.reshape(-1, 3).astype(float)
            flat_sum = flat[:, 0] + flat[:, 1] + flat[:, 2]
            flat_sat = np.max(flat, axis=1) - np.min(flat, axis=1)

            primary   = flat[(flat_sum < 690) & (flat_sat > 10)]
            secondary = flat[flat_sum < 640]

            if len(primary) >= 3:
                filtered = primary
                src = "pdf_raster"
            elif len(secondary) >= 3:
                filtered = secondary
                src = "pdf_raster_grey"
            else:
                filtered = None

            if filtered is None or len(filtered) < 3:
                r, g, b = 152, 152, 152
                src = "pdf_no_swatch"
            else:
                r = int(np.median(filtered[:, 0]))
                g = int(np.median(filtered[:, 1]))
                b = int(np.median(filtered[:, 2]))

            hex_colour = rgb_to_hex(r, g, b)

            # CIE76 deduplication
            c_arr = np.uint8([[[r, g, b]]])
            lab1  = cv2.cvtColor(c_arr, cv2.COLOR_RGB2LAB).astype(float)[0, 0]
            if any(np.sqrt(np.sum((lab1 - sl) ** 2)) < dedup_threshold
                   for sl in seen_labs):
                sat = max(r, g, b) - min(r, g, b)
                if sat > 15:
                    print(f"[CAD PDF] Dup colour for '{label}' ({hex_colour}) — "
                          f"keeping with dup tag.")
                    entries.append({
                        "label": label,
                        "rgb": [r, g, b],
                        "hex": hex_colour,
                        "source": src + "_dup",
                        "swatch_bbox": [sx1, sy1, sx2, sy2],
                        "text_bbox": [int(tx0_pt * scale), int(ty0_pt * scale),
                                      int(tx1_pt * scale), int(ty1_pt * scale)],
                    })
                    continue
            seen_labs.append(lab1)

            entries.append({
                "label": label,
                "rgb": [r, g, b],
                "hex": hex_colour,
                "source": src,
                "swatch_bbox": [sx1, sy1, sx2, sy2],
                "text_bbox": [int(tx0_pt * scale), int(ty0_pt * scale),
                              int(tx1_pt * scale), int(ty1_pt * scale)],
            })
            print(f"[CAD PDF] '{label}'  {hex_colour}  RGB({r},{g},{b})  [{src}]")

        if not entries:
            print("[CAD PDF] WARNING: No entries extracted from PDF.")
            return (json.dumps([]), image, 0)

        # ---- Build preview image -------------------------------------------
        # Crop to legend area for a manageable preview
        _tx0_ref = label_lines[0]["bbox"][0]
        lx1 = max(0, int((_tx0_ref - swatch_width_pts - 5) * scale))
        ly1 = max(0, int(ky0 * scale))
        lx2 = min(W_img, int((_tx0_ref + 200) * scale))
        ly2 = min(H_img, int(ky1_actual * scale))

        legend_crop = img_np[ly1:ly2, lx1:lx2].copy()
        # Shift entry bboxes into crop-local coordinates for annotation
        shifted = []
        for e in entries:
            se = dict(e)
            se["swatch_bbox"] = [
                e["swatch_bbox"][0] - lx1, e["swatch_bbox"][1] - ly1,
                e["swatch_bbox"][2] - lx1, e["swatch_bbox"][3] - ly1,
            ]
            se["text_bbox"] = [
                e["text_bbox"][0] - lx1, e["text_bbox"][1] - ly1,
                e["text_bbox"][2] - lx1, e["text_bbox"][3] - ly1,
            ]
            se["rgb"] = tuple(e["rgb"])
            shifted.append(se)

        preview_np = draw_legend_annotations(legend_crop, shifted)
        preview_tensor = numpy_to_tensor(preview_np)

        legend_json = json.dumps(entries, indent=2)
        print(f"[CAD PDF] Extracted {len(entries)} legend entries.")
        return (legend_json, preview_tensor, len(entries))


# ---------------------------------------------------------------------------
# CAD_LegendDisplay
# ---------------------------------------------------------------------------

class CAD_LegendDisplay:
    """
    Formats and prints the legend JSON as a readable table.
    Connect to CAD_SwatchExtractor's legend_json output.
    Output text appears in the ComfyUI node's preview area.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "legend_json": ("STRING", {"multiline": True, "default": "[]"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("formatted_table",)
    OUTPUT_NODE = True
    FUNCTION = "display"
    CATEGORY = "CAD Legend Processor"

    def display(self, legend_json: str):
        try:
            entries = json.loads(legend_json)
        except json.JSONDecodeError:
            msg = "[CAD Legend] Invalid JSON in legend_json input."
            print(msg)
            return {"ui": {"text": [msg]}, "result": (msg,)}

        if not entries:
            msg = "No legend entries found."
            return {"ui": {"text": [msg]}, "result": (msg,)}

        lines = ["Legend Entries", "=" * 50]
        for i, e in enumerate(entries):
            r, g, b = e.get("rgb", [0, 0, 0])
            hex_val = e.get("hex", "#??????")
            label = e.get("label", "unknown")
            lines.append(f"{i+1:2d}. {hex_val}  RGB({r:3d},{g:3d},{b:3d})  {label}")

        lines.append("=" * 50)
        lines.append(f"Total: {len(entries)} entries")
        table = "\n".join(lines)
        print(table)
        return {"ui": {"text": [table]}, "result": (table,)}


# ---------------------------------------------------------------------------
# Node registrations
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "CAD_LegendCropFromFlorence2": CAD_LegendCropFromFlorence2,
    "CAD_ManualLegendCrop":        CAD_ManualLegendCrop,
    "CAD_SwatchExtractor":         CAD_SwatchExtractor,
    "CAD_PDFSwatchExtractor":      CAD_PDFSwatchExtractor,
    "CAD_LegendDisplay":           CAD_LegendDisplay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CAD_LegendCropFromFlorence2": "CAD: Crop Legend (Florence-2)",
    "CAD_ManualLegendCrop":        "CAD: Crop Legend (Manual)",
    "CAD_SwatchExtractor":         "CAD: Swatch Extractor",
    "CAD_PDFSwatchExtractor":      "CAD: PDF Swatch Extractor",
    "CAD_LegendDisplay":           "CAD: Legend Display",
}
