"""
color_utils.py — Shared color math and mask generation utilities
for the ComfyUI CAD Legend Processor.
"""

import json
import numpy as np
import cv2
from typing import Optional, Tuple, List, Dict, Any


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
RGB = Tuple[int, int, int]
BBox = Tuple[int, int, int, int]  # x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Tensor ↔ NumPy helpers (ComfyUI uses [B,H,W,C] float32 tensors)
# ---------------------------------------------------------------------------

def tensor_to_numpy(tensor) -> np.ndarray:
    """[B,H,W,C] float32 tensor → [H,W,3] uint8 RGB array."""
    import torch
    if isinstance(tensor, torch.Tensor):
        arr = tensor[0].cpu().numpy()
    else:
        arr = np.array(tensor[0])
    return np.clip(arr * 255, 0, 255).astype(np.uint8)


def numpy_to_tensor(arr: np.ndarray):
    """[H,W,3] uint8 RGB array → [1,H,W,C] float32 tensor."""
    import torch
    return torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0)


def mask_uint8_to_tensor(mask: np.ndarray):
    """[H,W] uint8 (0/255) mask → [1,H,W] float32 (0/1) tensor."""
    import torch
    return torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0)


def masks_to_batch_tensor(masks: List[np.ndarray]):
    """List of [H,W] uint8 masks → [N,H,W] float32 tensor."""
    import torch
    tensors = [mask_uint8_to_tensor(m)[0] for m in masks]  # each [H,W]
    return torch.stack(tensors, dim=0)


# ---------------------------------------------------------------------------
# Basic color conversions
# ---------------------------------------------------------------------------

def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02X}{g:02X}{b:02X}"


def hex_to_rgb(hex_color: str) -> RGB:
    h = hex_color.lstrip('#')
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def rgb_to_hsv(r: int, g: int, b: int) -> Tuple[int, int, int]:
    """RGB (0-255) → OpenCV HSV (H:0-179, S:0-255, V:0-255)."""
    bgr = np.uint8([[[b, g, r]]])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0]
    return (int(hsv[0]), int(hsv[1]), int(hsv[2]))


def cie76_distance_scalar(rgb1: RGB, rgb2: RGB) -> float:
    """Perceptual color distance between two RGB colours using CIE76 in LAB space."""
    c1 = np.uint8([[list(rgb1)]])
    c2 = np.uint8([[list(rgb2)]])
    lab1 = cv2.cvtColor(c1, cv2.COLOR_RGB2LAB).astype(float)[0, 0]
    lab2 = cv2.cvtColor(c2, cv2.COLOR_RGB2LAB).astype(float)[0, 0]
    return float(np.sqrt(np.sum((lab1 - lab2) ** 2)))


# ---------------------------------------------------------------------------
# Color sampling
# ---------------------------------------------------------------------------

def sample_dominant_color(image_rgb: np.ndarray, bbox: BBox,
                           edge_shrink: float = 0.15) -> RGB:
    """
    Sample the dominant (median) colour from a bbox region.
    Shrinks inward by `edge_shrink` fraction to avoid anti-aliased edges.
    """
    x1, y1, x2, y2 = bbox
    h_img, w_img = image_rgb.shape[:2]

    # Clamp to image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)

    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return (128, 128, 128)

    px = max(1, int(w * edge_shrink))
    py = max(1, int(h * edge_shrink))

    region = image_rgb[y1 + py: y2 - py, x1 + px: x2 - px]
    if region.size == 0:
        region = image_rgb[y1:y2, x1:x2]

    r = int(np.median(region[:, :, 0]))
    g = int(np.median(region[:, :, 1]))
    b = int(np.median(region[:, :, 2]))
    return (r, g, b)


def region_color_variance(image_rgb: np.ndarray, bbox: BBox) -> float:
    """Return mean std-dev of RGB channels in a bbox (lower = more uniform)."""
    x1, y1, x2, y2 = bbox
    region = image_rgb[y1:y2, x1:x2]
    if region.size == 0:
        return 999.0
    return float(np.mean([np.std(region[:, :, c]) for c in range(3)]))


# ---------------------------------------------------------------------------
# Swatch detection: find colour patch to the left of a text bounding box
# ---------------------------------------------------------------------------

def find_swatch_left_of_text(
    image_rgb: np.ndarray,
    text_bbox: BBox,
    search_width: int = 100,
    gap: int = 4,
    variance_threshold: float = 18.0,
) -> Optional[Tuple[BBox, RGB]]:
    """
    Search for a solid-colour swatch immediately to the left of a text bbox.

    Returns (swatch_bbox, (r, g, b)) or None if no solid patch found.
    """
    h_img, w_img = image_rgb.shape[:2]
    tx1, ty1, tx2, ty2 = text_bbox
    text_h = max(1, ty2 - ty1)

    # Candidate region: to the left of the text
    sx2 = max(0, tx1 - gap)
    sx1 = max(0, sx2 - search_width)
    sy1 = max(0, ty1 - 2)
    sy2 = min(h_img, ty2 + 2)

    if sx2 <= sx1 or sy2 <= sy1:
        return None

    region = image_rgb[sy1:sy2, sx1:sx2]
    variance = float(np.mean([np.std(region[:, :, c]) for c in range(3)]))

    if variance <= variance_threshold:
        color = sample_dominant_color(image_rgb, (sx1, sy1, sx2, sy2))
        return (sx1, sy1, sx2, sy2), color

    # Fallback: try a tighter centre strip
    pad = max(2, text_h // 3)
    cy = (sy1 + sy2) // 2
    fx1 = max(0, sx2 - pad * 3)
    fy1, fy2 = max(0, cy - pad), min(h_img, cy + pad)
    region2 = image_rgb[fy1:fy2, fx1:sx2]
    if region2.size > 0:
        variance2 = float(np.mean([np.std(region2[:, :, c]) for c in range(3)]))
        if variance2 <= variance_threshold:
            color = sample_dominant_color(image_rgb, (fx1, fy1, sx2, fy2))
            return (fx1, fy1, sx2, fy2), color

    return None


def find_swatch_right_of_text(
    image_rgb: np.ndarray,
    text_bbox: BBox,
    search_width: int = 100,
    gap: int = 4,
    variance_threshold: float = 18.0,
) -> Optional[Tuple[BBox, RGB]]:
    """Search for a solid-colour swatch immediately to the right of a text bbox."""
    h_img, w_img = image_rgb.shape[:2]
    tx1, ty1, tx2, ty2 = text_bbox

    sx1 = min(w_img, tx2 + gap)
    sx2 = min(w_img, sx1 + search_width)
    sy1 = max(0, ty1 - 2)
    sy2 = min(h_img, ty2 + 2)

    if sx2 <= sx1 or sy2 <= sy1:
        return None

    region = image_rgb[sy1:sy2, sx1:sx2]
    variance = float(np.mean([np.std(region[:, :, c]) for c in range(3)]))

    if variance <= variance_threshold:
        color = sample_dominant_color(image_rgb, (sx1, sy1, sx2, sy2))
        return (sx1, sy1, sx2, sy2), color

    return None


# ---------------------------------------------------------------------------
# Florence-2 output parsers
# ---------------------------------------------------------------------------

def parse_florence2_detection(result_json: str, label_filter: str = "") -> List[BBox]:
    """
    Parse Florence-2 detection/region JSON output.
    Handles: OPEN_VOCABULARY_DETECTION, DENSE_REGION_CAPTION, REGION_PROPOSAL, OD.

    Returns list of (x1,y1,x2,y2) bounding boxes.
    If label_filter is set, only return boxes whose label contains it (case-insensitive).
    Pass label_filter="" to return all boxes regardless of label.
    """
    try:
        data = json.loads(result_json)
    except (json.JSONDecodeError, TypeError):
        return []

    # Unwrap outer key — Florence-2 always wraps in the task token key
    if isinstance(data, dict):
        for key in data:
            ku = key.upper()
            if any(x in ku for x in ("DETECTION", "GROUNDING", "CAPTION", "PROPOSAL", "<OD>")):
                data = data[key]
                break

    bboxes = data.get("bboxes", [])
    # Try multiple label key names used by different Florence-2 tasks
    labels = (data.get("bboxes_labels")
              or data.get("labels")
              or [""] * len(bboxes))

    results = []
    for bbox, label in zip(bboxes, labels):
        if label_filter and label_filter.lower() not in label.lower():
            continue
        if len(bbox) == 4:
            results.append((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
    return results


def parse_florence2_ocr_with_region(result_json: str, image_wh: Tuple[int, int]) -> List[Dict[str, Any]]:
    """
    Parse Florence-2 OCR_WITH_REGION JSON.
    Returns list of {"text": str, "bbox": (x1,y1,x2,y2)}.

    Florence-2 returns quad_boxes as 8 values [x1,y1,x2,y2,x3,y3,x4,y4] in pixel coords.
    We convert to axis-aligned bbox.
    """
    W, H = image_wh
    try:
        data = json.loads(result_json)
    except (json.JSONDecodeError, TypeError):
        return []

    if isinstance(data, dict):
        for key in data:
            if "OCR" in key.upper():
                data = data[key]
                break

    quad_boxes = data.get("quad_boxes", [])
    labels = data.get("labels", [])

    results = []
    for quad, label in zip(quad_boxes, labels):
        label = label.strip()
        if not label:
            continue

        if len(quad) == 8:
            xs = [quad[0], quad[2], quad[4], quad[6]]
            ys = [quad[1], quad[3], quad[5], quad[7]]
        elif len(quad) == 4:
            xs = [quad[0], quad[2]]
            ys = [quad[1], quad[3]]
        else:
            continue

        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))

        # Some Florence builds return normalised [0,999] — detect and scale
        if x2 <= 1000 and y2 <= 1000 and (x2 < W * 0.1 or y2 < H * 0.1):
            x1 = int(x1 * W / 1000)
            y1 = int(y1 * H / 1000)
            x2 = int(x2 * W / 1000)
            y2 = int(y2 * H / 1000)

        results.append({"text": label, "bbox": (x1, y1, x2, y2)})

    return results


# ---------------------------------------------------------------------------
# Colour mask generation
# ---------------------------------------------------------------------------

def create_mask_lab(image_rgb: np.ndarray, target_rgb: RGB,
                    tolerance: float = 20.0) -> np.ndarray:
    """
    Binary mask via CIE76 perceptual distance in LAB space.
    Best for flat fills — handles neutrals, pastels, and saturated colours equally well.

    Returns uint8 [H,W] mask: 255 = match.
    """
    img_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    target_arr = np.uint8([[list(target_rgb)]])
    target_lab = cv2.cvtColor(target_arr, cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]

    diff = img_lab - target_lab
    distance = np.sqrt(np.sum(diff ** 2, axis=2))
    return ((distance <= tolerance) * 255).astype(np.uint8)


def create_mask_hsv(image_rgb: np.ndarray, target_rgb: RGB,
                    tolerance: int = 15) -> np.ndarray:
    """
    Binary mask via HSV range thresholding.
    Better for highly saturated colours; handles hue wrap-around for reds.

    Returns uint8 [H,W] mask: 255 = match.
    """
    img_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h_t, s_t, v_t = rgb_to_hsv(*target_rgb)

    h_tol = tolerance if s_t > 30 else 179
    s_tol = min(80, tolerance * 3)
    v_tol = min(80, tolerance * 3)

    lo = np.array([max(0, h_t - h_tol), max(0, s_t - s_tol), max(0, v_t - v_tol)])
    hi = np.array([min(179, h_t + h_tol), min(255, s_t + s_tol), min(255, v_t + v_tol)])
    mask = cv2.inRange(img_hsv, lo, hi)

    # Red wraps around hue=0/179
    if h_t - h_tol < 0:
        lo2 = np.array([max(0, 179 + h_t - h_tol), lo[1], lo[2]])
        hi2 = np.array([179, hi[1], hi[2]])
        mask = cv2.bitwise_or(mask, cv2.inRange(img_hsv, lo2, hi2))
    elif h_t + h_tol > 179:
        lo2 = np.array([0, lo[1], lo[2]])
        hi2 = np.array([h_t + h_tol - 179, hi[1], hi[2]])
        mask = cv2.bitwise_or(mask, cv2.inRange(img_hsv, lo2, hi2))

    return mask


def apply_morphology(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Clean mask with morphological close (fill gaps) then open (remove noise).
    kernel_size=3 is gentle; use 5-7 for noisier inputs.
    """
    if kernel_size < 2:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask


# ---------------------------------------------------------------------------
# Legend annotation drawing (for preview output)
# ---------------------------------------------------------------------------

def draw_legend_annotations(image_rgb: np.ndarray,
                             entries: List[Dict[str, Any]]) -> np.ndarray:
    """
    Draw coloured bounding boxes and labels on a copy of the image.
    entries: list of {"label", "rgb", "swatch_bbox", "text_bbox"}
    """
    out = image_rgb.copy()
    for entry in entries:
        label = entry.get("label", "?")
        color = entry.get("rgb", (255, 0, 0))
        bgr = (color[2], color[1], color[0])

        # Draw swatch box
        if "swatch_bbox" in entry:
            x1, y1, x2, y2 = entry["swatch_bbox"]
            cv2.rectangle(out, (x1, y1), (x2, y2), bgr, 2)

        # Draw text bbox
        if "text_bbox" in entry:
            tx1, ty1, tx2, ty2 = entry["text_bbox"]
            cv2.rectangle(out, (tx1, ty1), (tx2, ty2), (0, 200, 0), 1)

        # Print hex label
        if "swatch_bbox" in entry:
            x1, y1 = entry["swatch_bbox"][0], entry["swatch_bbox"][1]
            hex_str = rgb_to_hex(*color)
            cv2.putText(out, hex_str, (x1, max(0, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2)
            cv2.putText(out, hex_str, (x1, max(0, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    return out
