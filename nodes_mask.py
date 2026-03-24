"""
nodes_mask.py — Colour mask generation, selection, and saving nodes.

Nodes:
  CAD_BatchMaskFromLegend  — Generate one mask per legend entry using the full drawing
  CAD_MaskSelector         — Pick a single mask from the batch by label or index
  CAD_SaveLabeledMasks     — Save all masks to disk with label-based filenames
  CAD_MaskPreviewGrid      — Combine all masks into a single preview grid image
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import cv2

from .color_utils import (
    tensor_to_numpy,
    numpy_to_tensor,
    mask_uint8_to_tensor,
    masks_to_batch_tensor,
    create_mask_lab,
    create_mask_hsv,
    apply_morphology,
    rgb_to_hex,
)


def _sanitize_filename(name: str) -> str:
    """Strip characters not safe for filenames."""
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    return safe.strip().strip('.')[:80] or "unnamed"


# ---------------------------------------------------------------------------
# CAD_BatchMaskFromLegend
# ---------------------------------------------------------------------------

class CAD_BatchMaskFromLegend:
    """
    Generates one binary mask per legend entry for the full drawing image.

    Mask method:
      LAB   — CIE76 perceptual distance (recommended for all flat-fill CAD drawings)
      HSV   — Hue-saturation thresholding (useful for highly saturated colours)
      BOTH  — Logical OR of LAB and HSV masks (max recall, use when fills are inconsistent)

    tolerance:
      LAB: perceptual distance (12–25 works well for clean vector PDFs)
      HSV: hue tolerance in OpenCV units (10–20 is typical)

    morphology_kernel:
      3 = gentle clean-up. Use 5–7 for noisier scanned drawings.
      0 = disabled.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "legend_json": ("STRING", {"multiline": True, "default": "[]"}),
                "mask_method": (["LAB", "HSV", "BOTH"], {"default": "LAB"}),
                "tolerance": ("FLOAT", {
                    "default": 20.0,
                    "min": 1.0,
                    "max": 100.0,
                    "step": 0.5,
                    "display": "slider"
                }),
                "morphology_kernel": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 15,
                    "step": 2,
                    "display": "slider"
                }),
                "invert_masks": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MASK", "STRING", "INT")
    RETURN_NAMES = ("masks", "labels_json", "mask_count")
    FUNCTION = "generate_masks"
    CATEGORY = "CAD Legend Processor"

    def generate_masks(self, image, legend_json: str, mask_method: str,
                       tolerance: float, morphology_kernel: int,
                       invert_masks: bool):

        img_np = tensor_to_numpy(image)

        try:
            entries = json.loads(legend_json)
        except json.JSONDecodeError:
            raise ValueError("[CAD Mask] legend_json is not valid JSON. "
                             "Connect CAD_SwatchExtractor's legend_json output.")

        if not entries:
            import torch
            H, W = img_np.shape[:2]
            empty = torch.zeros(1, H, W)
            return (empty, json.dumps([]), 0)

        masks = []
        labels = []

        for entry in entries:
            label = entry.get("label", "unknown")
            rgb = entry.get("rgb", [128, 128, 128])
            r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])

            if mask_method == "LAB":
                mask = create_mask_lab(img_np, (r, g, b), tolerance=tolerance)
            elif mask_method == "HSV":
                mask = create_mask_hsv(img_np, (r, g, b), tolerance=int(tolerance))
            else:  # BOTH
                m_lab = create_mask_lab(img_np, (r, g, b), tolerance=tolerance)
                m_hsv = create_mask_hsv(img_np, (r, g, b), tolerance=int(tolerance))
                mask = cv2.bitwise_or(m_lab, m_hsv)

            if morphology_kernel > 0:
                mask = apply_morphology(mask, kernel_size=morphology_kernel)

            if invert_masks:
                mask = cv2.bitwise_not(mask)

            masks.append(mask)
            labels.append(label)

            coverage = float(np.sum(mask > 0)) / mask.size * 100
            print(f"[CAD Mask] '{label}' ({rgb_to_hex(r,g,b)}) — "
                  f"coverage: {coverage:.1f}%")

        batch_tensor = masks_to_batch_tensor(masks)
        labels_json = json.dumps(labels)

        return (batch_tensor, labels_json, len(masks))


# ---------------------------------------------------------------------------
# CAD_MaskSelector
# ---------------------------------------------------------------------------

class CAD_MaskSelector:
    """
    Select a single mask from the batch produced by CAD_BatchMaskFromLegend.

    Use `label_name` to select by legend text (partial, case-insensitive match).
    Use `index` as fallback when label is empty or not matched.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "labels_json": ("STRING", {"multiline": False, "default": "[]"}),
                "label_name": ("STRING", {
                    "default": "",
                    "placeholder": "Partial label match (case-insensitive)"
                }),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                    "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("MASK", "STRING", "INT")
    RETURN_NAMES = ("mask", "matched_label", "matched_index")
    FUNCTION = "select"
    CATEGORY = "CAD Legend Processor"

    def select(self, masks, labels_json: str, label_name: str, index: int):
        import torch

        try:
            labels = json.loads(labels_json)
        except json.JSONDecodeError:
            labels = []

        total = masks.shape[0]
        chosen_index = index

        if label_name.strip():
            query = label_name.strip().lower()
            for i, lbl in enumerate(labels):
                if query in lbl.lower():
                    chosen_index = i
                    break
            else:
                print(f"[CAD Mask] Label '{label_name}' not found — using index {index}.")

        chosen_index = max(0, min(chosen_index, total - 1))
        selected = masks[chosen_index].unsqueeze(0)  # [1, H, W]
        matched_label = labels[chosen_index] if chosen_index < len(labels) else f"index_{chosen_index}"

        return (selected, matched_label, chosen_index)


# ---------------------------------------------------------------------------
# CAD_SaveLabeledMasks
# ---------------------------------------------------------------------------

class CAD_SaveLabeledMasks:
    """
    Saves each mask from the batch to disk as a PNG file named after its label.

    Files are saved to: ComfyUI/output/{output_subfolder}/{prefix}_{label}.png

    White (255) = masked region, Black (0) = background.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "labels_json": ("STRING", {"multiline": False, "default": "[]"}),
                "output_subfolder": ("STRING", {"default": "cad_masks"}),
                "filename_prefix": ("STRING", {"default": "mask"}),
                "overwrite": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_paths",)
    OUTPUT_NODE = True
    FUNCTION = "save_masks"
    CATEGORY = "CAD Legend Processor"

    def save_masks(self, masks, labels_json: str, output_subfolder: str,
                   filename_prefix: str, overwrite: bool):

        try:
            labels = json.loads(labels_json)
        except json.JSONDecodeError:
            labels = []

        # Resolve output directory relative to ComfyUI's output folder
        base_output = self._find_comfyui_output()
        out_dir = os.path.join(base_output, _sanitize_filename(output_subfolder))
        os.makedirs(out_dir, exist_ok=True)

        saved = []
        total = masks.shape[0]

        for i in range(total):
            label = labels[i] if i < len(labels) else f"entry_{i:03d}"
            safe_label = _sanitize_filename(label)
            prefix = _sanitize_filename(filename_prefix) if filename_prefix else "mask"
            filename = f"{prefix}_{i:03d}_{safe_label}.png"
            filepath = os.path.join(out_dir, filename)

            if not overwrite and os.path.exists(filepath):
                print(f"[CAD Mask] Skipping existing: {filepath}")
                saved.append(filepath)
                continue

            # Convert tensor slice [H,W] float32 → uint8
            mask_np = (masks[i].cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(filepath, mask_np)
            saved.append(filepath)
            print(f"[CAD Mask] Saved: {filepath}")

        result_text = "\n".join(saved) if saved else "No masks saved."
        print(f"[CAD Mask] {len(saved)} mask(s) saved to: {out_dir}")
        return {"ui": {"text": [result_text]}, "result": (result_text,)}

    @staticmethod
    def _find_comfyui_output() -> str:
        """Walk up the file tree from this file to find ComfyUI's output folder."""
        current = Path(__file__).resolve()
        for parent in current.parents:
            candidate = parent / "output"
            if candidate.is_dir():
                return str(candidate)
        # Fallback: use a temp dir
        fallback = Path.home() / "ComfyUI_cad_masks"
        fallback.mkdir(exist_ok=True)
        return str(fallback)


# ---------------------------------------------------------------------------
# CAD_MaskPreviewGrid
# ---------------------------------------------------------------------------

class CAD_MaskPreviewGrid:
    """
    Combines all masks into a single labelled grid image for visual inspection.
    Output is an IMAGE tensor (white-on-black, labelled with legend text).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "labels_json": ("STRING", {"multiline": False, "default": "[]"}),
                "legend_json": ("STRING", {"multiline": True, "default": "[]"}),
                "columns": ("INT", {"default": 4, "min": 1, "max": 12, "step": 1}),
                "thumb_size": ("INT", {"default": 256, "min": 64, "max": 1024, "step": 32}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview_grid",)
    FUNCTION = "make_grid"
    CATEGORY = "CAD Legend Processor"

    def make_grid(self, masks, labels_json: str, legend_json: str,
                  columns: int, thumb_size: int):

        try:
            labels = json.loads(labels_json)
        except json.JSONDecodeError:
            labels = []

        try:
            entries = json.loads(legend_json)
            color_map = {e["label"]: e.get("rgb", [200, 200, 200]) for e in entries}
        except (json.JSONDecodeError, KeyError):
            color_map = {}

        total = masks.shape[0]
        rows = (total + columns - 1) // columns
        pad = 4
        label_h = 22
        cell_h = thumb_size + label_h + pad * 2
        cell_w = thumb_size + pad * 2

        grid = np.zeros((rows * cell_h, columns * cell_w, 3), dtype=np.uint8)

        for i in range(total):
            row, col = divmod(i, columns)
            label = labels[i] if i < len(labels) else f"mask_{i}"
            accent = color_map.get(label, [200, 200, 200])
            accent_bgr = (int(accent[2]), int(accent[1]), int(accent[0]))

            # Resize mask thumbnail
            mask_np = (masks[i].cpu().numpy() * 255).astype(np.uint8)
            thumb = cv2.resize(mask_np, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)

            # Convert mask to RGB: white = masked, tinted background
            thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)

            # Colour-tint the masked regions with the legend colour at 40% opacity
            overlay = np.zeros_like(thumb_rgb)
            overlay[:, :] = accent_bgr
            mask_bool = thumb > 127
            thumb_rgb[mask_bool] = (
                (thumb_rgb[mask_bool].astype(float) * 0.5 +
                 overlay[mask_bool].astype(float) * 0.5).astype(np.uint8)
            )

            # Draw accent border
            cv2.rectangle(thumb_rgb, (0, 0), (thumb_size - 1, thumb_size - 1), accent_bgr, 2)

            # Place thumbnail in grid
            y0 = row * cell_h + pad
            x0 = col * cell_w + pad
            grid[y0: y0 + thumb_size, x0: x0 + thumb_size] = thumb_rgb

            # Label text
            ty = y0 + thumb_size + 14
            tx = x0
            cv2.putText(grid, label[:28], (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2)
            cv2.putText(grid, label[:28], (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)

        return (numpy_to_tensor(grid),)


# ---------------------------------------------------------------------------
# Node registrations
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "CAD_BatchMaskFromLegend": CAD_BatchMaskFromLegend,
    "CAD_MaskSelector":        CAD_MaskSelector,
    "CAD_SaveLabeledMasks":    CAD_SaveLabeledMasks,
    "CAD_MaskPreviewGrid":     CAD_MaskPreviewGrid,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CAD_BatchMaskFromLegend": "CAD: Batch Masks from Legend",
    "CAD_MaskSelector":        "CAD: Select Mask by Label",
    "CAD_SaveLabeledMasks":    "CAD: Save Labeled Masks",
    "CAD_MaskPreviewGrid":     "CAD: Mask Preview Grid",
}
