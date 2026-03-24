"""
comfyui_cad_legend — CAD Orthographic Plan Legend Processor
============================================================
Reads PDF CAD drawings, detects the key/legend, extracts colour-label pairs,
and generates binary masks for each colour in the drawing.

Node categories:
  CAD Legend Processor/
    ├─ CAD: PDF to Image
    ├─ CAD: PDF Swatch Extractor   ← direct PDF+raster (no Florence-2)
    ├─ CAD: Crop Legend (Florence-2)
    ├─ CAD: Crop Legend (Manual)
    ├─ CAD: Swatch Extractor
    ├─ CAD: Legend Display
    ├─ CAD: Batch Masks from Legend
    ├─ CAD: Select Mask by Label
    ├─ CAD: Save Labeled Masks
    └─ CAD: Mask Preview Grid

Required external custom nodes:
  - kijai/ComfyUI-Florence2  (legend detection + OCR)

Optional:
  - ComfyUI-Documents        (alternative PDF loader)
  - ComfyUI-Impact-Pack      (additional mask utilities)
"""

from .nodes_pdf    import NODE_CLASS_MAPPINGS as _PDF_NODES,    NODE_DISPLAY_NAME_MAPPINGS as _PDF_NAMES
from .nodes_legend import NODE_CLASS_MAPPINGS as _LEGEND_NODES, NODE_DISPLAY_NAME_MAPPINGS as _LEGEND_NAMES
from .nodes_mask   import NODE_CLASS_MAPPINGS as _MASK_NODES,   NODE_DISPLAY_NAME_MAPPINGS as _MASK_NAMES

NODE_CLASS_MAPPINGS = {
    **_PDF_NODES,
    **_LEGEND_NODES,
    **_MASK_NODES,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **_PDF_NAMES,
    **_LEGEND_NAMES,
    **_MASK_NAMES,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

WEB_DIRECTORY = None
