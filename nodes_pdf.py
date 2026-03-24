"""
nodes_pdf.py — CAD_PDFToImage node
Converts a PDF page to a high-resolution ComfyUI image tensor using PyMuPDF.
"""

import os
import numpy as np
from .color_utils import numpy_to_tensor

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


class CAD_PDFToImage:
    """
    Load a single page from a PDF file and output it as a ComfyUI IMAGE tensor.

    Recommended DPI for CAD drawings: 600 (balances text legibility vs file size).
    Use 300 for a quick preview, 1200 for very fine annotation text.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pdf_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Absolute path to PDF file"
                }),
                "page_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                    "step": 1,
                    "display": "number"
                }),
                "dpi": ("INT", {
                    "default": 600,
                    "min": 72,
                    "max": 1200,
                    "step": 50,
                    "display": "slider"
                }),
                "output_color": (["RGB", "RGBA"], {"default": "RGB"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "page_count", "image_width", "image_height")
    FUNCTION = "load_pdf_page"
    CATEGORY = "CAD Legend Processor"

    def load_pdf_page(self, pdf_path: str, page_index: int, dpi: int,
                      output_color: str):
        if not PYMUPDF_AVAILABLE:
            raise ImportError(
                "PyMuPDF is not installed. Run: pip install PyMuPDF\n"
                "Or install via requirements.txt in comfyui_cad_legend."
            )

        pdf_path = pdf_path.strip().strip('"').strip("'")
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF not found: '{pdf_path}'")

        doc = fitz.open(pdf_path)
        page_count = doc.page_count

        if page_index >= page_count:
            doc.close()
            raise ValueError(
                f"page_index {page_index} out of range — "
                f"PDF has {page_count} page(s) (0-indexed)."
            )

        page = doc[page_index]

        # Render at requested DPI (default PDF DPI is 72)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        colorspace = fitz.csRGB if output_color == "RGB" else fitz.csRGBA
        pix = page.get_pixmap(matrix=mat, colorspace=colorspace, alpha=(output_color == "RGBA"))
        doc.close()

        # Convert to numpy — PyMuPDF gives us bytes in RGB(A) order
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )

        # Ensure 3-channel RGB
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        elif arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)

        h, w = arr.shape[:2]
        tensor = numpy_to_tensor(arr)

        return (tensor, page_count, w, h)


NODE_CLASS_MAPPINGS = {
    "CAD_PDFToImage": CAD_PDFToImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CAD_PDFToImage": "CAD: PDF to Image",
}
