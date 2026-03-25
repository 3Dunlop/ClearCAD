# ClearCAD

**Automated legend extraction and colour masking for PDF CAD drawings.**

ClearCAD is a [ComfyUI](https://github.com/comfyanonymous/ComfyUI) custom node package that reads PDF CAD orthographic plans, detects the KEY/LEGEND, extracts every colour–label pair, and generates per-colour binary masks across the entire drawing.

Built for UK Traffic Regulation Order (TRO) plans and similar CAD output where coloured zones represent different features (cycleways, bus lanes, footways, tactile paving, road markings, etc.). Works with any PDF CAD drawing that has a colour-coded legend/key.

---

## Overview

```
PDF CAD Drawing
     │
     ▼
Render at 600 DPI ──► Locate KEY heading ──► Extract label positions
     │                                              │
     ▼                                              ▼
Raster image ◄──── Sample swatch pixels ◄──── Swatch coordinates
     │                      │
     ▼                      ▼
Full drawing         Colour–label pairs
     │                      │
     ▼                      ▼
CIE76 LAB matching ──► Binary masks per colour ──► Individual PNGs
                                                    Composite overlay
                                                    Legend JSON
```

Given a multi-page PDF of CAD plans, ClearCAD:

1. **Renders** each PDF page to a high-resolution raster image (600 DPI default)
2. **Locates** the KEY/LEGEND heading using PDF text extraction via [PyMuPDF](https://pymupdf.readthedocs.io/)
3. **Identifies** each label's text position and its adjacent colour swatch
4. **Samples** the actual rendered pixels at each swatch position — not the PDF vector metadata colours (see [Why Not Vector Colours?](#why-not-vector-colours))
5. **Generates** per-colour binary masks using CIE76 perceptual distance in CIELAB colour space
6. **Saves** individual mask PNGs, composite overlays, and structured JSON

---

## Installation

### Prerequisites

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) installed and working
- Python 3.10+
- GPU recommended but not required

### Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/3Dunlop/ClearCAD.git
cd ClearCAD
pip install -r requirements.txt
```

Restart ComfyUI. The nodes appear under the **CAD Legend Processor** category.

### Optional: Florence-2 Fallback

For PDFs without a text layer (pure raster scans), install the Florence-2 vision model nodes:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/kijai/ComfyUI-Florence2.git
```

The Florence-2 model (`microsoft/Florence-2-large`, ~1.5 GB) downloads automatically on first use. This is **completely optional** — the primary extraction path uses PyMuPDF text parsing and requires no external model.

---

## Nodes (10 total)

### PDF Input

| Node | Description |
|------|-------------|
| **CAD: PDF to Image** | Renders a PDF page as a ComfyUI IMAGE tensor at configurable DPI (72–1200). Returns the image, page count, and dimensions. |

### Legend Extraction

| Node | Description |
|------|-------------|
| **CAD: PDF Swatch Extractor** | **Primary path.** Direct PDF text extraction + raster pixel sampling. Finds the KEY heading, locates label text positions, samples rendered pixels at each swatch zone. No external model needed. |
| **CAD: Swatch Extractor** | Florence-2 fallback path. Takes a cropped legend image + Florence-2 OCR JSON, searches for solid-colour swatches adjacent to each text label. |
| **CAD: Crop Legend (Florence-2)** | Crops the legend region using Florence-2 `DENSE_REGION_CAPTION` detection output. |
| **CAD: Crop Legend (Manual)** | Crops the legend region using manually specified pixel coordinates. Useful for consistent drawing templates. |
| **CAD: Legend Display** | Formats legend JSON as a readable table in the ComfyUI node preview. Shows hex values, RGB, and label names. |

### Mask Generation & Output

| Node | Description |
|------|-------------|
| **CAD: Batch Masks from Legend** | Generates one binary mask per legend entry. Supports LAB (CIE76), HSV, or combined matching with configurable tolerance and morphological cleanup. |
| **CAD: Select Mask by Label** | Selects a single mask from the batch by partial label match (case-insensitive) or numeric index. |
| **CAD: Save Labeled Masks** | Saves all masks to disk as PNGs. Filenames are derived from legend labels. Output directory is relative to ComfyUI's output folder. |
| **CAD: Mask Preview Grid** | Combines all masks into a single colour-tinted grid image for quick visual inspection. Each cell is labelled and tinted with the legend colour. |

---

## Workflows

Two ready-to-use workflow JSON files are included in `workflows/`:

### `cad_legend_pdf_direct.json` — Recommended

```
PDF to Image ──► PDF Swatch Extractor ──► Legend Display
                        │
                        ▼
              Batch Masks from Legend ──► Save Labeled Masks
                        │
                        ▼
                  Mask Preview Grid
```

Minimal 6-node workflow. Enter the PDF path and DPI, run the queue. No external models needed.

### `cad_legend_processor.json` — Florence-2 Path

```
PDF to Image ──► Florence-2 Detect ──► Crop Legend ──► Florence-2 OCR ──► Swatch Extractor ──► Batch Masks
```

Full 11-node workflow using Florence-2 for both legend detection and OCR. More flexible for unusual legend layouts but slower.

---

## How It Works

### Why Not Vector Colours?

CAD drawings exported as PDF store vector graphics with colour values attached. The natural approach — read these colours from PDF metadata and search the raster for matches — **fails completely**.

The colours in PDF vector metadata (`page.get_drawings()`) are *authoring* colours from the CAD application. When the PDF renderer converts vectors to pixels, several transformations break the colour correspondence:

| What happens | Example |
|-------------|---------|
| **Colour space conversion** | Vector `#00DD6E` (green) renders as `#D0E080` (yellow-green) |
| **Hatching patterns** | Diagonal lines over fills create a blended raster that matches neither colour |
| **Anti-aliasing** | Edge pixels are blends of the fill and background |
| **CMYK→sRGB conversion** | PDF internal CMYK values map to different sRGB values depending on the rendering intent |

Using vector colours produces masks with 0% coverage — they match nothing in the rendered image.

### The Solution: Raster Pixel Sampling

ClearCAD sidesteps the metadata problem entirely:

1. **Text positions are reliable.** PyMuPDF's `page.get_text("dict")` returns exact bounding boxes (in PDF points) for every text span. The KEY heading and label positions are extracted with sub-point accuracy.

2. **Swatch positions are predictable.** In standard CAD legend layouts, each colour swatch is immediately to the left of its text label. The swatch zone is at `[tx0 - 45pt, tx0 - 3pt]` horizontally, matching the label's vertical extent.

3. **Pixel sampling is ground truth.** By sampling the rendered raster at the swatch position, we get the colour *exactly as it appears in the image*. This is the same image we'll be generating masks from, so the colours match by definition.

The pixel filter chain excludes paper background:

```
Sampled pixels (300-500 per swatch)
     │
     ├─► Primary: (R+G+B < 690) AND (max-min > 10)  →  Coloured pixels
     │
     ├─► Secondary: (R+G+B < 640)  →  Dark grey pixels (hatching)
     │
     └─► Fallback: assign #989898  →  No swatch present (e.g., CARRIAGEWAY)

Filtered pixels → median(R), median(G), median(B) → target colour
```

The median (not mean) is used because it's robust to anti-aliased edge outliers.

### CIE76 Perceptual Colour Matching

Masks are generated using Euclidean distance in [CIELAB colour space](https://en.wikipedia.org/wiki/CIELAB_color_space):

```
ΔE = sqrt((L₁-L₂)² + (a₁-a₂)² + (b₁-b₂)²)
```

LAB is perceptually uniform — equal distances correspond to equal perceived colour differences regardless of hue. This is critical for CAD drawings that use pastels (light blue, light pink, light green) which are numerically close in RGB but visually distinct.

| ΔE Value | Perception |
|----------|-----------|
| 0–1 | Not perceptible |
| 1–2 | Barely perceptible |
| 2–10 | Perceptible at close inspection |
| 10–25 | Clearly different colours |
| 25+ | Distinctly different colours |

Default mask tolerance: **25.0 ΔE** (catches the target colour plus anti-aliased edges).
Default dedup threshold: **12.0 ΔE** (legend entries closer than this are flagged as duplicates).

### Morphological Cleanup

Raw masks have jagged edges and isolated noise. Two cleanup passes:

1. **Close** (dilate → erode): fills small holes inside mask regions
2. **Open** (erode → dilate): removes isolated noise pixels

Both use an elliptical kernel (default size 3). Increase to 5–7 for noisier scanned drawings.

---

## Configuration Reference

### CAD: PDF Swatch Extractor

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `pdf_path` | — | — | Absolute path to the PDF file |
| `page_index` | 0 | 0–9999 | Zero-indexed page number |
| `dpi` | 600 | 72–1200 | Must match the DPI used in CAD: PDF to Image |
| `key_height_pts` | 130 | 20–500 | PDF points below the KEY heading to search for labels |
| `swatch_width_pts` | 45 | 5–150 | PDF points to the left of label text to sample for colour |
| `dedup_threshold` | 12.0 | 1.0–50.0 | CIE76 ΔE threshold for duplicate colour detection |

### CAD: Batch Masks from Legend

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `mask_method` | LAB | LAB/HSV/BOTH | `LAB`: CIE76 perceptual distance (recommended). `HSV`: hue-saturation thresholding (better for vivid colours). `BOTH`: union of both methods (maximum recall). |
| `tolerance` | 20.0 | 1.0–100.0 | Colour distance threshold. LAB: 15–25 typical. HSV: 10–20 typical. |
| `morphology_kernel` | 3 | 0–15 | Cleanup kernel size. 0 = off, 3 = gentle, 5–7 = noisy scans. |
| `invert_masks` | false | — | Invert mask polarity (black ↔ white) |

### CAD: Swatch Extractor (Florence-2 path)

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `swatch_side` | LEFT | LEFT/RIGHT/BOTH | Side to search for colour swatches relative to text |
| `swatch_search_width` | 90 | 10–400 | Pixel width of the swatch search region |
| `variance_threshold` | 18.0 | 1.0–80.0 | Max RGB std-dev for a region to count as "solid colour" |
| `dedup_cie76_threshold` | 8.0 | 1.0–50.0 | CIE76 ΔE threshold for duplicate skipping |

### CAD: PDF to Image

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `pdf_path` | — | — | Absolute path to the PDF file |
| `page_index` | 0 | 0–9999 | Zero-indexed page number |
| `dpi` | 600 | 72–1200 | Render resolution. 300 = quick preview, 600 = production, 1200 = fine annotation text |
| `output_color` | RGB | RGB/RGBA | Output colour mode |

---

## Standalone Test Script

`test_pipeline.py` runs the full extraction and masking pipeline outside ComfyUI, useful for development, debugging, and batch processing:

```bash
cd ComfyUI/custom_nodes/ClearCAD

# Process page 0 (default)
python test_pipeline.py "D:/CAD/your_drawing.pdf"

# Process a specific page
python test_pipeline.py "D:/CAD/your_drawing.pdf" 2
```

Output is saved to `D:/CAD/output/` (configurable in the script):

```
D:/CAD/output/
├── page00_raw.png              # Full-resolution render
├── page00_key_region.png       # Drawing with KEY region highlighted in green
├── page00_key_crop.png         # Cropped KEY area for inspection
├── page00_legend.json          # Structured legend data
├── page00_composite.png        # All masks overlaid on drawing (45% tint)
└── page00_masks/
    ├── 000_PROPOSED_CHANNEL_ALIGNMENT.png
    ├── 001_PROPOSED_VERGE-LANDSCAPING.png
    ├── 002_CARRIAGEWAY.png
    ├── 003_PROPOSED_24-7_BUS_LANE.png
    ├── 004_PROPOSED_FOOTWAY.png
    ├── 005_PROPOSED_CYCLEWAY.png
    ├── 006_PROPOSED_SHARED_FOOTWAY_AND_CYCLEWAY.png
    ├── 007_PROPOSED_TACTILE_PAVING.png
    ├── 008_PROPOSED_ROAD_MARKINGS.png
    ├── 009_DOUBLE_YELLOW_LINE_MARKINGS.png
    └── 010_PROPOSED_TRAFFIC_SIGNALS.png
```

The test script automatically falls back to Florence-2 if PDF text extraction returns no results (e.g., rasterised PDFs with no embedded text layer).

---

## Test Results

Tested against a 4-page UK TRO plan (BSIP Newhaven A259, East Sussex) at 600 DPI. All pages render to 7017 x 4959 px.

### Page 0 — Overview Plan

```
 1. #CDCDCD  RGB(205,205,205)  PROPOSED CHANNEL ALIGNMENT            0.74%
 2. #D2EE81  RGB(210,238,129)  PROPOSED VERGE/LANDSCAPING            0.64%
 3. #989898  RGB(152,152,152)  CARRIAGEWAY                           0.04%  [no swatch]
 4. #FFC0BF  RGB(255,192,191)  PROPOSED 24/7 BUS LANE                0.66%
 5. #FFEFC0  RGB(255,239,192)  PROPOSED FOOTWAY                      0.24%
 6. #81A0FF  RGB(129,160,255)  PROPOSED CYCLEWAY                     0.08%
 7. #C0DFFF  RGB(192,223,255)  PROPOSED SHARED FOOTWAY AND CYCLEWAY  0.65%
 8. #FFC08F  RGB(255,192,143)  PROPOSED TACTILE PAVING               0.17%
 9. #000000  RGB(  0,  0,  0)  PROPOSED ROAD MARKINGS                1.84%
10. #FFC41A  RGB(255,196, 26)  DOUBLE YELLOW LINE MARKINGS           0.01%
11. #CDD9FF  RGB(205,217,255)  PROPOSED TRAFFIC SIGNALS              0.45%  [dup]
```

### Page 1 — Sheet 01 of 03

```
 1. #CDCDCD  PROPOSED CHANNEL ALIGNMENT            2.52%
 2. #D2EE81  PROPOSED VERGE/LANDSCAPING            0.44%
 3. #989898  CARRIAGEWAY                           0.07%
 4. #FFC0BF  PROPOSED 24/7 BUS LANE                3.47%
 5. #FFEFC0  PROPOSED FOOTWAY                      0.68%
 6. #81A0FF  PROPOSED CYCLEWAY                     0.04%
 7. #C0DFFF  PROPOSED SHARED FOOTWAY AND CYCLEWAY  2.33%
 8. #FFC083  PROPOSED TACTILE PAVING               0.00%
 9. #000000  PROPOSED ROAD MARKINGS                2.79%
10. #FFC41A  DOUBLE YELLOW LINE MARKINGS           0.02%
11. #CDD9FF  PROPOSED TRAFFIC SIGNALS              1.96%
```

### Page 2 — Sheet 02 of 03

```
 1. #CDCDCD  PROPOSED CHANNEL ALIGNMENT            1.66%
 2. #D2EE81  PROPOSED VERGE/LANDSCAPING            6.54%
 3. #989898  CARRIAGEWAY                           0.05%
 4. #FFC0BF  PROPOSED 24/7 BUS LANE                2.89%
 5. #FFEFC0  PROPOSED FOOTWAY                      0.84%
 6. #81A0FF  PROPOSED CYCLEWAY                     0.19%
 7. #C0DFFF  PROPOSED SHARED FOOTWAY AND CYCLEWAY  1.54%
 8. #FFC081  PROPOSED TACTILE PAVING               0.04%
 9. #000000  PROPOSED ROAD MARKINGS                2.50%
10. #FFC41A  DOUBLE YELLOW LINE MARKINGS           0.11%
11. #CDD9FF  PROPOSED TRAFFIC SIGNALS              1.19%
```

### Page 3 — Sheet 03 of 03

```
 1. #CDCDCD  PROPOSED CHANNEL ALIGNMENT            0.32%
 2. #D2EE81  PROPOSED VERGE/LANDSCAPING            0.11%
 3. #989898  CARRIAGEWAY                           0.04%
 4. #FFC0BF  PROPOSED 24/7 BUS LANE                0.94%
 5. #FFEFC0  PROPOSED FOOTWAY                      0.42%
 6. #81A0FF  PROPOSED CYCLEWAY                     0.32%
 7. #C0DFFF  PROPOSED SHARED FOOTWAY AND CYCLEWAY  0.32%
 8. #FFC081  PROPOSED TACTILE PAVING               0.01%
 9. #000000  PROPOSED ROAD MARKINGS                2.60%
10. #FFC41A  DOUBLE YELLOW LINE MARKINGS           0.04%
11. #CDD9FF  PROPOSED TRAFFIC SIGNALS              0.20%
```

### Observations

- **Consistent extraction**: All 4 pages detect exactly 11 legend entries with the same hex values (±2 RGB units for TACTILE PAVING due to anti-aliasing)
- **Coverage varies by sheet content**: Page 2 has the most verge/landscaping (6.54%) — that section of the A259 has wide grass medians
- **ROAD MARKINGS (black)** is the noisiest mask at 1.84–2.79% — includes all structural line work
- **CARRIAGEWAY** at 0.04–0.07% indicates the grey fallback colour barely matches anything — expected since there's no dedicated swatch
- **TRAFFIC SIGNALS** flagged as duplicate of SHARED FOOTWAY (both pale blue, ΔE ≈ 9.8) but still gets its own mask entry

---

## Technical Architecture

### File Structure

```
ClearCAD/
├── __init__.py          # Package entry point — aggregates node registrations
├── color_utils.py       # Colour math, tensor helpers, mask generation, swatch detection
├── nodes_pdf.py         # CAD_PDFToImage node
├── nodes_legend.py      # 5 legend extraction nodes (PDF + Florence-2 paths)
├── nodes_mask.py        # 4 mask generation and output nodes
├── test_pipeline.py     # Standalone test script (runs outside ComfyUI)
├── requirements.txt     # Python dependencies
├── workflows/
│   ├── cad_legend_pdf_direct.json    # Recommended workflow (no Florence-2)
│   └── cad_legend_processor.json     # Full workflow with Florence-2
├── LICENSE              # MIT
└── README.md
```

### Key Modules

**`color_utils.py`** — Shared utilities used by all nodes:
- `tensor_to_numpy()` / `numpy_to_tensor()` — convert between ComfyUI's `[B,H,W,C]` float32 tensors and `[H,W,3]` uint8 NumPy arrays
- `create_mask_lab()` — generates a binary mask via CIE76 distance in LAB space
- `create_mask_hsv()` — generates a binary mask via HSV range thresholding (handles hue wrap-around for reds)
- `apply_morphology()` — close + open cleanup with elliptical kernel
- `find_swatch_left_of_text()` / `find_swatch_right_of_text()` — searches for solid-colour patches adjacent to text bounding boxes
- `parse_florence2_detection()` / `parse_florence2_ocr_with_region()` — parsers for Florence-2 JSON output formats
- `draw_legend_annotations()` — draws coloured bounding boxes and hex labels on an image for preview

**`nodes_pdf.py`** — PDF rendering via PyMuPDF. Handles DPI scaling (`zoom = dpi / 72.0`), colour space selection, and conversion to ComfyUI tensor format.

**`nodes_legend.py`** — Legend extraction with two paths:
- `CAD_PDFSwatchExtractor`: Opens the PDF, extracts text with `page.get_text("dict")`, finds the KEY heading, filters label lines by position, samples raster pixels at swatch coordinates, deduplicates via CIE76 distance
- `CAD_SwatchExtractor`: Parses Florence-2 OCR JSON, searches for solid-colour regions adjacent to each text bounding box

**`nodes_mask.py`** — Mask generation and output:
- `CAD_BatchMaskFromLegend`: Iterates legend entries, generates masks via LAB/HSV/both, applies morphological cleanup, returns stacked tensor
- `CAD_MaskSelector`: Partial string matching on labels for single-mask selection
- `CAD_SaveLabeledMasks`: Writes mask PNGs with sanitised label-based filenames
- `CAD_MaskPreviewGrid`: Renders a labelled grid with colour-tinted mask thumbnails

### Coordinate System

ClearCAD operates across two coordinate systems:

| System | Units | Used for |
|--------|-------|----------|
| **PDF points** | 1 pt = 1/72 inch | Text bounding boxes, swatch positions, KEY heading location |
| **Pixels** | Depends on DPI | Raster image, mask arrays, output PNGs |

Conversion: `pixel = pdf_points × (dpi / 72.0)`

At 600 DPI, the scale factor is 8.333. A standard A1 CAD sheet (841 × 594 mm) renders to approximately 19,843 × 14,003 pixels.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| [PyMuPDF](https://pymupdf.readthedocs.io/) | ≥ 1.23.0 | PDF rendering and text extraction |
| [OpenCV](https://opencv.org/) (headless) | ≥ 4.8.0 | Colour space conversion (RGB↔LAB↔HSV), morphological operations |
| [NumPy](https://numpy.org/) | ≥ 1.24.0 | Array operations, pixel filtering, median computation |
| [Pillow](https://pillow.readthedocs.io/) | ≥ 9.0.0 | Image I/O for Florence-2 path |
| [PyTorch](https://pytorch.org/) | — | Tensor operations (provided by ComfyUI) |

Optional (Florence-2 fallback only):
| Package | Purpose |
|---------|---------|
| [transformers](https://huggingface.co/docs/transformers/) | Tokenizer and image processor |
| [safetensors](https://github.com/huggingface/safetensors) | Model weight loading |
| [accelerate](https://huggingface.co/docs/accelerate/) | Device placement |

---

## Known Limitations

- **Black (`#000000`) masks are noisy (1.84–2.79% coverage).** PROPOSED ROAD MARKINGS samples as black, which also matches all structural line work, text, boundary outlines, title blocks, and the drawing frame. This is inherent to black being used as both a map symbol and a structural element in CAD output. Future versions will use PDF vector stroke width to filter structural lines.
- **CARRIAGEWAY has no swatch (0.04–0.07% coverage).** The KEY area for CARRIAGEWAY contains no coloured fill — it falls back to `#989898` grey, which matches almost nothing in the drawing. This is correct behaviour (CARRIAGEWAY is typically uncoloured in TRO plans), but the resulting mask is not useful for texturing.
- **Very similar colours flagged as duplicates.** PROPOSED TRAFFIC SIGNALS (#CDD9FF, pale blue) and PROPOSED SHARED FOOTWAY AND CYCLEWAY (#C0DFFF, pale blue) have a CIE76 distance of ~9.8 ΔE, below the 12.0 default threshold. Both entries appear in the legend JSON but TRAFFIC SIGNALS is tagged `_dup`. Increase `dedup_threshold` to 15+ to separate them, at the risk of allowing true duplicates through.
- **Body text near KEY can be captured.** Text lines like "FOR CONTINUATION SEE VIEWPORT 1B" that appear within the KEY Y-range are now filtered by keyword blocklist (CONTINUATION, VIEWPORT, SEE SHEET, etc.), but unusual annotation text may still slip through.
- **No text layer = no PDF extraction.** Pure raster scans without embedded text require the Florence-2 fallback path.
- **Legend heading required.** The PDF extractor searches for "KEY", "LEGEND", or similar headings. Non-standard headings fall back to inferring legend position from "PROPOSED" text lines.
- **Single-page legend assumption.** Each page re-extracts its own legend independently. Multi-page plans that share a single KEY on page 0 don't yet propagate it to subsequent pages.
- **TACTILE PAVING colour varies slightly between pages** (±12 RGB units) due to anti-aliasing of the hatched swatch pattern. This doesn't affect mask quality since the 25 ΔE tolerance absorbs the variation.

---

## Roadmap

- [ ] Georeferencing: extract OS grid references from PDF text, compute affine transform to British National Grid (EPSG:27700)
- [ ] Aerial imagery overlay: fetch satellite/aerial tiles matching the drawing extent and composite masks with real-world textures
- [ ] Multi-page legend propagation: extract legend once and apply to all pages
- [ ] Black mask refinement: use PDF vector stroke width to distinguish road markings from structural line work
- [ ] Batch processing: process entire PDFs (all pages) in a single run

---

## License

MIT — see [LICENSE](LICENSE).
