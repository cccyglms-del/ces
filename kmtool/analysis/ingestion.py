import io
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import fitz
except ImportError:  # pragma: no cover - optional dependency warning handled upstream
    fitz = None


KM_PAGE_TERMS = (
    "kaplan",
    "survival",
    "overall survival",
    "progression-free",
    "number at risk",
    "months",
    "hazard ratio",
    "log-rank",
)


def load_image_bytes(file_bytes):
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


def pil_to_array(image):
    return np.array(image.convert("RGB"))


def array_to_pil(image_array):
    return Image.fromarray(image_array.astype("uint8"), mode="RGB")


def crop_image(image, bbox):
    if bbox is None:
        return image
    left, top, right, bottom = [int(value) for value in bbox]
    left = max(left, 0)
    top = max(top, 0)
    right = min(right, image.width)
    bottom = min(bottom, image.height)
    if right <= left or bottom <= top:
        return image
    return image.crop((left, top, right, bottom))


def render_pdf_pages(pdf_bytes, dpi=180):
    if fitz is None:
        raise RuntimeError("PyMuPDF is not installed. Install requirements to enable PDF support.")
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    page_texts = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    for index in range(len(document)):
        page = document[index]
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
        pages.append(image)
        page_texts.append(page.get_text("text"))
    return pages, page_texts


def score_pdf_page(image, page_text):
    text = (page_text or "").lower()
    keyword_score = sum(4 for term in KM_PAGE_TERMS if term in text)

    image_np = pil_to_array(image)
    grayscale = np.mean(image_np, axis=2)
    non_white = np.mean(grayscale < 245)
    horizontal_variation = np.mean(np.abs(np.diff(grayscale, axis=1)) > 12)
    vertical_variation = np.mean(np.abs(np.diff(grayscale, axis=0)) > 12)

    visual_score = (non_white * 20.0) + (horizontal_variation * 20.0) + (vertical_variation * 20.0)
    total = keyword_score + visual_score
    return {
        "score": round(float(total), 3),
        "keyword_score": round(float(keyword_score), 3),
        "visual_score": round(float(visual_score), 3),
        "text_excerpt": " ".join(text.split())[:220],
    }


def rank_pdf_pages(pages, page_texts):
    ranked = []
    for index, (image, text) in enumerate(zip(pages, page_texts)):
        scores = score_pdf_page(image, text)
        ranked.append(
            {
                "page_index": index,
                "image": image,
                "page_text": text,
                "score": scores["score"],
                "keyword_score": scores["keyword_score"],
                "visual_score": scores["visual_score"],
                "text_excerpt": scores["text_excerpt"],
            }
        )
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked


def detect_candidate_plot_bbox(image):
    image_np = pil_to_array(image)
    grayscale = np.mean(image_np, axis=2)
    content_mask = grayscale < 245
    row_density = np.mean(content_mask, axis=1)
    col_density = np.mean(content_mask, axis=0)

    plot_bbox = _dense_grid_bbox(row_density, col_density, image.width, image.height)
    if plot_bbox is not None:
        return plot_bbox

    rows = np.where(row_density > 0.05)[0]
    cols = np.where(col_density > 0.05)[0]
    if len(rows) == 0 or len(cols) == 0:
        return (0, 0, image.width, image.height)

    top = int(max(rows[0] - 12, 0))
    bottom = int(min(rows[-1] + 12, image.height))
    left = int(max(cols[0] - 12, 0))
    right = int(min(cols[-1] + 12, image.width))
    return (left, top, right, bottom)


def _dense_grid_bbox(row_density, col_density, width, height):
    upper_row_limit = max(1, int(height * 0.85))
    strong_row_threshold = max(0.22, float(np.quantile(row_density[:upper_row_limit], 0.995)) * 0.55)
    strong_col_threshold = max(0.18, float(np.quantile(col_density, 0.995)) * 0.55)

    strong_rows = np.where(row_density[:upper_row_limit] >= strong_row_threshold)[0]
    strong_cols = np.where(col_density >= strong_col_threshold)[0]
    if len(strong_rows) < 2 or len(strong_cols) < 2:
        return None

    top = max(0, int(strong_rows[0] - height * 0.01))
    bottom = min(height, int(strong_rows[-1] + height * 0.01))
    left = max(0, int(strong_cols[0] - width * 0.01))
    right = min(width, int(strong_cols[-1] + width * 0.01))

    if (bottom - top) < int(height * 0.20) or (right - left) < int(width * 0.25):
        return None
    return (left, top, right, bottom)


def detect_candidate_risk_bbox(image, plot_bbox):
    image_np = pil_to_array(image)
    grayscale = np.mean(image_np, axis=2)
    content_mask = grayscale < 245

    start_row = min(image.height - 1, int(plot_bbox[3] + image.height * 0.01))
    end_row = min(image.height, int(plot_bbox[3] + image.height * 0.28))
    if end_row <= start_row:
        return None

    lower_mask = content_mask[start_row:end_row]
    row_density = np.mean(lower_mask, axis=1)
    content_rows = np.where(row_density > 0.01)[0]
    if len(content_rows) == 0:
        return None

    top = max(0, int(start_row + content_rows[0] - 4))
    bottom = min(image.height, int(start_row + content_rows[-1] + 8))

    region = content_mask[top:bottom]
    col_density = np.mean(region, axis=0)
    content_cols = np.where(col_density > 0.01)[0]
    if len(content_cols) == 0:
        return (0, top, image.width, bottom)

    left = max(0, int(content_cols[0] - 6))
    right = min(image.width, int(content_cols[-1] + 6))
    return (left, top, right, bottom)


def normalize_uploaded_input(uploaded_file):
    file_name = uploaded_file.name
    file_bytes = uploaded_file.getvalue()
    if file_name.lower().endswith(".pdf"):
        return {"source_type": "pdf", "file_name": file_name, "file_bytes": file_bytes}
    return {"source_type": "image", "file_name": file_name, "file_bytes": file_bytes}


def build_manual_crop(width, height, crop_percentages):
    if not crop_percentages:
        return None
    left = int(width * crop_percentages[0] / 100.0)
    top = int(height * crop_percentages[1] / 100.0)
    right = int(width * crop_percentages[2] / 100.0)
    bottom = int(height * crop_percentages[3] / 100.0)
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)
