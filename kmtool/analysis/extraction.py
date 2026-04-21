import math
from collections import defaultdict

import numpy as np
from PIL import Image

from kmtool.analysis.ingestion import crop_image, detect_candidate_plot_bbox, detect_candidate_risk_bbox
from kmtool.analysis.llm import call_multimodal_chart_review
from kmtool.analysis.ocr import guess_risk_table_rows, infer_axis_bounds, run_ocr
from kmtool.models import ArmMapping, AxisBounds, CurveSeries, ExtractionReview

try:
    import cv2
except ImportError:  # pragma: no cover - dependency managed by requirements
    cv2 = None


def _ensure_cv2():
    if cv2 is None:
        raise RuntimeError("opencv-python-headless is required for curve extraction.")


def preprocess_image(image):
    _ensure_cv2()
    image_np = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    denoised = cv2.bilateralFilter(bgr, 7, 50, 50)
    enhanced = cv2.convertScaleAbs(denoised, alpha=1.15, beta=0)
    return cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)


def resolve_plot_region(image, manual_crop=None):
    if manual_crop is not None:
        return crop_image(image, manual_crop), manual_crop
    bbox = detect_candidate_plot_bbox(image)
    return crop_image(image, bbox), bbox


def _suppress_border_axes(mask):
    cleaned = mask.copy()
    binary = mask > 0
    row_density = np.mean(binary, axis=1)
    col_density = np.mean(binary, axis=0)
    height, width = mask.shape

    if row_density.size:
        row_threshold = max(0.55, float(np.quantile(row_density, 0.995)) * 0.70)
        for row_index, density in enumerate(row_density):
            if density >= row_threshold and (row_index < height * 0.12 or row_index > height * 0.88):
                cleaned[max(0, row_index - 1) : min(height, row_index + 2), :] = 0

    if col_density.size:
        col_threshold = max(0.55, float(np.quantile(col_density, 0.995)) * 0.70)
        for col_index, density in enumerate(col_density):
            if density >= col_threshold and (col_index < width * 0.12 or col_index > width * 0.88):
                cleaned[:, max(0, col_index - 1) : min(width, col_index + 2)] = 0

    return cleaned


def _filter_components(mask, min_pixels=40):
    components, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    for index in range(1, components):
        area = stats[index, cv2.CC_STAT_AREA]
        if area < min_pixels:
            continue
        component_mask = (labels == index).astype("uint8") * 255
        filtered = cv2.bitwise_or(filtered, component_mask)
    return filtered


def _mask_span_stats(mask):
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return {
            "x_span_ratio": 0.0,
            "y_span_ratio": 0.0,
            "covered_x_ratio": 0.0,
            "x_min": 0,
            "x_max": 0,
            "y_min": 0,
            "y_max": 0,
        }
    height, width = mask.shape
    unique_x = np.unique(xs)
    return {
        "x_span_ratio": float(xs.max() - xs.min() + 1) / float(max(width, 1)),
        "y_span_ratio": float(ys.max() - ys.min() + 1) / float(max(height, 1)),
        "covered_x_ratio": float(unique_x.size) / float(max(width, 1)),
        "x_min": int(xs.min()),
        "x_max": int(xs.max()),
        "y_min": int(ys.min()),
        "y_max": int(ys.max()),
    }


def _is_valid_curve_mask(mask):
    stats = _mask_span_stats(mask)
    if stats["x_span_ratio"] < 0.35:
        return False
    if stats["covered_x_ratio"] < 0.20:
        return False
    if stats["y_span_ratio"] < 0.03:
        return False
    return True


def _trace_curve(mask):
    height, width = mask.shape
    points = []
    for x_value in range(width):
        ys = np.where(mask[:, x_value] > 0)[0]
        if ys.size:
            points.append((x_value, int(np.median(ys))))
    if len(points) < 12:
        return []
    deduped = []
    last_x = None
    for point in points:
        if point[0] != last_x:
            deduped.append(point)
            last_x = point[0]
    return deduped


def _quantized_color_candidates(image_np):
    flat = image_np.reshape(-1, 3).astype("float32")
    brightness = flat.mean(axis=1)
    spread = flat.std(axis=1)
    useful = flat[(brightness < 250) & ((spread > 10) | (brightness < 130))]
    if useful.size == 0:
        return []
    quantized = (useful // 32 * 32).astype("uint8")
    unique, counts = np.unique(quantized, axis=0, return_counts=True)
    candidates = []
    for color, count in zip(unique, counts):
        color_tuple = tuple(int(value) for value in color.tolist())
        score = float(count) + (np.std(color.astype("float32")) * 20.0)
        candidates.append((score, color_tuple, int(count)))
    candidates.sort(reverse=True)
    return candidates


def _extract_color_curve_masks(image_np, max_curves=4):
    masks = []
    for _, color, count in _quantized_color_candidates(image_np):
        if len(masks) >= max_curves:
            break
        rgb = np.array(color, dtype="int16")
        distance = np.linalg.norm(image_np.astype("int16") - rgb, axis=2)
        mask = np.where(distance < 34, 255, 0).astype("uint8")
        mask = _suppress_border_axes(mask)
        mask = cv2.medianBlur(mask, 3)
        mask = _filter_components(mask, min_pixels=max(30, count // 25))
        coverage = float(np.count_nonzero(mask)) / float(mask.size)
        if coverage < 0.001 or coverage > 0.2:
            continue
        if not _is_valid_curve_mask(mask):
            continue
        masks.append((color, mask))
    return masks


def _extract_grayscale_curve_masks(image_np, max_curves=3):
    grayscale = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    dark_mask = np.where(grayscale < 140, 255, 0).astype("uint8")
    dark_mask = _suppress_border_axes(dark_mask)
    dark_mask = cv2.medianBlur(dark_mask, 3)
    components, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
    masks = []
    ranked = []
    for index in range(1, components):
        area = stats[index, cv2.CC_STAT_AREA]
        if area < 40:
            continue
        ranked.append((area, index))
    ranked.sort(reverse=True)
    for _, index in ranked[:max_curves]:
        component_mask = np.where(labels == index, 255, 0).astype("uint8")
        if not _is_valid_curve_mask(component_mask):
            continue
        masks.append(((32, 32, 32), component_mask))
    return masks


def _pixel_points_to_data(points, plot_width, plot_height, axis_bounds):
    if not points:
        return []
    x_range = axis_bounds.x_max - axis_bounds.x_min
    y_range = axis_bounds.y_max - axis_bounds.y_min
    series = []
    for x_pixel, y_pixel in points:
        time_value = axis_bounds.x_min + (float(x_pixel) / max(plot_width - 1, 1)) * x_range
        survival = axis_bounds.y_max - (float(y_pixel) / max(plot_height - 1, 1)) * y_range
        if axis_bounds.y_max > 1.5:
            survival = survival / 100.0
        series.append((time_value, min(max(survival, 0.0), 1.0)))

    grouped = defaultdict(list)
    for time_value, survival in series:
        grouped[round(time_value, 3)].append(survival)

    ordered_times = sorted(grouped.keys())
    cleaned = []
    monotone_floor = 1.0
    for time_value in ordered_times:
        survival = float(np.median(grouped[time_value]))
        monotone_floor = min(monotone_floor, survival)
        cleaned.append((time_value, monotone_floor))
    return cleaned


def _estimate_curve_confidence(mask, data_points, axis_bounds):
    if not data_points:
        return 0.0
    coverage = float(np.count_nonzero(mask)) / float(mask.size)
    monotonic_penalty = 0.0
    survivals = [point[1] for point in data_points]
    for index in range(1, len(survivals)):
        if survivals[index] > survivals[index - 1] + 0.03:
            monotonic_penalty += 0.05
    confidence = 0.45 + min(0.35, coverage * 8.0) - monotonic_penalty
    if axis_bounds.y_max > 1.5:
        confidence -= 0.02
    return max(0.05, min(0.95, confidence))


def extract_curve_series(plot_image, axis_bounds, arm_label_overrides=None):
    image_np = preprocess_image(plot_image)
    color_masks = _extract_color_curve_masks(image_np)
    if not color_masks:
        color_masks = _extract_grayscale_curve_masks(image_np)

    curve_series = []
    for index, (color, mask) in enumerate(color_masks):
        pixel_points = _trace_curve(mask)
        data_points = _pixel_points_to_data(pixel_points, plot_image.width, plot_image.height, axis_bounds)
        if not data_points:
            continue
        label = "Arm {0}".format(index + 1)
        if arm_label_overrides and index < len(arm_label_overrides) and arm_label_overrides[index]:
            label = arm_label_overrides[index]
        warnings = []
        if len(data_points) < 25:
            warnings.append("Curve tracing captured limited points; review manually.")
        confidence = _estimate_curve_confidence(mask, data_points, axis_bounds)
        curve_series.append(
            CurveSeries(
                curve_id="curve_{0}".format(index + 1),
                arm_label=label,
                detected_color=color,
                pixel_points=pixel_points,
                data_points=data_points,
                confidence=confidence,
                warnings=warnings,
                detected_censor_count=max(0, len(pixel_points) // 50),
            )
        )
    return curve_series


def reproject_curves(curves, plot_image, axis_bounds, arm_label_overrides=None):
    updated = []
    for index, curve in enumerate(curves):
        data_points = _pixel_points_to_data(curve.pixel_points, plot_image.width, plot_image.height, axis_bounds)
        label = curve.arm_label
        if arm_label_overrides and index < len(arm_label_overrides) and arm_label_overrides[index]:
            label = arm_label_overrides[index]
        updated.append(
            CurveSeries(
                curve_id=curve.curve_id,
                arm_label=label,
                detected_color=curve.detected_color,
                pixel_points=curve.pixel_points,
                data_points=data_points,
                confidence=curve.confidence,
                warnings=list(curve.warnings),
                detected_censor_count=curve.detected_censor_count,
            )
        )
    return updated


def extract_km_data(request, image, config, arm_label_overrides=None):
    plot_image, plot_bbox = resolve_plot_region(image, manual_crop=request.manual_crop)
    ocr_text, ocr_warnings = run_ocr(plot_image)
    risk_bbox = detect_candidate_risk_bbox(image, plot_bbox)
    if risk_bbox is not None:
        risk_region = crop_image(image, risk_bbox)
    else:
        risk_region = image.crop((0, int(image.height * 0.72), image.width, image.height))
    risk_ocr_text, risk_ocr_warnings = run_ocr(risk_region)

    axis_dict = infer_axis_bounds(ocr_text, request.time_unit or None)
    if request.manual_axis_bounds is not None:
        axis_bounds = request.manual_axis_bounds
    else:
        axis_bounds = AxisBounds(**axis_dict)

    llm_guidance = call_multimodal_chart_review(plot_image, ocr_text, config)
    llm_labels = llm_guidance.get("arm_labels") or []
    if axis_bounds.time_unit == "months" and llm_guidance.get("time_unit"):
        axis_bounds.time_unit = llm_guidance["time_unit"]

    labels = arm_label_overrides or llm_labels
    curves = extract_curve_series(plot_image, axis_bounds, arm_label_overrides=labels)
    if not curves:
        raise RuntimeError("No curve candidates were found. Try manual crop or axis bounds.")

    curve_labels = [curve.arm_label for curve in curves]
    risk_rows = guess_risk_table_rows(risk_ocr_text or ocr_text, curve_labels)
    arm_mappings = [
        ArmMapping(
            curve_id=curve.curve_id,
            arm_label=curve.arm_label,
            detected_color=curve.detected_color,
            source="llm+cv" if llm_labels else "cv",
        )
        for curve in curves
    ]

    warnings = []
    warnings.extend(ocr_warnings)
    warnings.extend(risk_ocr_warnings)
    warnings.extend(llm_guidance.get("notes", []))
    if len(curves) < 2:
        warnings.append("Fewer than two curves were detected; pairwise analysis may be incomplete.")
    if not risk_rows:
        warnings.append("No risk table could be parsed automatically; reconstruction will use a heuristic fallback.")

    mean_curve_confidence = float(np.mean([curve.confidence for curve in curves])) if curves else 0.0
    mean_curve_confidence += float(llm_guidance.get("confidence_adjustment", 0.0))
    if not risk_rows:
        mean_curve_confidence -= 0.08
    mean_curve_confidence = max(0.05, min(0.95, mean_curve_confidence))

    review = ExtractionReview(
        arm_mappings=arm_mappings,
        axis_bounds=axis_bounds,
        risk_table_rows=risk_rows,
        confidence_score=mean_curve_confidence,
        warnings=warnings,
        ocr_text=(ocr_text + "\n" + risk_ocr_text).strip(),
        llm_notes=list(llm_guidance.get("notes", [])),
    )
    return {
        "plot_image": plot_image,
        "plot_bbox": plot_bbox,
        "risk_bbox": risk_bbox,
        "curves": curves,
        "review": review,
    }
