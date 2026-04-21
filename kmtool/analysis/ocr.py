import re
from typing import List, Sequence

from kmtool.models import RiskTableRow

try:
    import pytesseract
except ImportError:  # pragma: no cover - optional dependency warning handled upstream
    pytesseract = None


def run_ocr(image, psm=6):
    if pytesseract is None:
        return "", ["pytesseract not installed; OCR disabled."]
    try:
        config = "--oem 3 --psm {0}".format(psm)
        text = pytesseract.image_to_string(image, config=config)
        return text, []
    except Exception as exc:  # pragma: no cover - depends on local Tesseract install
        return "", ["OCR failed: {0}".format(exc)]


def infer_axis_bounds(ocr_text, time_unit_hint=None):
    text = (ocr_text or "").lower()
    numbers = [float(match) for match in re.findall(r"(?<!\d)(\d+(?:\.\d+)?)", text)]
    unit = time_unit_hint or "months"
    for candidate in ("months", "month", "weeks", "week", "days", "day", "years", "year"):
        if candidate in text:
            unit = candidate.rstrip("s") + "s"
            break

    x_max = max(numbers) if numbers else 100.0
    if x_max <= 1.0:
        x_max = 100.0

    uses_percent = "%" in text or "percent" in text
    y_max = 100.0 if uses_percent else 1.0
    return {
        "x_min": 0.0,
        "x_max": float(x_max),
        "y_min": 0.0,
        "y_max": float(y_max),
        "time_unit": unit,
    }


def guess_risk_table_rows(ocr_text, arm_labels):
    if not ocr_text:
        return []
    lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]
    numeric_lines = []
    for line in lines:
        values = re.findall(r"\d+", line)
        if len(values) >= 3:
            numeric_lines.append((line, [int(value) for value in values]))

    if len(numeric_lines) < 2:
        return []

    times = None
    count_rows = []
    for raw_line, values in numeric_lines:
        if values == sorted(values) and len(set(values)) > 1:
            times = values
            continue
        count_rows.append((raw_line, values))

    if times is None:
        longest = max(numeric_lines, key=lambda item: len(item[1]))
        times = list(range(0, len(longest[1]) * 6, 6))
        count_rows = numeric_lines[:]

    rows = []
    for arm_index, arm_label in enumerate(arm_labels):
        if arm_index >= len(count_rows):
            break
        raw_line, values = count_rows[arm_index]
        if len(values) != len(times):
            limit = min(len(values), len(times))
            values = values[:limit]
            current_times = times[:limit]
        else:
            current_times = times
        label = arm_label or raw_line.split()[0]
        rows.append((label, current_times, values))

    if not rows:
        return []

    merged = []
    for time_index in range(len(rows[0][1])):
        arm_counts = {}
        time_point = float(rows[0][1][time_index])
        for label, _, values in rows:
            if time_index < len(values):
                arm_counts[label] = int(values[time_index])
        merged.append(RiskTableRow(time=time_point, arm_counts=arm_counts))
    return merged


def parse_manual_risk_table(table_text):
    rows = []
    if not table_text.strip():
        return rows
    lines = [line.strip() for line in table_text.splitlines() if line.strip()]
    header = [cell.strip() for cell in lines[0].split(",")]
    arm_labels = header[1:]
    for line in lines[1:]:
        parts = [cell.strip() for cell in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            time_point = float(parts[0])
        except ValueError:
            continue
        counts = {}
        for label, cell in zip(arm_labels, parts[1:]):
            try:
                counts[label] = int(float(cell))
            except ValueError:
                continue
        rows.append(RiskTableRow(time=time_point, arm_counts=counts))
    return rows


def risk_table_to_csv(rows, arm_labels):
    if not rows:
        return "time,{0}".format(",".join(arm_labels))
    header = ["time"] + list(arm_labels)
    lines = [",".join(header)]
    for row in rows:
        values = [str(row.time)]
        for label in arm_labels:
            values.append(str(row.arm_counts.get(label, "")))
        lines.append(",".join(values))
    return "\n".join(lines)
