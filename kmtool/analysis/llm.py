import base64
import io
import json
import re

import requests


def _image_to_data_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/png;base64,{0}".format(encoded)


def _extract_json_payload(text):
    if not text:
        return {}
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end > start:
        candidate = candidate[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return {}


def call_multimodal_chart_review(image, ocr_text, config):
    if not config.llm_enabled:
        return {"notes": ["LLM disabled because credentials are missing."], "arm_labels": []}

    url = "{0}/chat/completions".format(config.llm_api_base)
    prompt = (
        "You are reviewing a Kaplan-Meier survival chart extraction.\n"
        "Respond with strict JSON only using the keys: arm_labels, time_unit, notes, confidence_adjustment.\n"
        "arm_labels must be a short list of labels inferred from the legend.\n"
        "time_unit must be one of months, weeks, days, years if inferable.\n"
        "notes must list OCR corrections, panel warnings, or legend interpretations.\n"
        "confidence_adjustment must be between -0.3 and 0.1.\n"
        "OCR text:\n{0}\n".format(ocr_text[:3000])
    )
    payload = {
        "model": config.llm_model,
        "temperature": 0.1,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _image_to_data_url(image)}},
                ],
            }
        ],
    }
    headers = {
        "Authorization": "Bearer {0}".format(config.llm_api_key),
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=config.llm_timeout_seconds)
        response.raise_for_status()
        body = response.json()
        message = body["choices"][0]["message"]["content"]
        if isinstance(message, list):
            message = "\n".join(item.get("text", "") for item in message if isinstance(item, dict))
        parsed = _extract_json_payload(message)
        if not parsed:
            return {"notes": ["LLM returned non-JSON guidance."], "arm_labels": []}
        parsed.setdefault("notes", [])
        parsed.setdefault("arm_labels", [])
        parsed.setdefault("confidence_adjustment", 0.0)
        return parsed
    except Exception as exc:  # pragma: no cover - network dependent
        return {"notes": ["LLM review failed: {0}".format(exc)], "arm_labels": []}
