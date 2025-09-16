from __future__ import annotations
import base64, json, re, requests
from io import BytesIO
from pathlib import Path
from PIL import Image

LM_BASE  = "http://localhost:1234/v1"     # LM Studio server
LM_MODEL = "google/gemma-3n-e4b"                  # set to the exact model name in LM Studio

SYSTEM_PROMPT = (
    "You are an expert architecture image analyst. "
    "Respond with EXACTLY one JSON object, no code fences, no prose. "
    'Schema: {"summary": "<string>", '
    '"denotation": "<string>", "connotation": "<string>", "myth": "<string>", '
    '"materials": ["<string>", ...], "style": ["<string>", ...], "notes": ["<string>", ...]}'
)

def _extract_json(text: str) -> dict:
    # Accept ```json ... ``` or plain JSON
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidate = m.group(1) if m else None
    if not candidate:
        m2 = re.search(r"\{.*\}", text, flags=re.DOTALL)
        candidate = m2.group(0) if m2 else None
    if not candidate:
        raise ValueError("No JSON object found.")
    return json.loads(candidate)

def _encode_image_as_data_url(path: str | Path, max_side: int = 1024) -> str:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = max(w, h) / max_side if max(w, h) > max_side else 1.0
    if scale > 1:
        img = img.resize((int(w / scale), int(h / scale)))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def analyze_with_lmstudio(image_path: str | Path,
                          depth_path: str | Path | None = None,
                          goals: str | None = None,
                          model: str = LM_MODEL,
                          timeout_s: int = 180) -> dict:
    """
    Send image (and optional depth map) to LM Studio vision model and return a JSON dict.
    """
    content = []
    content.append({"type": "text", "text": goals or
                   "Analyze the architectural image and produce the JSON schema described."})
    content.append({"type": "image_url", "image_url": {"url": _encode_image_as_data_url(image_path)}})
    if depth_path:
        content.append({"type": "text", "text": "Here is the corresponding depth map for geometric cues."})
        content.append({"type": "image_url", "image_url": {"url": _encode_image_as_data_url(depth_path)}})

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": content},
        ],
        "temperature": 0
    }

    # Don't use response_format - our JSON extraction is robust enough
    # and LM Studio's json_schema requires a full schema definition

    try:
        r = requests.post(f"{LM_BASE}/chat/completions", json=payload, timeout=timeout_s)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]

        try:
            return json.loads(content)
        except Exception:
            # fallback: strip fences or find the first {...}
            return _extract_json(content)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"LM Studio request failed: {e}")
    except Exception as e:
        raise RuntimeError(f"Vision analysis failed: {e}")
