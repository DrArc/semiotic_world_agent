import json, re, requests

LM_BASE = "http://localhost:1234/v1"   # LM Studio OpenAI-compatible server
LM_MODEL = "google/gemma-3-27b"                   # set to the model name you started in LM Studio

SYSTEM_PROMPT = (
    "You are a controller for a design tool. "
    "Respond with EXACTLY one JSON object, no code fences, no backticks, no prose. "
    'Schema: {"action": "generate_image|generate_depth|monochrome_clean|comfy360|generate_3d|analyze_image", '
    '"prompt": "<string, required for generate_image>", '
    '"style": "<string, optional>", '
    '"image": "<string, optional for analyze_image>", '
    '"depth": "<string, optional for analyze_image>", '
    '"goals": "<string, optional for analyze_image>"}'
)

def _extract_json_blocks(text: str) -> dict:
    """
    Accepts content that may include ```json ... ``` or prose.
    Returns the first valid JSON object found.
    """
    # 1) Prefer fenced block ```json ... ```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidate = None
    if m:
        candidate = m.group(1)
    else:
        # 2) Fallback: first {...} region
        m2 = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m2:
            candidate = m2.group(0)
        else:
            raise ValueError("No JSON object found in response.")
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e

def ask_agent(user_text: str) -> dict:
    payload = {
        "model": LM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text}
        ],
        "temperature": 0
    }

    # Don't use response_format - our JSON extraction is robust enough
    # and LM Studio's json_schema requires a full schema definition

    r = requests.post(f"{LM_BASE}/chat/completions", json=payload, timeout=120)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]

    try:
        return _extract_json_blocks(content)
    except Exception:
        # Return diagnostic so the UI can show what came back
        return {"action": "noop", "error": "Agent did not return valid JSON", "raw": content}

