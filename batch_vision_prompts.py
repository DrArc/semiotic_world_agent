"""
batch_vision_prompts.py
-----------------------
Analyze a folder of images using a **local LM Studio** vision-enabled model and
generate a **semiotics-forward architecture prompt** per image. Results are written
to a CSV named **metadata.csv** with two columns: filename,prompt.
"""

from __future__ import annotations
import argparse, base64, csv, json, re, sys
from io import BytesIO
from pathlib import Path
from typing import List, Iterable

import requests
from PIL import Image

LM_BASE  = "http://localhost:1234/v1"
LM_MODEL = "google/gemma-3-27b"

SYSTEM_PROMPT = (
    "You are an expert architecture prompt engineer and image analyst. "
    "Your task: given one image, infer style, material palette, environment, lighting, and photography approach, "
    "then produce a **single prompt line** for an image generator that is **semiotics-forward**. "
    "Constrain to <220 words. "
    "MANDATORY: include the phrase 'axonometric at 45°'. "
    'Respond with EXACTLY one JSON object, no code fences, no prose: {"prompt": "<one-line prompt>"}'
)

USER_INSTR = (
    "Look carefully at the image and synthesize a prompt. "
    "Include: architecture style, environment/context, material palette, light mood, photography type. "
    "Bias toward denotation/connotation clarity (Barthes): "
    "briefly suggest the connotation (e.g., civic transparency vs surveillance, domesticity vs spectacle) in the wording. "
    "MANDATORY: add 'axonometric at 45°'. Return JSON only."
)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def _encode_image_data_url(path: Path, max_side: int = 1024) -> str:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = max(w, h) / max_side if max(w, h) > max_side else 1.0
    if scale > 1.0:
        img = img.resize((int(w / scale), int(h / scale)))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def _extract_json(text: str) -> dict:
    # Accept ```json ... ``` or plain JSON or first {...}
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidate = m.group(1) if m else None
    if not candidate:
        m2 = re.search(r"\{.*\}", text, flags=re.DOTALL)
        candidate = m2.group(0) if m2 else None
    if not candidate:
        raise ValueError("No JSON object found in model response.")
    return json.loads(candidate)

def call_lm(image_path: Path, model: str, max_side: int, timeout_s: int) -> str:
    data_url = _encode_image_data_url(image_path, max_side=max_side)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": USER_INSTR},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]},
        ],
        "temperature": 0.2
    }
    r = requests.post(f"{LM_BASE}/chat/completions", json=payload, timeout=timeout_s)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    try:
        obj = json.loads(content)
    except Exception:
        obj = _extract_json(content)
    prompt = (obj.get("prompt") or "").strip()
    if not prompt:
        raise RuntimeError("Model returned empty prompt.")
    # normalize whitespace
    prompt = re.sub(r"\s+", " ", prompt)
    return prompt

def iter_images(src: Path, patterns: List[str]) -> Iterable[Path]:
    if not src.exists():
        raise FileNotFoundError(f"--src not found: {src}")
    if src.is_file() and src.suffix.lower() in IMG_EXTS:
        yield src
        return
    pats = [p.strip() for p in patterns if p.strip()]
    if not pats:
        pats = ["*.png", "*.jpg", "*.jpeg"]
    for pat in pats:
        for p in sorted(src.rglob(pat)):
            if p.suffix.lower() in IMG_EXTS:
                yield p

def main():
    ap = argparse.ArgumentParser(description="Batch vision analysis → semiotics-forward prompts CSV")
    ap.add_argument("--src", required=True, help="Source directory path containing images")
    ap.add_argument("--out", default="metadata.csv", help="Output CSV path (default: metadata.csv)")
    ap.add_argument("--model", default=LM_MODEL, help="LM Studio model name")
    ap.add_argument("--max-side", type=int, default=1024, help="Max image side before sending to model")
    ap.add_argument("--timeout", type=int, default=180, help="Per-image timeout seconds")
    ap.add_argument("--pattern", default="*.png,*.jpg,*.jpeg", help="Comma list of glob patterns")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)

    patterns = [s.strip() for s in args.pattern.split(",")]
    images = list(iter_images(src, patterns))
    if not images:
        print(f"[FATAL] no images found in {src} matching {patterns}")
        sys.exit(1)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "prompt"])
        for img in images:
            try:
                print(f"[INFO] analyzing: {img}")
                prompt = call_lm(img, args.model, args.max_side, args.timeout)
                w.writerow([img.name, prompt])
            except Exception as e:
                print(f"[WARN] failed on {img.name}: {e}")
                # fallback prompt that still respects format
                fallback = (
                    "axonometric at 45°, architectural study, semiotics-oriented description; "
                    "style unspecified; materials unspecified; soft diffuse light; "
                    "photography axonometric study"
                )
                w.writerow([img.name, fallback])
    print(f"[DONE] wrote {out.resolve()}")

if __name__ == "__main__":
    main()
