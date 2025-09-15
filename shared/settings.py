import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

ORCH_RUNNER = os.getenv("ORCH_RUNNER", "venv")

FLUX_PY    = Path(os.getenv("FLUX_PY", ROOT / "01_ai_generation/.venv/bin/python"))
DEPTH_PY   = Path(os.getenv("DEPTH_PY", ROOT / "03_depth_from_image/.venv/bin/python"))
KONTEXT_PY = Path(os.getenv("KONTEXT_PY", ROOT / "02_remove_background/.venv/bin/python"))
COMFY_PY   = Path(os.getenv("COMFY_PY", ROOT / "05_360_from_image/.venv/bin/python"))

COMFY_HOST = os.getenv("COMFY_HOST", "http://127.0.0.1:8188")

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", ROOT / "outputs")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
