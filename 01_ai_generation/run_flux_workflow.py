import argparse
import os
import sys
from pathlib import Path
import torch
from diffusers import FluxPipeline

# Add Hugging Face token support
def load_hf_token():
    """Load Hugging Face token from environment or api/keys.py"""
    # Try environment variables first
    token = (
        os.environ.get("HUGGINGFACE_HUB_TOKEN") or
        os.environ.get("HUGGINGFACE_TOKEN") or
        os.environ.get("HF_TOKEN")
    )
    
    if not token:
        # Try to load from api/keys.py
        try:
            # Add parent directory to path to find api/keys.py
            parent_dir = Path(__file__).parent.parent
            sys.path.insert(0, str(parent_dir))
            from api.keys import HF_TOKEN
            token = HF_TOKEN
        except ImportError:
            print("[WARN] No Hugging Face token found. Using public models only.")
            return None
    
    if token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False, new_session=False)
            print("[INFO] Hugging Face token loaded successfully")
        except Exception as e:
            print(f"[WARN] HuggingFace login failed: {e}")
    
    return token

W, H = 1024, 1024  # both divisible by 16

def infer_device():
    if torch.cuda.is_available(): return "cuda"
    try:
        if torch.backends.mps.is_available(): return "mps"
    except Exception:
        pass
    return "cpu"

def pick_dtype(device):
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32

def main(prompt, out_path, steps, seed, model_id):
    # Load Hugging Face token first
    load_hf_token()
    
    device = infer_device()
    dtype = pick_dtype(device)
    print(f"[INFO] device={device} dtype={dtype}")

    # Use torch_dtype (per model card / your logs); dtype was being ignored.
    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)

    # Prefer running fully on GPU; only use offload if .to() fails.
    try:
        pipe.to(device)
        print("[INFO] pipeline on device")
    except Exception as e:
        print(f"[WARN] pipe.to({device}) failed: {e}; enabling CPU offload (slower).")
        pipe.enable_model_cpu_offload()

    # Schnell expects guidance_scale=0.0 and very few steps.
    # The example uses a CPU generator; keep it simple and stable.
    gen = torch.Generator("cpu").manual_seed(seed)

    print(f"[INFO] generating {W}x{H}, steps={steps}")
    img = pipe(
        prompt,
        width=W, height=H,
        guidance_scale=0.0,
        num_inference_steps=steps,      # 1–4 is the sweet spot
        max_sequence_length=256,
        generator=gen,
    ).images[0]

    out_path = Path(out_path)
    if not out_path.suffix:
        out_path = out_path.with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(out_path)
    print(f"[DONE] saved {out_path} size={img.size}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20)  # start at 2; try 3–4 if needed
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="black-forest-labs/FLUX.1-schnell")
    args = ap.parse_args()
    main(args.prompt, args.out, args.steps, args.seed, args.model)
