import argparse
import os
import sys
from pathlib import Path
import torch
from diffusers import FluxPipeline
from peft import PeftModel

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

def pick_dtype(device, override: str | None = None):
    if override:
        override = override.lower()
        if override == "float16": return torch.float16
        if override == "bfloat16": return torch.bfloat16
        if override == "float32": return torch.float32
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32

def load_semiocity_lora(pipe, lora_path, device, dtype):
    """Load SemioCity LoRa for FLUX pipeline using the correct approach"""
    try:
        # Get the project root directory (two levels up from this file)
        project_root = Path(__file__).parent.parent
        lora_dir = project_root / "models" / "SemioCity_LoRa"
        lora_filename = "Semiocity_flux.safetensors"
        
        if not lora_dir.exists():
            print(f"[WARN] LoRa directory not found: {lora_dir}")
            return pipe
            
        lora_file_path = lora_dir / lora_filename
        if not lora_file_path.exists():
            print(f"[WARN] LoRa file not found: {lora_file_path}")
            return pipe
            
        print(f"[INFO] Loading SemioCity LoRa from directory: {lora_dir}")
        print(f"[INFO] LoRa filename: {lora_filename}")
        
        # Use the correct approach from the blog
        pipe.load_lora_weights(
            str(lora_dir), 
            weight_name=lora_filename, 
            prefix=None
        )
        
        print("[INFO] SemioCity LoRa loaded successfully")
        
        return pipe
        
    except Exception as e:
        print(f"[WARN] Failed to load LoRa: {e}")
        print("[INFO] Continuing without LoRa...")
        return pipe

def main(prompt, out_path, steps, seed, model_id, use_lora=True, guidance: float | None = None, dtype_override: str | None = None):
    # Load Hugging Face token first
    load_hf_token()
    
    device = infer_device()
    dtype = pick_dtype(device, dtype_override)
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

    # Load SemioCity LoRa for architectural generation (if enabled)
    if use_lora:
        pipe = load_semiocity_lora(pipe, None, device, dtype)  # Path is handled inside the function

    # Use CPU generator for stability (CUDA generator was causing errors)
    gen = torch.Generator("cpu").manual_seed(seed)

    # Guidance defaults: FLUX.1-dev generally benefits from 3.0–5.0; schnell from ~0.0–1.0
    if guidance is not None:
        guidance_scale = float(guidance)
    else:
        if "FLUX.1-dev" in model_id:
            guidance_scale = 3.5
        else:
            guidance_scale = 1.0 if use_lora else 0.0
    
    print(f"[INFO] generating {W}x{H}, steps={steps}, guidance_scale={guidance_scale}")
    img = pipe(
        prompt,
        width=W, height=H,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        max_sequence_length=256,
        generator=gen,
    ).images[0]

    out_path = Path(out_path)
    if not out_path.suffix:
        out_path = out_path.with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(out_path)
    print(f"[DONE] saved {out_path} size={img.size}")
    
    # Free VRAM for Step 2
    import gc
    try:
        del pipe, img
    except NameError:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print("[INFO] VRAM freed for Step 2")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20)  # 20 steps work well with LoRa, 2-4 for schnell without LoRa
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="black-forest-labs/FLUX.1-dev", help="Default to FLUX.1-dev (best with SemioCity LoRa)")
    ap.add_argument("--guidance", type=float, default=None, help="Classifier-free guidance scale")
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=None, help="Force dtype override")
    ap.add_argument("--no-lora", action="store_true", help="Disable SemioCity LoRa")
    args = ap.parse_args()
    main(args.prompt, args.out, args.steps, args.seed, args.model, use_lora=not args.no_lora, guidance=args.guidance, dtype_override=args.dtype)
