import os
import sys
import argparse
from pathlib import Path
import gc
import time

import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image

# Use the same prompt as the working script
DEFAULT_PROMPT = "convert image into a clean monochrome architectural model with no textures,ambient lighting and clean smooth geometry while removing details such as people, railings, background, water and trees"  #Prompt used by Michal Gryko's comfyui workflow

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

def run_kontext(image_path: str, output_path: str, prompt: str = DEFAULT_PROMPT,
                steps: int = 20, guidance: float = 3.0):
    """
    Run Kontext background removal - memory-safe approach
    """
    print(f"[INFO] Kontext: loading {image_path}")
    
    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32  # Key: no bf16 on CPU
    
    pipe = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=dtype,
        use_safetensors=True
    )
    
    # Memory-friendly placement
    if use_cuda:
        try:
            pipe.to("cuda")
            print("[INFO] Kontext on CUDA")
        except torch.cuda.OutOfMemoryError:
            print("[WARN] OOM on CUDA; enabling CPU offload")
            torch.cuda.empty_cache()
            pipe.enable_model_cpu_offload()  # Keeps most work on GPU but swaps modules
            print("[INFO] Kontext using CPU offload")
    else:
        pipe.to("cpu")
        print("[INFO] Kontext on CPU (float32)")
    
    # Memory optimizations
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    
    # Load image
    image = load_image(image_path).convert("RGB")
    print(f"[INFO] Image loaded, size: {image.size}")
    print(f"[INFO] Inference: steps={steps}, guidance={guidance}")
    
    # Run inference
    t0 = time.time()
    out = pipe(
        image=image,
        prompt=prompt,
        guidance_scale=guidance,
        num_inference_steps=steps,
        generator=torch.Generator("cpu").manual_seed(42),
    ).images[0]
    
    inference_time = time.time() - t0
    print(f"[INFO] Kontext done in {inference_time:.1f}s")
    
    # Save output
    out.save(output_path)
    print(f"[DONE] Saved to: {output_path}")
    
    # Cleanup
    del out, image, pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return output_path

# Original working script (only runs when executed directly)
if __name__ == "__main__":
    # This is your original working code - unchanged
    pipe = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16
    )
    pipe.to("cuda")

    image = load_image("output\images\gen_20250911-161206.png").convert("RGB")
    prompt = "convert image into a clean monochrome architectural model with no textures,ambient lighting and clean smooth geometry while removing details such as people, railings, background, water and trees"
    image = pipe(
        image=image,
        prompt=prompt,
        guidance_scale=2.5,
        generator=torch.Generator().manual_seed(42),
    ).images[0]
    image.save("flux-kontext.png")
