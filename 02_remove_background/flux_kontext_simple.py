import os
import sys
import argparse
from pathlib import Path

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

# Global pipeline cache
_KONTEXT_PIPE = None

def get_kontext_pipe():
    """Get cached Kontext pipeline"""
    global _KONTEXT_PIPE
    
    if _KONTEXT_PIPE is None:
        print("[INFO] Loading Kontext pipeline (first time)")
        _KONTEXT_PIPE = FluxKontextPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16
        )
        _KONTEXT_PIPE.to("cuda")
        print("[INFO] Kontext pipeline loaded")
    
    return _KONTEXT_PIPE

def run_kontext(image_path: str, output_path: str, prompt: str = DEFAULT_PROMPT):
    """
    Run Kontext background removal - uses the same logic as the working script
    but accepts dynamic image path instead of hardcoded one
    """
    
    # Get the pipeline (same as working script)
    pipe = get_kontext_pipe()
    
    # Load image (dynamic path instead of hardcoded)
    image = load_image(image_path).convert("RGB")
    
    # Use exact same parameters as working script
    prompt = prompt or DEFAULT_PROMPT
    image = pipe(
        image=image,
        prompt=prompt,
        guidance_scale=2.5,
        generator=torch.Generator().manual_seed(42),
    ).images[0]
    
    # Save to dynamic output path instead of hardcoded
    image.save(output_path)
    print(f"[DONE] Saved to: {output_path}")
    
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
