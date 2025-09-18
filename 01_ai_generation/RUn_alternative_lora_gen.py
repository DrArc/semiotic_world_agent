import argparse
import os, sys
from pathlib import Path
import torch
from diffusers import FluxPipeline

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

def add_lora(pipe, lora_path, adapter_name="lora", weight=1.0, fuse=False):
    p = Path(lora_path)
    if not p.exists():
        raise FileNotFoundError(f"LoRA not found: {p}")

    # Diffusers accepts either a file path or a dir + weight_name; prefer the file path.
    if p.is_file():
        pipe.load_lora_weights(str(p), adapter_name=adapter_name, local_files_only=True)
    else:
        # Pick the first .safetensors in the folder
        st_files = sorted(p.glob("*.safetensors"))
        if not st_files:
            raise FileNotFoundError(f"No .safetensors in {p}")
        pipe.load_lora_weights(str(p), weight_name=st_files[0].name, adapter_name=adapter_name, local_files_only=True)

    # Set scale and optionally fuse for speed
    pipe.set_adapters(adapter_name, adapter_weights=weight)
    if fuse:
        pipe.fuse_lora(lora_scale=weight)
        # Free adapter modules after fusing; fused weights stay in place
        pipe.unload_lora_weights(reset_to_overwritten_params=True)

def main(prompt, out_path, steps, seed, model_id, lora, lora_weight, fuse_lora, guidance):
    # 1) Auth (optional)
    # load_hf_token()

    # 2) Device + dtype
    device = infer_device()
    dtype = pick_dtype(device)
    print(f"[INFO] device={device} dtype={dtype}")

    # 3) Pipeline
    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype, use_safetensors=True)
    try:
        pipe.to(device)
        print("[INFO] pipeline on device")
    except Exception as e:
        print(f"[WARN] pipe.to({device}) failed: {e}; enabling CPU offload.")
        pipe.enable_model_cpu_offload()

    # 4) Attach local LoRA if given
    if lora:
        name = Path(lora).stem
        print(f"[INFO] loading LoRA: {lora} as '{name}', weight={lora_weight}, fuse={fuse_lora}")
        add_lora(pipe, lora, adapter_name=name, weight=lora_weight, fuse=fuse_lora)
        try:
            print("[INFO] active adapters:", pipe.get_active_adapters())
        except Exception:
            pass

    # 5) Flux flavor‑aware defaults
    is_schnell = "schnell" in model_id.lower()
    if is_schnell:
        guidance_scale = 0.0
        steps = min(steps, 8)         
        extra = dict(max_sequence_length=256)
    else:
        guidance_scale = guidance     
        extra = {}

    # For reproducibility a CPU generator is fine
    gen = torch.Generator("cpu").manual_seed(seed)
    print(f"[INFO] generating {W}x{H}, steps={steps}, gs={guidance_scale}")
    img = pipe(
        prompt,
        width=W, height=H,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        generator=gen,
        **extra,
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
    ap.add_argument("--steps", type=int, default=4)           # sensible default for schnell
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="black-forest-labs/FLUX.1-schnell")
    ap.add_argument("--lora", default=None, help="Path to .safetensors or folder")
    ap.add_argument("--lora-weight", type=float, default=0.8)
    ap.add_argument("--fuse-lora", action="store_true")
    ap.add_argument("--guidance", type=float, default=3.5)    # used for dev
    args = ap.parse_args()
    main(args.prompt, args.out, args.steps, args.seed, args.model, args.lora, args.lora_weight, args.fuse_lora, args.guidance)