"""Hi-res refine/upscale a generated image via SDXL img2img with the epoch-12 LoRA.

Usage: python scripts/upscale.py <in.png> <out.png> "<prompt>" [denoise] [target_px]
Upscales by re-rendering at higher resolution with low denoise, preserving
composition while adding in-style detail.
"""
from __future__ import annotations
import sys
import torch
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline, EulerAncestralDiscreteScheduler

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "models" / "sd_xl_base_1.0.safetensors"
LORA = ROOT / "output" / "v3" / "tk3k_portrait_v3.safetensors"

inp = Path(sys.argv[1]); outp = Path(sys.argv[2]); prompt = sys.argv[3]
denoise = float(sys.argv[4]) if len(sys.argv) > 4 else 0.35
target = int(sys.argv[5]) if len(sys.argv) > 5 else 1536
NEGATIVE = ("low quality, blurry, jpeg artifacts, signature, text, watermark, "
            "extra limbs, deformed, multiple characters, reference sheet")

print(f"loading base for upscale ...", flush=True)
pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
    str(BASE), torch_dtype=torch.float16, use_safetensors=True, add_watermarker=False
)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")
pipe.enable_vae_slicing()
pipe.load_lora_weights(str(LORA.parent), weight_name=LORA.name)

img = Image.open(inp).convert("RGB").resize((target, target), Image.LANCZOS)
gen = torch.Generator(device="cuda").manual_seed(12345)
out = pipe(
    prompt=prompt, negative_prompt=NEGATIVE, image=img,
    strength=denoise, num_inference_steps=40, guidance_scale=6.5, generator=gen,
).images[0]
outp.parent.mkdir(parents=True, exist_ok=True)
out.save(outp)
print(f"wrote {outp} ({out.size[0]}x{out.size[1]})", flush=True)
