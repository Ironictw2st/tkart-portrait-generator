"""
Generate a TW3K artset for a new character using the trained LoRA.

Generates ONE image and writes it identically into all three emotion slots
(norm/happy/angry × large/small panels). The mod's folder layout requires
those files to exist, but using the same image avoids identity drift between
variants — and the original SAD modpack itself often re-uses near-identical
art across emotion slots for the same character.

Outputs in the same folder layout as SAD_artsets/SAD_*:

    output/SAD_<slug>_<element>/
        large_panel/{norm,happy,angry}/noanim.png   # 1200x1200 RGBA (identical)
        small_panel/{norm,happy,angry}/noanim.png   # 1120x1120 RGBA (identical)

Backgrounds are recovered from the LoRA's predictable near-white output via a
brightness-threshold mask. Works because every training image was composited
onto pure white. If the LoRA produces dark backgrounds (rare), pass
--use-rembg to fall back to a learned segmentation model.

Usage:
    python scripts/generate.py --slug cao_pi --gender male --element fire --seed 42
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
OUT_ROOT = ROOT / "output" / "artsets"

EMOTIONS = ("norm", "happy", "angry")
EMOTION_TO_PROMPT = {
    "norm": "norm expression",
    "happy": "happy expression",
    "angry": "angry expression",
}
GEN_RES = 1024  # SDXL native; resized to 1200/1120 in post


def build_prompt(gender: str, element: str | None, emotion: str) -> str:
    """Match the kohya sample_prompts.txt format that produced the clean epoch-6
    samples. Specifically: NO "chinese" qualifier (kohya's sample prompts didn't
    use it) and the element tag stays bare ("fire element"), not "fire wuxing
    element". The LoRA generalizes across the training-caption variants but
    the sample format is what we know empirically works at 30-step euler_a.
    """
    gender_word = "man" if gender == "male" else "woman"
    parts = [
        "tk3k_portrait",
        # `solo` is a Danbooru-derived tag SDXL knows from its base training.
        # It strongly anchors composition to a single subject without altering
        # the LoRA-driven painted style.
        "solo",
        # Framing cues. Single-token "full body portrait" turned out to be too
        # weak — some seeds still landed on bust crops. Stacking redundant
        # framing tokens (full body / head to toe / standing / feet visible)
        # gives the model multiple reinforcing signals to render the entire
        # figure rather than zoom in on the upper body.
        "full body portrait",
        "head to toe",
        "standing pose",
        "feet visible",
        gender_word,
        EMOTION_TO_PROMPT[emotion],
        "white background",
    ]
    if element:
        parts.append(f"{element} element")
    return ", ".join(parts)


def whiteness_alpha(img_rgb: Image.Image, fade_start: int = 235, fade_end: int = 252) -> Image.Image:
    """Convert near-white BG to transparency.

    Pixels with min(RGB) >= fade_end => fully transparent.
    Pixels with min(RGB) <= fade_start => fully opaque.
    Linear ramp between.

    This works because the LoRA was trained on hard-white-composited images, so
    its output BG converges to ~(255,255,255) while the subject stays well
    below the threshold.
    """
    arr = np.asarray(img_rgb.convert("RGB"), dtype=np.uint8)
    min_chan = arr.min(axis=2)
    span = max(fade_end - fade_start, 1)
    alpha = np.clip(255 - ((min_chan - fade_start) * 255 // span), 0, 255).astype(np.uint8)
    rgba = np.dstack([arr, alpha])
    return Image.fromarray(rgba, mode="RGBA")


def load_pipeline(base_model: Path, lora_path: Path, lora_weight: float):
    from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

    pipe = StableDiffusionXLPipeline.from_single_file(
        str(base_model),
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        add_watermarker=False,
    )
    # Match the scheduler kohya used to render training samples — euler_a.
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(str(lora_path.parent), weight_name=lora_path.name)
    pipe.fuse_lora(lora_scale=lora_weight)
    pipe.to("cuda")
    pipe.enable_vae_slicing()
    return pipe


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True, help='character name slug, e.g. "cao_pi"')
    ap.add_argument("--gender", choices=["male", "female"], default="male")
    ap.add_argument("--element", choices=["wood", "fire", "water", "earth", "metal"], required=True)
    ap.add_argument("--seed", type=int, default=None,
                    help="seed for generation. Defaults to a stable hash of slug+element so each "
                         "character gets a unique-but-reproducible seed without manual picking. "
                         "Pass an explicit int to override.")
    ap.add_argument("--base", default=str(MODELS / "sd_xl_base_1.0.safetensors"))
    ap.add_argument("--lora", default=str(ROOT / "output" / "tk3k_portrait_v2.safetensors"))
    # LoRA scale 1.0 matches the training-time scale and the kohya sample renderer.
    ap.add_argument("--lora-weight", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=30)
    # CFG 7.5 is kohya's sample-renderer default. We saw the cloud epoch-6 samples
    # render cleanly at this CFG without a negative prompt; deviating produced the
    # multi-pose reference-sheet outputs.
    ap.add_argument("--cfg", type=float, default=7.5)
    # Negatives serve two purposes:
    # (1) Suppress SDXL's prior toward concept-art reference sheets / multi-pose.
    # (2) Suppress literal-element interpretation. At CFG 7.5 the model reads
    #     "fire element" as "casting fire magic" by default, producing torches /
    #     fireballs / flaming weapons. We want it as a costume-color cue only.
    ap.add_argument("--negative", default=(
        "multiple poses, reference sheet, character sheet, multiple views, "
        "side view, back view, multiple characters, two people, pair, duo, "
        "group, split image, collage, "
        # Anti-bust-crop negatives. SDXL's prior toward portrait framing is
        # strong; without these some seeds zoom in to the head/chest. Pairing
        # these with the "head to toe / feet visible" positive cues gives
        # robust full-body composition across seeds.
        "bust portrait, headshot, half body, cropped, close-up, waist up, "
        "chest up, portrait crop, "
        "torch, fireball, flame, fire effects, holding fire, magical effects, "
        "glowing weapon, energy effects, spell casting, "
        "low quality, blurry, signature, text, watermark"
    ))
    ap.add_argument("--use-rembg", action="store_true", help="use rembg for BG removal instead of threshold")
    args = ap.parse_args()

    base = Path(args.base)
    lora = Path(args.lora)
    if not base.exists():
        raise SystemExit(f"base model not found: {base}")
    if not lora.exists():
        raise SystemExit(f"lora not found: {lora}")

    # If no --seed given, derive one from slug+element. This gives each character
    # a unique starting noise pattern by default, fixing the "every character
    # looks the same because they all use seed 1337" problem. The hash is stable
    # so re-running the same character produces the same output.
    if args.seed is None:
        import hashlib
        digest = hashlib.sha1(f"{args.slug}|{args.element}".encode("utf-8")).digest()
        args.seed = int.from_bytes(digest[:4], "big")
        print(f"  (auto-seed from slug+element: {args.seed})", flush=True)

    out_dir = OUT_ROOT / f"SAD_{args.slug}_{args.element}"
    (out_dir / "large_panel").mkdir(parents=True, exist_ok=True)
    (out_dir / "small_panel").mkdir(parents=True, exist_ok=True)

    print(f"loading pipeline (base={base.name}, lora={lora.name}@{args.lora_weight})...", flush=True)
    pipe = load_pipeline(base, lora, args.lora_weight)

    if args.use_rembg:
        from rembg import remove as rembg_remove
        to_alpha = lambda im: rembg_remove(im)
    else:
        to_alpha = whiteness_alpha

    # Generate the character once, then write the same image into all three
    # emotion slots. The mod requires norm/happy/angry files but using the same
    # image avoids identity drift and cuts generation cost to 1/3.
    prompt = build_prompt(args.gender, args.element, "norm")
    print(f"  seed={args.seed} :: {prompt}", flush=True)
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    rgb = pipe(
        prompt=prompt,
        negative_prompt=args.negative,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        width=GEN_RES,
        height=GEN_RES,
        generator=gen,
    ).images[0]

    rgba = to_alpha(rgb)
    large = rgba.resize((1200, 1200), Image.LANCZOS)
    small = rgba.resize((1120, 1120), Image.LANCZOS)
    for emotion in EMOTIONS:
        large_path = out_dir / "large_panel" / emotion / "noanim.png"
        small_path = out_dir / "small_panel" / emotion / "noanim.png"
        large_path.parent.mkdir(parents=True, exist_ok=True)
        small_path.parent.mkdir(parents=True, exist_ok=True)
        large.save(large_path, format="PNG", optimize=True)
        small.save(small_path, format="PNG", optimize=True)

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    print(f"done. wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
