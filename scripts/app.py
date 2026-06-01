"""
TW3K Portrait Generator — local Gradio UI for the trained SDXL LoRA.

Run:
    .venv/Scripts/python.exe scripts/app.py
Then open http://127.0.0.1:7860

Design notes:
- The SDXL base pipeline is loaded ONCE and kept resident in VRAM. Switching LoRA
  checkpoints only unloads/loads the (small) LoRA weights, not the 7GB base.
- The img2img upscale pipeline shares components with the txt2img pipeline, so it
  costs no extra VRAM.
- Prompt assembly mirrors the v3 caption vocabulary and the empirically-good
  generation settings from scripts/generate.py (euler_a, CFG ~6.5, full-body cues
  + anti-crop / anti-reference-sheet negatives).
"""
from __future__ import annotations

import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import gradio as gr
from PIL import Image
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    EulerAncestralDiscreteScheduler,
)

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "models" / "sd_xl_base_1.0.safetensors"
LORA_DIR = ROOT / "output" / "v3"
OUT_DIR = ROOT / "output" / "ui"
ARTSET_DIR = ROOT / "output" / "artsets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ELEMENTS = ["(none)", "wood", "fire", "water", "earth", "metal"]
ARMOR = ["unarmored", "lightly armored", "heavily armored"]
EMOTIONS = ("norm", "happy", "angry")

# Wuxing element tags ("fire wuxing", "water wuxing", ...) make SDXL render literal
# elemental MAGIC/VFX (fireballs, splashes, vines, floating rocks, sparks) instead of
# treating the element as a costume-palette cue. This big negative keeps it as colour only.
ELEMENT_NEG = (
    "fire, flames, fireball, flaming weapon, flaming sword, burning, embers, sparks, smoke, "
    "torch, lit torch, lantern glow, "
    "water, waves, water splash, splashing water, rain, raindrops, droplets, ripples, mist, wet, puddle, "
    "vines, leaves, falling leaves, foliage, tree branches, growing plants, flower petals, "
    "rocks, boulders, floating rocks, earth spikes, sand, dust, dust storm, debris, cracked ground, "
    "lightning, electricity, electric arc, metal shards, flying blades, "
    "magic, magical effects, magic circle, spell casting, casting spell, summoning, "
    "energy, energy aura, aura, glow, glowing, glowing weapon, glowing eyes, glowing runes, "
    "elemental effects, elemental magic, particles, sparkles, light rays, special effects, vfx, "
    "fog, smoke effects, swirling energy"
)
# Strong scene/background suppression — the LoRA's white-bg association is easily
# overridden by "warlord / commanding" cues, which pull in walls, throne rooms,
# battlefields. Clean white also matters for the transparent-BG masking feature.
SCENE_NEG = (
    "background, detailed background, scenery, landscape, wall, stone wall, brick wall, carved wall, "
    "architecture, building, pillars, interior, indoor, room, temple, throne room, palace, hall, "
    "environment, battlefield, outdoors, trees, sky, clouds, floor, ground, furniture, shadows on wall"
)
# Weapons off by default — the user wants unarmed figures. (Add a weapon back by
# describing it in the box and deleting the matching term here for that gen.)
WEAPON_NEG = (
    "weapon, sword, blade, longsword, saber, katana, spear, lance, polearm, halberd, guandao, glaive, "
    "axe, mace, club, war hammer, bow, crossbow, arrows, quiver, dagger, knife, staff, scepter, whip, "
    "holding weapon, holding a sword, sheathed sword, scabbard, weapon in hand"
)
DEFAULT_NEG = (
    "multiple poses, reference sheet, character sheet, multiple views, side view, back view, "
    "multiple characters, two people, group, split image, collage, "
    "low quality, blurry, jpeg artifacts, signature, text, watermark, extra limbs, deformed, "
    + SCENE_NEG + ", " + WEAPON_NEG + ", " + ELEMENT_NEG
)
FULLBODY_NEG = ("bust portrait, headshot, half body, cropped, close-up, waist up, chest up, "
                "cowboy shot, upper body, knees up, portrait crop, zoomed in")

# ---------------------------------------------------------------- pipeline cache
_S = {"pipe": None, "img2img": None, "lora": None, "meta": {}}


def _load_base():
    if _S["pipe"] is None:
        print("loading SDXL base (one-time) ...", flush=True)
        pipe = StableDiffusionXLPipeline.from_single_file(
            str(BASE), torch_dtype=torch.float16, use_safetensors=True, add_watermarker=False
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.to("cuda")
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
        _S["pipe"] = pipe
        _S["img2img"] = StableDiffusionXLImg2ImgPipeline(**pipe.components)
    return _S["pipe"]


def _ensure_lora(name: str):
    pipe = _load_base()
    if _S["lora"] != name:
        if _S["lora"] is not None:
            pipe.unload_lora_weights()
        pipe.load_lora_weights(str(LORA_DIR), weight_name=name)
        _S["lora"] = name
    return pipe


def list_loras():
    files = sorted(p.name for p in LORA_DIR.glob("*.safetensors"))
    choices = []
    for f in files:
        if f.endswith("-000002.safetensors"):
            lbl = "epoch 2"
        elif "-0000" in f:
            ep = int(f.split("-0000")[1].split(".")[0])
            lbl = f"epoch {ep}"
        else:
            lbl = "epoch 12 (final)"
        choices.append((lbl, f))
    # order by epoch
    choices.sort(key=lambda c: (999 if "final" in c[0] else int(c[0].split()[1])))
    return choices


def build_prompt(gender, element, armor, fullbody, desc):
    parts = ["tk3k_portrait"]
    if fullbody:
        parts += ["solo", "(full body:1.5)", "(full length:1.4)", "head to toe", "entire figure visible",
                  "standing", "feet visible", "full figure small in frame"]
    parts.append("man" if gender == "man" else "woman")
    if element and element != "(none)":
        parts.append(f"{element} wuxing")
    parts.append(armor)
    if desc and desc.strip():
        parts.append(desc.strip().strip(","))
    parts += ["full body portrait", "(plain white background:1.4)", "simple background", "isolated on white background"]
    return ", ".join(parts)


def whiteness_alpha(img_rgb: Image.Image, fade_start: int = 235, fade_end: int = 252) -> Image.Image:
    arr = np.asarray(img_rgb.convert("RGB"), dtype=np.uint8)
    mn = arr.min(axis=2)
    span = max(fade_end - fade_start, 1)
    a = np.clip(255 - ((mn - fade_start) * 255 // span), 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([arr, a]), "RGBA")


# ---------------------------------------------------------------- actions
def _remote_generate(url, lora_name, lora_w, gender, element, armor, fullbody, desc, negative, steps, cfg, n_images, seed):
    """Forward the generation to a RunPod pod running this same app, then pull the
    images back. We force compute='Local' on the remote call so the pod renders
    locally (no recursion)."""
    if not url or not url.strip():
        raise gr.Error("Enter the RunPod URL, or switch Compute back to Local.")
    import shutil
    from gradio_client import Client
    rc = Client(url.strip())
    res = rc.predict(lora_name, float(lora_w), gender, element, armor, bool(fullbody), desc, negative,
                     int(steps), float(cfg), int(n_images), int(seed), "Local", "", api_name="/generate")
    gallery_remote, info = (res[0], res[1]) if isinstance(res, (list, tuple)) else (res, "")
    prompt = build_prompt(gender, element, armor, fullbody, desc)
    ts = time.strftime("%Y%m%d_%H%M%S")
    gallery, paths = [], []
    for i, item in enumerate(gallery_remote or []):
        if isinstance(item, dict):
            src, cap = item.get("image") or item.get("path"), item.get("caption")
        elif isinstance(item, (list, tuple)):
            src, cap = item[0], (item[1] if len(item) > 1 else None)
        else:
            src, cap = item, None
        dst = OUT_DIR / f"remote_{ts}_{i}.png"
        try:
            shutil.copy(src, dst)
        except Exception:
            continue
        _S["meta"][str(dst)] = {"prompt": prompt, "seed": None, "lora": lora_name}
        gallery.append((str(dst), cap or f"remote {i}"))
        paths.append(str(dst))
    return gallery, paths, "[RunPod] " + (info or "")


def generate(lora_name, lora_w, gender, element, armor, fullbody, desc, negative, steps, cfg, n_images, seed,
             compute="Local", runpod_url=""):
    if not lora_name:
        raise gr.Error("Pick a LoRA checkpoint first.")
    if compute == "RunPod":
        return _remote_generate(runpod_url, lora_name, lora_w, gender, element, armor, fullbody, desc,
                                negative, steps, cfg, n_images, seed)
    pipe = _ensure_lora(lora_name)
    prompt = build_prompt(gender, element, armor, fullbody, desc)
    neg = negative or ""
    if fullbody and FULLBODY_NEG not in neg:
        neg = (neg + ", " + FULLBODY_NEG).strip(", ")

    n = int(n_images)
    if seed is None or int(seed) < 0:
        base_seed = random.randint(0, 2**31 - 1)
    else:
        base_seed = int(seed)
    seeds = [base_seed + i for i in range(n)]

    # Portrait aspect when full-body is on — the single biggest lever for getting a
    # head-to-toe figure instead of a 3/4 crop. Square only when full-body is off.
    W, H = (832, 1216) if fullbody else (1024, 1024)
    ts = time.strftime("%Y%m%d_%H%M%S")
    gallery, paths = [], []
    for i, s in enumerate(seeds):
        g = torch.Generator(device="cuda").manual_seed(s)
        img = pipe(
            prompt=prompt, negative_prompt=neg,
            num_inference_steps=int(steps), guidance_scale=float(cfg),
            width=W, height=H, generator=g,
            cross_attention_kwargs={"scale": float(lora_w)},
        ).images[0]
        p = OUT_DIR / f"{ts}_s{s}.png"
        img.save(p)
        _S["meta"][str(p)] = {"prompt": prompt, "seed": s, "lora": lora_name}
        gallery.append((str(p), f"seed {s}"))
        paths.append(str(p))

    info = f"PROMPT:\n{prompt}\n\nNEGATIVE:\n{neg}\n\nseeds: {seeds}  |  checkpoint: {lora_name}"
    return gallery, paths, info


def on_select(paths, evt: gr.SelectData):
    if not paths:
        return None, "—"
    sel = paths[evt.index]
    return sel, f"selected: {Path(sel).name}"


def do_upscale(sel_path, lora_name, lora_w, target):
    if not sel_path:
        raise gr.Error("Select a result in the gallery first.")
    _ensure_lora(lora_name)
    img2img = _S["img2img"]
    meta = _S["meta"].get(sel_path, {})
    prompt = meta.get("prompt", "tk3k_portrait, full body portrait, white background")
    src = Image.open(sel_path).convert("RGB")
    sc = int(target) / max(src.size)  # preserve aspect; scale longest side to target
    im = src.resize((max(8, round(src.width * sc) // 8 * 8), max(8, round(src.height * sc) // 8 * 8)), Image.LANCZOS)
    g = torch.Generator(device="cuda").manual_seed(12345)
    out = img2img(
        prompt=prompt,
        negative_prompt="low quality, blurry, jpeg artifacts, deformed, extra limbs, signature, text, watermark",
        image=im, strength=0.35, num_inference_steps=40, guidance_scale=6.5, generator=g,
        cross_attention_kwargs={"scale": float(lora_w)},
    ).images[0]
    p = Path(sel_path).with_name(Path(sel_path).stem + f"_up{int(target)}.png")
    out.save(p)
    return out, f"upscaled -> {p.name} ({out.size[0]}x{out.size[1]})"


def do_transparent(sel_path):
    if not sel_path:
        raise gr.Error("Select a result in the gallery first.")
    import export_proper as EP
    rgba = EP.tight_crop(EP.remove_bg(Image.open(sel_path), "isnet-general-use"))
    p = Path(sel_path).with_name(Path(sel_path).stem + "_rgba.png")
    rgba.save(p)
    return rgba, f"transparent BG (rembg) -> {p.name}"


def do_export_proper(sel_path, char_id):
    """Export the selected image into the mod's ProperFormat character structure
    (composites + stills) under output/mod/. Backgrounds removed via rembg."""
    if not sel_path:
        raise gr.Error("Select a result in the gallery first.")
    if not char_id or not char_id.strip():
        raise gr.Error("Enter a character ID, e.g. ep_hero_special_metal_yeonbul.")
    import export_proper as EP
    out = EP.export(Path(sel_path), char_id.strip(), ROOT / "output" / "mod", "isnet-general-use")
    return f"exported ProperFormat → {out}"


# ---------------------------------------------------------------- UI
with gr.Blocks(title="TW3K Portrait Generator") as demo:
    gr.Markdown("## TW3K Portrait Generator  ·  `tk3k_portrait` LoRA")
    sel_state = gr.State(None)
    paths_state = gr.State([])

    with gr.Row():
        with gr.Column(scale=1):
            lora_dd = gr.Dropdown(list_loras(), value="tk3k_portrait_v3.safetensors",
                                  label="LoRA checkpoint")
            with gr.Row():
                compute = gr.Radio(["Local", "RunPod"], value="Local", label="Compute", scale=1)
                runpod_url = gr.Textbox(label="RunPod URL", scale=2,
                                        placeholder="https://<podid>-7860.proxy.runpod.net")
            with gr.Row():
                gender = gr.Radio(["man", "woman"], value="man", label="Gender")
                element = gr.Dropdown(ELEMENTS, value="(none)", label="Wuxing element")
            armor = gr.Radio(ARMOR, value="lightly armored", label="Armor")
            fullbody = gr.Checkbox(value=True, label="Force full body (framing cues + anti-crop)")
            desc = gr.Textbox(label="Description (free tags)", lines=3,
                              placeholder="flowing silk scholar robe, official's cap, holding a scroll, dignified expression")
            with gr.Accordion("Advanced", open=False):
                negative = gr.Textbox(value=DEFAULT_NEG, label="Negative prompt", lines=3)
                lora_w = gr.Slider(0.0, 1.5, value=1.0, step=0.05, label="LoRA weight")
                steps = gr.Slider(10, 50, value=30, step=1, label="Steps")
                cfg = gr.Slider(1.0, 12.0, value=6.5, step=0.5, label="CFG")
            with gr.Row():
                n_images = gr.Slider(1, 8, value=4, step=1, label="Images (seeds)")
                seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)
            gen_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=2):
            gallery = gr.Gallery(label="Results (click one to select)", columns=4, height=460,
                                 object_fit="contain")
            info = gr.Textbox(label="Prompt / seeds", lines=4, interactive=False)
            sel_label = gr.Markdown("selected: —")
            with gr.Tab("Upscale"):
                with gr.Row():
                    up_target = gr.Dropdown([1280, 1536, 1920], value=1536, label="Target px")
                    up_btn = gr.Button("Upscale selected")
                up_out = gr.Image(label="Upscaled", height=460)
            with gr.Tab("Transparent BG"):
                tr_btn = gr.Button("Strip white background")
                tr_out = gr.Image(label="Transparent (RGBA)", height=460, image_mode="RGBA")
            with gr.Tab("Export (ProperFormat)"):
                char_id = gr.Textbox(label="Character ID",
                                     placeholder="ep_hero_special_metal_yeonbul")
                art_btn = gr.Button("Export selected result to mod ProperFormat")
                art_status = gr.Markdown()
            with gr.Tab("Export existing image"):
                gr.Markdown("Upload any existing image (generated earlier, or external) and export it.")
                up_img = gr.Image(type="filepath", label="Image to export", height=300)
                up_char_id = gr.Textbox(label="Character ID",
                                        placeholder="ep_hero_special_metal_yeonbul")
                up_export_btn = gr.Button("Export uploaded image to ProperFormat")
                up_export_status = gr.Markdown()

    gen_btn.click(generate,
                  [lora_dd, lora_w, gender, element, armor, fullbody, desc, negative, steps, cfg, n_images, seed,
                   compute, runpod_url],
                  [gallery, paths_state, info], api_name="generate")
    gallery.select(on_select, [paths_state], [sel_state, sel_label])
    up_btn.click(do_upscale, [sel_state, lora_dd, lora_w, up_target], [up_out, sel_label])
    tr_btn.click(do_transparent, [sel_state], [tr_out, sel_label])
    art_btn.click(do_export_proper, [sel_state, char_id], [art_status])
    up_export_btn.click(do_export_proper, [up_img, up_char_id], [up_export_status])


if __name__ == "__main__":
    # On the RunPod pod, set GRADIO_SERVER_NAME=0.0.0.0 so the proxy can reach it.
    demo.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
                server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
                inbrowser=False, show_error=True, theme=gr.themes.Soft())
