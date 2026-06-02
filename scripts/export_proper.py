"""
Export a generated portrait into the TW3K mod's "ProperFormat" character structure.

Target layout (per ProperFormat/ui/characters/<id>/):
  composites/large_panel/{norm,happy,angry}/<emotion>.png   RGBA, width 1089
  composites/small_panel/{norm,happy,angry}/<emotion>.png   RGBA, width 779 (= large x0.715)
  stills/<type>/<id>.png        + stills/<type>/large/<id>.png   (large = base x1.6)
    types & base sizes: mini 30x30, bobbleheads 84x138,
    halfbody_small 110x84, halfbody_large 312x250, unitcards 82x272

The single generated image is reused across all three emotion slots (the source
mod does the same). The grey/painted background is removed with rembg (the LoRA
never produces pure white, so threshold keying doesn't work). Stills are
proportional crops of the figure (head / head+chest / head-to-waist), anchored at
the top-centre of the figure's alpha bounding box.

Usage:
  python scripts/export_proper.py <generated.png> <character_id> [--out output/mod] [--model isnet-general-use]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, PngImagePlugin

ROOT = Path(__file__).resolve().parent.parent

# The mod requires a PNG "Comment" chunk on each composite giving per-expression
# pivot/offset (see Z:/TKArt/vaca/characters/check_metadata.py). This is the tool's
# REQUIRED_COMMENT default — used when the source image carries no Comment of its own.
REQUIRED_COMMENT = (
    "[type:angry;x:0;y:80;z-order:0;pivot_x:0.5000;pivot_y:0.5000;]"
    "[type:happy;x:0;y:80;z-order:0;pivot_x:0.5000;pivot_y:0.5000;]"
    "[type:norm;x:0;y:80;z-order:0;pivot_x:0.5000;pivot_y:0.5000;]"
)


def save_with_comment(img: Image.Image, path: Path, comment: str) -> None:
    meta = PngImagePlugin.PngInfo()
    meta.add_text("Comment", comment)
    img.save(path, pnginfo=meta)

COMPOSITE_SIZE = 1200       # fixed square canvas for BOTH large_panel and small_panel
FIT_H, FIT_W = 0.86, 0.92   # max fraction of the canvas the figure may fill (height, width)
EMOTIONS = ("norm", "happy", "angry")

# still type -> (base_w, base_h). The "large/" variant is round(base * 1.6).
STILL_SIZES = {
    "mini":           (30, 30),
    "bobbleheads":    (84, 138),
    "halfbody_small": (110, 84),
    "halfbody_large": (312, 250),
    "unitcards":      (82, 272),
}
# Stills are face-centred crops. Each entry is (top_units, height_units) in units of
# the detected face height `fh`: crop top = face_cy + top_units*fh, height = height_units*fh.
# Width follows the target aspect. (Top-anchoring fails on plumed helmets, so we anchor
# on the face instead.)
STILL_CROP = {
    "mini":           (-1.0, 1.8),   # square head icon
    "bobbleheads":    (-1.1, 2.0),   # head + helmet (near-mini, just a touch of collar)
    "halfbody_small": (-1.0, 2.3),   # head + shoulders
    "halfbody_large": (-1.1, 2.4),   # head + shoulders (wide framing to fill landscape canvas)
    "unitcards":      (-1.1, 5.5),   # head down to ~mid-torso, narrow
}
LARGE_MULT = 1.6

# Inner-content framing. Types listed here do NOT fill their canvas: the figure is
# scaled to fit the `content` box (aspect preserved), then placed on the full-size
# transparent still — horizontally centred and floating `bottom` px above the bottom
# edge (the source mod leaves a nameplate gap under the bobblehead). The large/ variant
# is a clean LARGE_MULT upscale of the base: `content` and `bottom` are both multiplied
# by LARGE_MULT and re-rendered from the full-res figure, so the composition is pixel-
# identical to the base, just sharper. Source-measured reference: Zhao Yun bobblehead =
# 52x75 content @ bottom 20 (small). Types absent here fill the whole canvas.
STILL_FRAME = {
    "mini":           {"content": (23, 28),   "bottom": 0},
    "bobbleheads":    {"content": (54, 74),   "bottom": 20},
    "unitcards":      {"content": (78, 222),  "bottom": 1},
}

# Face-anchored framing (the half-body stills). The source mod frames these by a consistent
# HEAD size, NOT by fitting the whole figure — so the figure is scaled so the detected face
# is `face_h` px tall, then placed with the face horizontally centred and its CENTRE at
# `face_cy` px from the top; the body runs downward (chest-up) and is clipped at the canvas
# edge. The large/ variant multiplies both by LARGE_MULT. Calibrated from SAD_artsets
# halfbody_large/faces (detected face 45px @ cy 114 in 312x250 ≈ an ~86x84 visible head);
# small derived by the canvas-height ratio (84/250 ≈ 0.336).
STILL_FACE = {
    "halfbody_large": {"face_h": 45, "face_cy": 114},
    "halfbody_small": {"face_h": 15, "face_cy": 38},
}


def detect_face(fig: Image.Image):
    """Return (cx, cy, face_h) of the main face in figure coords. Detection is
    restricted to the TOP ~45% of the figure so armour/belt details lower down can
    never become a false anchor. Falls back to a proportional head position (top,
    centred) if no face is found there — never to a lower-body detection."""
    import cv2
    import numpy as np
    W, H = fig.size
    search_h = max(1, int(H * 0.45))
    top = fig.crop((0, 0, W, search_h))
    rgb = Image.new("RGB", (W, search_h), (255, 255, 255))
    rgb.paste(top, (0, 0), top)
    gray = cv2.cvtColor(np.asarray(rgb)[:, :, ::-1], cv2.COLOR_BGR2GRAY)
    faces = []
    for name in ("haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml"):
        casc = cv2.CascadeClassifier(cv2.data.haarcascades + name)
        faces += list(casc.detectMultiScale(gray, 1.1, 4, minSize=(int(W * 0.05), int(W * 0.05))))
    if faces:
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])   # largest, all already in top region
        return (fx + fw / 2.0, fy + fh / 2.0, float(fh))
    return (W / 2.0, H * 0.13, H * 0.11)   # head near top-centre

_session = {"s": None, "name": None}


def remove_bg(img: Image.Image, model: str) -> Image.Image:
    """Background removal via rembg → clean RGBA."""
    from rembg import remove, new_session
    if _session["name"] != model:
        _session["s"] = new_session(model)
        _session["name"] = model
    out = remove(img.convert("RGB"), session=_session["s"])
    return out.convert("RGBA")


def tight_crop(rgba: Image.Image, thresh: int = 20) -> Image.Image:
    """Crop to the figure's alpha bbox, ignoring faint sub-threshold pixels that
    rembg leaves behind (those would inflate the bbox and break centering)."""
    import numpy as np
    a = np.asarray(rgba.split()[3])
    mask = a > thresh
    if not mask.any():
        return rgba
    ys, xs = np.where(mask)
    return rgba.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))


def fit_width(fig: Image.Image, w: int) -> Image.Image:
    h = max(1, round(fig.height * w / fig.width))
    return fig.resize((w, h), Image.LANCZOS)


def make_composite(fig: Image.Image, size: int = COMPOSITE_SIZE) -> Image.Image:
    """Scale the figure to fit a fixed square canvas (preserving aspect) and centre
    it both horizontally and vertically on transparency."""
    scale = min(size * FIT_W / fig.width, size * FIT_H / fig.height)
    sf = fig.resize((max(1, round(fig.width * scale)), max(1, round(fig.height * scale))),
                    Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(sf, ((size - sf.width) // 2, (size - sf.height) // 2), sf)
    return canvas


def place_in_frame(region: Image.Image, canvas_wh, content_wh, bottom: int) -> Image.Image:
    """Tight-crop `region` to the figure, scale it to fit within `content_wh` (aspect
    preserved), then paste on a transparent `canvas_wh` — horizontally centred, `bottom`
    px above the bottom edge. The figure's rendered bbox ends up ≈ content_wh, so the
    content sizes are measured against the visible figure (matching the source mod)."""
    fig = tight_crop(region)
    cw, ch = content_wh
    scale = min(cw / fig.width, ch / fig.height)
    w, h = max(1, round(fig.width * scale)), max(1, round(fig.height * scale))
    sf = fig.resize((w, h), Image.LANCZOS)
    canvas = Image.new("RGBA", canvas_wh, (0, 0, 0, 0))
    canvas.paste(sf, ((canvas_wh[0] - w) // 2, canvas_wh[1] - bottom - h), sf)
    return canvas


def place_by_face(fig: Image.Image, face, canvas_wh, face_h_out, face_cy_out) -> Image.Image:
    """Scale `fig` so its detected face is `face_h_out` px tall, then place it on a
    transparent `canvas_wh` with the face horizontally centred and its CENTRE at
    `face_cy_out` px from the top. The body runs downward (chest-up framing) and is clipped
    at the canvas edges. Used for the half-body stills, which the source mod frames by a
    consistent head size rather than by fitting the whole figure."""
    cx, cy, fh = face
    s = face_h_out / fh
    scaled = fig.resize((max(1, round(fig.width * s)), max(1, round(fig.height * s))),
                        Image.LANCZOS)
    ox = round(canvas_wh[0] / 2 - cx * s)
    oy = round(face_cy_out - cy * s)
    canvas = Image.new("RGBA", canvas_wh, (0, 0, 0, 0))
    canvas.paste(scaled, (ox, oy), scaled)
    return canvas


def crop_still(fig: Image.Image, name: str, face) -> Image.Image:
    """Face-anchored crop matching the still's target aspect. Boxes may extend past
    the figure edges; PIL fills those with transparency (matching the source margins).
    For framed types the crop matches the inner-content aspect, not the canvas aspect."""
    cx, cy, fh = face
    frame = STILL_FRAME.get(name)
    tw, th = frame["content"] if frame else STILL_SIZES[name]
    aspect = tw / th
    top_u, h_u = STILL_CROP[name]
    ch = h_u * fh
    cw = ch * aspect
    left = cx - cw / 2.0
    top = cy + top_u * fh
    return fig.crop((round(left), round(top), round(left + cw), round(top + ch)))


def export(src: Path, char_id: str, out_root: Path, model: str) -> Path:
    im = Image.open(src)
    comment = im.info.get("Comment") or REQUIRED_COMMENT   # preserve source metadata if present
    # If the input already has real transparency, use it; else remove the background.
    if im.mode == "RGBA" and im.split()[3].getextrema()[0] < 250:
        fig = tight_crop(im.convert("RGBA"))
    else:
        print(f"removing background ({model}) ...", flush=True)
        fig = tight_crop(remove_bg(im, model))
    base = out_root / "ui" / "characters" / char_id

    # composites — one centred 1200x1200 image reused for both panels and all three
    # emotion slots, each stamped with the required Comment metadata. File = noanim.png.
    comp = make_composite(fig)
    for panel in ("large_panel", "small_panel"):
        for emo in EMOTIONS:
            d = base / "composites" / panel / emo
            d.mkdir(parents=True, exist_ok=True)
            save_with_comment(comp, d / "noanim.png", comment)

    # stills — face-anchored crops at exact sizes + 1.6x large variant
    face = detect_face(fig)
    print(f"face anchor (cx,cy,fh) = {tuple(round(v, 1) for v in face)}", flush=True)
    for name, (bw, bh) in STILL_SIZES.items():
        d = base / "stills" / name
        (d / "large").mkdir(parents=True, exist_ok=True)
        lw, lh = round(bw * LARGE_MULT), round(bh * LARGE_MULT)

        fc = STILL_FACE.get(name)
        if fc:
            # face-anchored: scale to a fixed HEAD size; body runs downward (chest-up).
            # large/ = same composition with face params ×LARGE_MULT.
            place_by_face(fig, face, (bw, bh), fc["face_h"], fc["face_cy"]).save(
                d / f"{char_id}.png")
            place_by_face(fig, face, (lw, lh),
                          fc["face_h"] * LARGE_MULT, fc["face_cy"] * LARGE_MULT).save(
                d / "large" / f"{char_id}.png")
            continue

        region = crop_still(fig, name, face)
        frame = STILL_FRAME.get(name)
        if frame:
            # padded placement: figure floats in a content box, not filling the canvas.
            # large/ = the same composition scaled by LARGE_MULT (content, margin, canvas
            # all ×mult), re-rendered from the full-res figure so it stays sharp.
            cw, ch = frame["content"]
            place_in_frame(region, (bw, bh), (cw, ch), frame["bottom"]).save(
                d / f"{char_id}.png")
            place_in_frame(region, (lw, lh),
                           (round(cw * LARGE_MULT), round(ch * LARGE_MULT)),
                           round(frame["bottom"] * LARGE_MULT)).save(
                d / "large" / f"{char_id}.png")
        else:
            region.resize((bw, bh), Image.LANCZOS).save(d / f"{char_id}.png")
            region.resize((lw, lh), Image.LANCZOS).save(d / "large" / f"{char_id}.png")

    print(f"wrote {base}", flush=True)
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="an image file, OR a directory to batch-export every image in")
    ap.add_argument("char_id", nargs="?",
                    help="character id (single-file mode). Defaults to the filename stem; "
                         "in directory mode each file uses its own stem.")
    ap.add_argument("--out", default=str(ROOT / "output" / "mod"))
    ap.add_argument("--model", default="isnet-general-use",
                    help="rembg model: isnet-general-use (sharper) or u2net")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    if src.is_dir():
        exts = {".png", ".jpg", ".jpeg", ".webp"}
        imgs = sorted(p for p in src.iterdir() if p.suffix.lower() in exts)
        if not imgs:
            sys.exit(f"no images found in {src}")
        print(f"batch-exporting {len(imgs)} images from {src}")
        for p in imgs:
            export(p, p.stem, out, a.model)
    else:
        export(src, a.char_id or src.stem, out, a.model)


if __name__ == "__main__":
    main()
