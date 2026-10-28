"""Generate synthetic views with open diffusion models, ONLY from the few-shot real images.

Strategies (the ablation axis; all Stable Diffusion 1.5 based, fp16, fit a 16 GB T4):
    inpaint   random regions of a real frame are repainted (occlusion / missing-region filling)
    outpaint  the frame is panned sideways and the newly revealed strip is painted (field-of-view extension)
    depth     ControlNet-depth (Depth-Anything-V2 depth) of a slightly panned frame -> nearby viewpoint
    edge      ControlNet-canny edges of a slightly panned frame -> nearby viewpoint

Sources are the 20-shot real set (src/common.py: fewshot_names), never test or gap frames.
Each source gets --variants images (variant 0 pans right, variant 1 pans left), so S = 2*N covers the
200% ratio for every few-shot size (5/10/20 use nested source subsets).

Output: <root>/synth/<strategy>/input/<strategy>_frame_XXXX_<variant>.jpg  (+ index.json)
The name embeds the source real frame, which 04_inject_synthetic.py uses for the leakage guard.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image, ImageFilter

from src.common import FEWSHOT_MAX, fewshot_names, frame_number, load_split

SD_INPAINT = "stable-diffusion-v1-5/stable-diffusion-inpainting"
SD_BASE = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CN_CANNY = "lllyasviel/control_v11p_sd15_canny"
CN_DEPTH = "lllyasviel/control_v11f1p_sd15_depth"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
NEGATIVE = "blurry, distorted, deformed, text, watermark, low quality"


def fit(img: Image.Image, long_side: int) -> Image.Image:
    """Resize to a multiple-of-8 size with the same aspect ratio (SD works best near 512-768)."""
    s = long_side / max(img.size)
    w, h = (round(img.width * s / 8) * 8, round(img.height * s / 8) * 8)
    return img.resize((w, h), Image.LANCZOS)


def pan(img: Image.Image, dx: int) -> tuple[Image.Image, Image.Image]:
    """Shift image by dx px (edge-replicated); returns (shifted, mask of the revealed strip)."""
    a = np.asarray(img)
    w = a.shape[1]
    if dx > 0:  # content moves right, strip revealed on the left
        out = np.pad(a, ((0, 0), (dx, 0), (0, 0)), mode="edge")[:, :w]
        m = np.zeros(a.shape[:2], np.uint8); m[:, : dx + 8] = 255
    else:
        out = np.pad(a, ((0, 0), (0, -dx), (0, 0)), mode="edge")[:, -dx:]
        m = np.zeros(a.shape[:2], np.uint8); m[:, w + dx - 8 :] = 255
    return Image.fromarray(out), Image.fromarray(m)


def random_mask(size: tuple[int, int], rng: np.random.Generator, coverage: float = 0.3) -> Image.Image:
    w, h = size
    m = np.zeros((h, w), np.uint8)
    while m.mean() / 255 < coverage:
        bw, bh = int(w * rng.uniform(0.15, 0.4)), int(h * rng.uniform(0.15, 0.4))
        x, y = int(rng.integers(0, w - bw)), int(rng.integers(0, h - bh))
        m[y : y + bh, x : x + bw] = 255
    return Image.fromarray(m)


def composite(orig: Image.Image, gen: Image.Image, mask: Image.Image) -> Image.Image:
    """Keep the real pixels outside the mask (the VAE round trip would otherwise shift colours)."""
    soft = mask.filter(ImageFilter.GaussianBlur(4))
    return Image.composite(gen.resize(orig.size), orig, soft)


class Generator:
    def __init__(self, strategy: str, prompt: str, steps: int, guidance: float, long_side: int):
        from diffusers import (ControlNetModel, StableDiffusionControlNetPipeline,
                               StableDiffusionInpaintPipeline)
        self.strategy, self.prompt, self.steps, self.guidance, self.long_side = strategy, prompt, steps, guidance, long_side
        dt = torch.float16
        if strategy in {"inpaint", "outpaint"}:
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(SD_INPAINT, torch_dtype=dt)
        else:
            cn = ControlNetModel.from_pretrained(CN_CANNY if strategy == "edge" else CN_DEPTH, torch_dtype=dt)
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(SD_BASE, controlnet=cn, torch_dtype=dt)
            if strategy == "depth":
                from transformers import pipeline
                self.depth = pipeline("depth-estimation", model=DEPTH_MODEL, device=0)
        self.pipe.safety_checker = None
        self.pipe.to("cuda")
        self.pipe.enable_attention_slicing()

    def control_image(self, img: Image.Image) -> Image.Image:
        if self.strategy == "edge":
            import cv2
            e = cv2.Canny(np.asarray(img.convert("L")), 100, 200)
            return Image.fromarray(np.stack([e] * 3, -1))
        d = self.depth(img)["depth"].resize(img.size)
        return Image.fromarray(np.stack([np.asarray(d.convert("L"))] * 3, -1))

    def __call__(self, real: Image.Image, variant: int, seed: int) -> tuple[Image.Image, dict]:
        img = fit(real.convert("RGB"), self.long_side)
        g = torch.Generator("cuda").manual_seed(seed)
        sign = 1 if variant % 2 == 0 else -1
        kw = dict(prompt=self.prompt, negative_prompt=NEGATIVE, num_inference_steps=self.steps,
                  guidance_scale=self.guidance, generator=g, width=img.width, height=img.height)
        if self.strategy == "inpaint":
            mask = random_mask(img.size, np.random.default_rng(seed))
            out = self.pipe(image=img, mask_image=mask, **kw).images[0]
            return composite(img, out, mask), {"mask_coverage": float(np.asarray(mask).mean() / 255)}
        if self.strategy == "outpaint":
            dx = sign * int(img.width * 0.15)
            shifted, mask = pan(img, dx)
            out = self.pipe(image=shifted, mask_image=mask, **kw).images[0]
            return composite(shifted, out, mask), {"pan_px": dx}
        dx = sign * int(img.width * 0.06)  # small pan = nearby viewpoint
        shifted, _ = pan(img, dx)
        out = self.pipe(image=self.control_image(shifted), **kw).images[0]
        return out, {"pan_px": dx}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/outputs"))
    ap.add_argument("--strategy", nargs="+", required=True, choices=["inpaint", "outpaint", "depth", "edge"])
    ap.add_argument("--prompt", default="a realistic photo of an indoor room, sharp focus, natural lighting",
                    help="describe YOUR scene for better consistency")
    ap.add_argument("--variants", type=int, default=2)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--long-side", type=int, default=768, help="generation resolution (04 resizes to the map size)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    split = load_split(args.root)
    sources = fewshot_names(split["pool"], FEWSHOT_MAX)
    assert not set(sources) & set(split["test"]), "test frame among generation sources"
    print(f"{len(sources)} source frames (few-shot set): {sources[0]} ... {sources[-1]}")

    for strategy in args.strategy:
        out_dir = args.root / "synth" / strategy / "input"
        out_dir.mkdir(parents=True, exist_ok=True)
        gen = Generator(strategy, args.prompt, args.steps, args.guidance, args.long_side)
        index = {}
        for src in sources:
            real = Image.open(args.root / "my_scene_full" / "input" / src)
            for v in range(args.variants):
                name = f"{strategy}_{Path(src).stem}_{v:02d}.jpg"
                img, info = gen(real, v, args.seed * 100000 + frame_number(src) * 10 + v)
                img.save(out_dir / name, quality=95)
                index[name] = {"source": src, "variant": v, "strategy": strategy, **info}
            print(f"[{strategy}] {src}")
        (out_dir.parent / "index.json").write_text(json.dumps(
            {"prompt": args.prompt, "steps": args.steps, "guidance": args.guidance, "seed": args.seed,
             "images": index}, indent=2))
        del gen
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
