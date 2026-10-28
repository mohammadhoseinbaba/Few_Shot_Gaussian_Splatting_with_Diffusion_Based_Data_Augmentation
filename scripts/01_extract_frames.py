"""Extract frames from a video, drop blurry ones, and build the full / gapped scenes.

Outputs (under --out):
    my_scene_full/input/frame_XXXX.jpg   all kept frames (reference set for the SfM map)
    my_scene/input/frame_XXXX.jpg        same names, minus the middle gap block
    boundary/before.jpg, after.jpg       frames just outside the gap (video-interpolation input)
    gap.json                             {start, end, before_frame, after_frame, ...}

Runs on a weak laptop: only needs ffmpeg/ffprobe. The blur filter additionally
needs numpy + pillow and is skipped with a warning if they are missing.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def probe_duration(video: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(video)],
        text=True,
    )
    return float(out.strip())


def sharpness_scores(paths: list[Path]) -> list[float] | None:
    """Variance of a 4-neighbour Laplacian on a downscaled grayscale image."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return None
    scores = []
    for p in paths:
        g = np.asarray(Image.open(p).convert("L").resize((640, 360)), dtype=np.float32)
        lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
        scores.append(float(lap.var()))
    return scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="scene root, e.g. data/outputs")
    ap.add_argument("--target-frames", type=int, default=200, help="frames sampled uniformly before the blur filter")
    ap.add_argument("--width", type=int, default=1280, help="output width (height keeps the aspect ratio)")
    ap.add_argument("--blur-drop", type=float, default=0.10, help="fraction of blurriest frames to drop (0 disables)")
    ap.add_argument("--gap-fraction", type=float, default=1 / 3, help="fraction of frames removed from the middle")
    ap.add_argument("--force", action="store_true", help="overwrite existing output")
    args = ap.parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            sys.exit(f"{tool} not found on PATH")

    full_dir = args.out / "my_scene_full" / "input"
    gap_dir = args.out / "my_scene" / "input"
    boundary_dir = args.out / "boundary"
    if full_dir.exists() and any(full_dir.iterdir()) and not args.force:
        sys.exit(f"{full_dir} is not empty; pass --force to overwrite")
    for d in (args.out / "my_scene_full", args.out / "my_scene", boundary_dir):
        shutil.rmtree(d, ignore_errors=True)

    duration = probe_duration(args.video)
    fps = args.target_frames / duration
    print(f"duration {duration:.1f}s -> sampling at {fps:.3f} fps for ~{args.target_frames} frames")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", str(args.video),
             "-vf", f"fps={fps},scale={args.width}:-2", "-q:v", "2", str(tmp / "raw_%04d.jpg")],
            check=True,
        )
        raw = sorted(tmp.glob("raw_*.jpg"))
        print(f"extracted {len(raw)} frames")

        keep = raw
        if args.blur_drop > 0:
            scores = sharpness_scores(raw)
            if scores is None:
                print("[WARN] numpy/pillow missing: skipping blur filter (pip install numpy pillow)")
            else:
                n_drop = int(len(raw) * args.blur_drop)
                worst = set(sorted(range(len(raw)), key=lambda i: scores[i])[:n_drop])
                keep = [p for i, p in enumerate(raw) if i not in worst]
                print(f"blur filter dropped {n_drop} frames")

        full_dir.mkdir(parents=True)
        names = []
        for i, p in enumerate(keep, start=1):  # renumber so names stay contiguous
            name = f"frame_{i:04d}.jpg"
            shutil.copy2(p, full_dir / name)
            names.append(name)

    n = len(names)
    gap_len = max(1, round(n * args.gap_fraction))
    start = (n - gap_len) // 2 + 1  # 1-based frame numbers, inclusive
    end = start + gap_len - 1
    if start < 2 or end > n - 1:
        sys.exit("gap does not leave boundary frames; lower --gap-fraction")

    gap_dir.mkdir(parents=True)
    for i, name in enumerate(names, start=1):
        if not (start <= i <= end):
            shutil.copy2(full_dir / name, gap_dir / name)

    boundary_dir.mkdir(parents=True)
    before, after = f"frame_{start - 1:04d}.jpg", f"frame_{end + 1:04d}.jpg"
    shutil.copy2(full_dir / before, boundary_dir / "before.jpg")
    shutil.copy2(full_dir / after, boundary_dir / "after.jpg")

    meta = {
        "video": str(args.video), "duration_sec": duration, "sample_fps": fps,
        "num_frames_full": n, "num_frames_gapped": n - gap_len,
        "start": start, "end": end,
        "before_frame": before, "after_frame": after,
    }
    (args.out / "gap.json").write_text(json.dumps(meta, indent=2))
    print(f"full set: {n} frames; gap {start}-{end} removed ({gap_len}); kept {n - gap_len}")
    if n < 100:
        print("[WARN] fewer than 100 frames: the brief asks for at least 100 for the full baseline")


if __name__ == "__main__":
    main()
