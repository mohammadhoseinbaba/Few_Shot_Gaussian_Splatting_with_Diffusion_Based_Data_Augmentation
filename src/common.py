"""Helpers shared by the pipeline scripts (frame naming, gap, test split, sampling)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
FRAME_RE = re.compile(r"frame_(\d+)")


def list_images(folder: Path) -> List[str]:
    return sorted(p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def frame_number(name: str) -> Optional[int]:
    """frame_0042.jpg -> 42; the first frame_<n> token anywhere in the name (also works for synth names)."""
    m = FRAME_RE.search(name)
    return int(m.group(1)) if m else None


def load_gap(root: Path) -> dict:
    return json.loads((root / "gap.json").read_text())


def in_gap(name: str, gap: dict) -> bool:
    idx = frame_number(name)
    return idx is not None and gap["start"] <= idx <= gap["end"]


def compute_test_names(registered_names: List[str], gap: dict, stride: int = 8) -> List[str]:
    """Held-out real test views: every `stride`-th registered real frame outside the gap.

    Single definition used by injection and by every GS condition, so all conditions
    (full-N, few-shot, +synthetic) are evaluated on exactly the same views.
    """
    candidates = sorted(n for n in registered_names if not in_gap(n, gap))
    return candidates[::stride]


def evenly_sample_names(names: List[str], k: int) -> List[str]:
    names = sorted(names)
    if k <= 0:
        return []
    if k >= len(names):
        return names.copy()
    return [names[min(int((i + 0.5) * len(names) / k), len(names) - 1)] for i in range(k)]


TEST_STRIDE = 8
FEWSHOT_MAX = 20  # the largest few-shot set; 5- and 10-shot sets are nested subsets of it
VARIANT_RE = re.compile(r"_(\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)


def load_split(root: Path, stride: int = TEST_STRIDE) -> dict:
    """Real-image split shared by every script.

    registered: frames registered in the full SfM map (written by 02_build_real_map.py)
    test:       held-out views (same for every condition)
    pool:       real training candidates for few-shot sampling (incomplete set: no gap, no test)
    """
    registered = json.loads((root / "my_scene_full" / "registered.json").read_text())
    gap = load_gap(root)
    test = compute_test_names(registered, gap, stride)
    pool = [n for n in sorted(registered) if not in_gap(n, gap) and n not in set(test)]
    return {"registered": sorted(registered), "gap": gap, "test": test, "pool": pool}


def fewshot_names(pool: List[str], n: int) -> List[str]:
    """Nested few-shot selection: 5 ⊂ 10 ⊂ 20, all evenly spread over the pool."""
    if FEWSHOT_MAX % n:
        raise ValueError(f"few-shot size must divide {FEWSHOT_MAX}, got {n}")
    return evenly_sample_names(pool, FEWSHOT_MAX)[:: FEWSHOT_MAX // n]


def variant_index(name: str) -> int:
    m = VARIANT_RE.search(name)
    return int(m.group(1)) if m else 0


def select_synth(files: List[str], source_frames: set, s: int) -> List[str]:
    """Pick s synthetic images whose source frame is in `source_frames`.

    Fills variant 0 first (evenly spread over sources), then variant 1, ... so ratios <= 100%
    use one image per source view at most and larger ratios add further variants.
    """
    cand = [f for f in files if frame_number(f) in source_frames]
    by_var: dict = {}
    for f in sorted(cand):
        by_var.setdefault(variant_index(f), []).append(f)
    out: List[str] = []
    for v in sorted(by_var):
        need = s - len(out)
        if need <= 0:
            break
        out += evenly_sample_names(by_var[v], min(need, len(by_var[v])))
    return out


def camera_to_pinhole_params(camera) -> List[float]:
    p = list(map(float, camera.params))
    if camera.model_name == "PINHOLE":
        return p[:4]
    if camera.model_name in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
        return [p[0], p[0], p[1], p[2]]  # radial distortion is ignored by GS
    if len(p) >= 4:
        return p[:4]
    raise ValueError(f"cannot convert camera model {camera.model_name} to PINHOLE")
