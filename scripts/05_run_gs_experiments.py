"""Build GS scenes for every condition, train graphdeco Gaussian Splatting, render, evaluate.

Adapted from run_gs_experiments.py of github.com/mxhrnxz/Few_Shot_Gaussian_Splatting_with_Diffusion_Based_Data_Augmentation.

Conditions
    full                      all real training frames (>=100), the full-N baseline
    fewN_real                 N real frames, no synthetic (0%)
    fewN_<strategy>_<ratio>   N real + S = round(N*ratio/100) synthetic images from 04_inject_synthetic.py
Controls shared by every condition: the SAME held-out real test views (src/common.py: load_split),
nested few-shot sets, identical GS hyper-parameters, synthetic images only on the training side.

Point-cloud init (--init-points visible, default): only 3D points seen by >= 2 *selected real training*
images are used. The original copied the full map's points into every scene, which leaks geometry from
frames that are not in the few-shot set (and from the test views' neighbours).

Results are appended to <runs>/results.jsonl (one line per condition, resumable) and summarised in
summary_results.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pycolmap

from src.common import camera_to_pinhole_params, fewshot_names, frame_number, list_images, load_split, select_synth


# ------------------------------------------------------------------ GS repo handling
def patch_gs_test_split(gs_repo: Path) -> None:
    """Stock graphdeco uses `llffhold=8` (every 8th sorted image) and NEVER reads sparse/0/test.txt.

    Setting the default to 0 makes `--eval` read test.txt. Idempotent; fails loudly if the file differs.
    """
    f = gs_repo / "scene" / "dataset_readers.py"
    src = f.read_text()
    if re.search(r"def readColmapSceneInfo\([^)]*llffhold=0\)", src):
        return
    new, n = re.subn(r"(def readColmapSceneInfo\([^)]*)llffhold=8\)", r"\1llffhold=0)", src)
    if n != 1:
        raise RuntimeError(f"cannot patch test split in {f}: signature not found; edit it manually (llffhold=0)")
    shutil.copy2(f, f.with_suffix(".py.orig"))
    f.write_text(new)
    print(f"[INFO] patched {f}: llffhold=0 so test.txt is used")


def gpu_poller(stop: threading.Event, stats: dict, interval: float = 1.0) -> None:
    peak = 0.0
    while not stop.is_set():
        try:
            out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                          stderr=subprocess.DEVNULL, text=True)
            peak = max(peak, max(float(x) for x in out.split()))
        except Exception:
            pass
        time.sleep(interval)
    stats["peak_gpu_mb"] = peak


def run(cmd: list, cwd: Path) -> None:
    cmd = [str(c) for c in cmd]
    print(">>>", " ".join(shlex.quote(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


# ------------------------------------------------------------------ scene building
def build_scene(out: Path, ref: pycolmap.Reconstruction, real_dir: Path, train_real: list, test: list,
                synth: list, synth_model: pycolmap.Reconstruction | None, synth_dir: Path | None,
                init_points: str) -> dict:
    """COLMAP text scene: real train + real test (poses from the full map) + synthetic (localized poses)."""
    shutil.rmtree(out, ignore_errors=True)
    (out / "images").mkdir(parents=True)
    sparse = out / "sparse" / "0"
    sparse.mkdir(parents=True)

    by_name = {im.name: im for im in ref.images.values()}
    entries = []  # (name, qvec wxyz, tvec, camera_id)
    for name in sorted(set(train_real) | set(test)):
        im = by_name[name]
        pose = im.cam_from_world()
        entries.append((name, list(pose.rotation.quat[[3, 0, 1, 2]]), list(pose.translation), im.camera_id))
        shutil.copy2(real_dir / name, out / "images" / name)
    if synth:
        syn_by_name = {im.name: im for im in synth_model.images.values()}
        for name in synth:
            im = syn_by_name[name]
            pose = im.cam_from_world()
            entries.append((name, list(pose.rotation.quat[[3, 0, 1, 2]]), list(pose.translation), im.camera_id))
            shutil.copy2(synth_dir / name, out / "images" / name)
    entries.sort()

    with open(sparse / "cameras.txt", "w") as f:
        for cid, cam in sorted(ref.cameras.items()):
            fx, fy, cx, cy = camera_to_pinhole_params(cam)
            f.write(f"{cid} PINHOLE {cam.width} {cam.height} {fx} {fy} {cx} {cy}\n")
    with open(sparse / "images.txt", "w") as f:
        for i, (name, q, t, cid) in enumerate(entries, start=1):
            f.write(f"{i} {' '.join(map(str, q))} {' '.join(map(str, t))} {cid} {name}\n\n")

    train_ids = {by_name[n].image_id for n in train_real}
    n_pts = 0
    with open(sparse / "points3D.txt", "w") as f:
        for pid, p in sorted(ref.points3D.items()):
            if init_points == "visible":
                if sum(1 for e in p.track.elements if e.image_id in train_ids) < 2:
                    continue
            f.write(f"{pid} {' '.join(map(str, p.xyz))} {' '.join(map(str, p.color.astype(int)))} {p.error}\n")
            n_pts += 1
    if n_pts < 500:
        print(f"[WARN] only {n_pts} initial points for {out.name}")
    (sparse / "test.txt").write_text("\n".join(test) + "\n")

    meta = {"num_real_train": len(train_real), "num_synth": len(synth), "num_test": len(test),
            "num_init_points": n_pts, "real_train": sorted(train_real), "synth": synth, "test": test}
    (out / "scene_metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


# ------------------------------------------------------------------ conditions
def make_conditions(args, split: dict, injected_files: dict) -> list:
    conds = []
    train_all = [n for n in split["registered"] if n not in set(split["test"])]
    if not args.no_full:
        conds.append({"name": "full", "n_real": len(train_all), "strategy": "none", "ratio": 0, "train_real": train_all, "synth": []})
    for n in args.fewshot:
        real = fewshot_names(split["pool"], n)
        conds.append({"name": f"few{n}_real", "n_real": n, "strategy": "none", "ratio": 0, "train_real": real, "synth": []})
        for strat in args.strategies:
            for ratio in args.ratios:
                if ratio == 0:
                    continue
                s = round(n * ratio / 100)
                files = select_synth(injected_files.get(strat, []), {frame_number(r) for r in real}, s)
                if len(files) < s:
                    print(f"[WARN] few{n}_{strat}_{ratio}: wanted {s} synthetic, only {len(files)} available")
                if files:
                    conds.append({"name": f"few{n}_{strat}_{ratio}", "n_real": n, "strategy": strat, "ratio": ratio,
                                  "train_real": real, "synth": files})
    return conds


def done_names(results: Path) -> set:
    if not results.exists():
        return set()
    return {json.loads(l)["name"] for l in results.read_text().splitlines() if l.strip()}


def write_summary(results: Path, out_csv: Path) -> None:
    rows = [json.loads(l) for l in results.read_text().splitlines() if l.strip()]
    keys = ["name", "n_real", "strategy", "ratio", "num_synth", "num_train_images", "train_time_min", "peak_gpu_mb",
            "PSNR", "SSIM", "LPIPS", "num_test", "iterations", "resolution"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/outputs"))
    ap.add_argument("--gs-repo", type=Path, required=True, help="path to graphdeco gaussian-splatting")
    ap.add_argument("--runs", type=Path, default=None, help="default <root>/gs_runs")
    ap.add_argument("--fewshot", type=int, nargs="+", default=[20], help="few-shot sizes (divisors of 20: 5 10 20)")
    ap.add_argument("--strategies", nargs="+", default=["inpaint", "outpaint", "depth", "edge"])
    ap.add_argument("--ratios", type=int, nargs="+", default=[0, 25, 50, 100], help="synthetic:real in %%; 0 = real-only")
    ap.add_argument("--no-full", action="store_true", help="skip the full-N baseline")
    ap.add_argument("--iterations", type=int, default=15000)
    ap.add_argument("--resolution", type=int, default=2, help="GS -r (1=full, 2=half, 4=quarter)")
    ap.add_argument("--init-points", choices=["visible", "full"], default="visible")
    ap.add_argument("--only", nargs="*", help="run only these condition names")
    ap.add_argument("--force", action="store_true", help="retrain conditions that already have results")
    ap.add_argument("--dry-run", action="store_true", help="list conditions and exit")
    args = ap.parse_args()

    # GS is launched from inside its own repo, so every path must be absolute
    args.root = args.root.resolve()
    args.gs_repo = args.gs_repo.resolve()
    runs = (args.runs or args.root / "gs_runs").resolve()
    scenes_root = args.root / "gs_scenes"
    results = runs / "results.jsonl"
    runs.mkdir(parents=True, exist_ok=True)

    split = load_split(args.root)
    print(f"real: {len(split['registered'])} registered, pool {len(split['pool'])}, test {len(split['test'])}")

    injected_files, injected_models = {}, {}
    for strat in args.strategies:
        d = args.root / f"{strat}_injected"
        if (d / "sparse" / "0" / "images.txt").exists():
            injected_files[strat] = sorted(n for n in list_images(d / "images") if n.startswith("synth_"))
            injected_models[strat] = d
        else:
            print(f"[WARN] {d} missing; run 04_inject_synthetic.py --strategy {strat}")

    conds = make_conditions(args, split, injected_files)
    if args.only:
        conds = [c for c in conds if c["name"] in set(args.only)]
    print(f"{len(conds)} conditions:", ", ".join(c["name"] for c in conds))
    if args.dry_run:
        return

    patch_gs_test_split(args.gs_repo)
    ref = pycolmap.Reconstruction(args.root / "my_scene_full" / "sfm_superpoint+superglue")
    real_dir = args.root / "my_scene_full" / "input"
    synth_models = {s: pycolmap.Reconstruction(d / "sparse" / "0") for s, d in injected_models.items()}

    for c in conds:
        if c["name"] in done_names(results) and not args.force:
            print(f"[SKIP] {c['name']} already done")
            continue
        print("\n" + "=" * 70 + f"\nCONDITION {c['name']}\n" + "=" * 70)
        strat = c["strategy"]
        scene = scenes_root / c["name"]
        meta = build_scene(scene, ref, real_dir, c["train_real"], split["test"], c["synth"],
                           synth_models.get(strat), injected_models.get(strat, Path()) / "images", args.init_points)
        model_dir = runs / c["name"]
        shutil.rmtree(model_dir, ignore_errors=True)

        stats: dict = {"peak_gpu_mb": 0.0}
        stop = threading.Event()
        th = threading.Thread(target=gpu_poller, args=(stop, stats), daemon=True)
        th.start()
        t0 = time.time()
        try:
            run([sys.executable, "train.py", "-s", scene, "-m", model_dir, "--eval", "-r", args.resolution,
                 "--iterations", args.iterations, "--save_iterations", args.iterations,
                 "--test_iterations", args.iterations], args.gs_repo)
        finally:
            stop.set()
            th.join()
        train_sec = time.time() - t0

        run([sys.executable, "render.py", "-m", model_dir, "--skip_train"], args.gs_repo)
        run([sys.executable, "metrics.py", "-m", model_dir], args.gs_repo)

        # Guard: GS must have evaluated exactly our test.txt views (not its own every-8th split)
        renders = sorted((model_dir / "test").glob("ours_*/renders/*.png"))
        if len(renders) != len(split["test"]):
            raise RuntimeError(f"{c['name']}: GS rendered {len(renders)} test views, expected {len(split['test'])}; "
                               "test split is not being read from test.txt")
        res = json.loads((model_dir / "results.json").read_text())
        m = next(iter(res.values()))
        row = {**{k: c[k] for k in ("name", "n_real", "strategy", "ratio")},
               "num_synth": meta["num_synth"], "num_train_images": meta["num_real_train"] + meta["num_synth"],
               "num_test": meta["num_test"], "train_time_min": round(train_sec / 60, 2),
               "peak_gpu_mb": round(stats["peak_gpu_mb"]), "PSNR": m["PSNR"], "SSIM": m["SSIM"], "LPIPS": m["LPIPS"],
               "iterations": args.iterations, "resolution": args.resolution, "init_points": args.init_points}
        with open(results, "a") as f:
            f.write(json.dumps(row) + "\n")
        write_summary(results, runs / "summary_results.csv")
        print("[RESULT]", json.dumps(row))
        shutil.rmtree(scene / "images", ignore_errors=True)  # keep disk small; scene_metadata.json lists the files

    write_summary(results, runs / "summary_results.csv")
    print(f"\nsummary: {runs / 'summary_results.csv'}")


if __name__ == "__main__":
    main()
