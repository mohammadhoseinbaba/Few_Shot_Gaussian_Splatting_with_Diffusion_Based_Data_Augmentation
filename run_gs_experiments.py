from __future__ import annotations

import csv
import json
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pycolmap

# ============================================================
# USER CONFIG
# ============================================================
PROJECT_ROOT = Path("/mnt/d/project_root")
GS_REPO = PROJECT_ROOT / "gaussian-splatting"
DATASETS_ROOT = PROJECT_ROOT / "GaussianSplatting" / "outputs"
RUNS_ROOT = PROJECT_ROOT / "GaussianSplatting" / "gs_runs"

PYTHON_EXE = sys.executable
TRAIN_RESOLUTION = 1
DATA_DEVICE = "cuda"
ITERATIONS = 30000
SAVE_ITERS = 30000
TEST_ITERS = 30000
USE_EVAL_SPLIT = True

FORCE_RETRAIN = True
FORCE_RENDER = False
SKIP_MISSING_SCENES = True

# ============================================================
# FEW-SHOT DESIGN
# ============================================================
# This script does FEW-SHOT ONLY.
# There is NO full all-frames baseline here.
#
# We build:
#   - fewshot20_real
#   - fewshot20_modelA_25syn
#   - fewshot20_modelA_50syn
#   - fewshot20_modelA_100syn
#   - fewshot20_modelA_200syn
#   - ... same for modelB/C/D
#
# using your CURRENT injected folders.
#
# The common 20 real train images are selected once from FEWSHOT_REAL_SOURCE
# and reused across all few-shot conditions for fairness.
# ============================================================
FEWSHOT_N = 20

MODELS = ["modelA", "modelB", "modelC", "modelD"]
SYN_RATIOS = [25, 50, 100, 200]

# Use one already-built injected dataset as the common source for:
# - real train pool
# - real-only test set
# This should be a dataset that already exists.
FEWSHOT_REAL_SOURCE = DATASETS_ROOT / "modelA_25syn_injected"

# Generated temporary GS-ready few-shot scenes will be stored here
GENERATED_FEWSHOT_ROOT = DATASETS_ROOT / f"fewshot{FEWSHOT_N}_generated"

valid_exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


# ============================================================
# HELPERS
# ============================================================
def run_cmd(cmd: List[str], cwd: Path) -> None:
    print("\n>>>", " ".join(shlex.quote(str(x)) for x in cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def ensure_gs_ready_scene(scene_dir: Path) -> None:
    """
    Make the scene folder look GS-ready.
    If images/ is missing but input/ exists, create a symlink images -> input.
    """
    images_dir = scene_dir / "images"
    input_dir = scene_dir / "input"

    if not images_dir.exists():
        if input_dir.exists():
            try:
                images_dir.symlink_to(input_dir, target_is_directory=True)
                print(f"[INFO] Created symlink: {images_dir} -> {input_dir}")
            except FileExistsError:
                pass
        else:
            raise FileNotFoundError(
                f"Scene {scene_dir} has neither images/ nor input/."
            )

    sparse0 = scene_dir / "sparse" / "0"
    if not sparse0.exists():
        raise FileNotFoundError(f"Missing sparse/0 in {scene_dir}")


def get_images_dir(scene_dir: Path) -> Path:
    ensure_gs_ready_scene(scene_dir)
    images_dir = scene_dir / "images"
    if images_dir.exists():
        return images_dir
    return scene_dir / "input"


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix in valid_exts)


def read_test_names(scene_dir: Path) -> List[str]:
    test_file = scene_dir / "sparse" / "0" / "test.txt"
    if not test_file.exists():
        return []
    with open(test_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


def poll_gpu_memory(stop_event: threading.Event, stats: Dict[str, float], interval_sec: float = 2.0) -> None:
    max_used = 0.0
    while not stop_event.is_set():
        try:
            result = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip().splitlines()
            if result:
                used = max(float(x.strip()) for x in result if x.strip())
                max_used = max(max_used, used)
        except Exception:
            pass
        time.sleep(interval_sec)
    stats["max_gpu_mem_mb"] = max_used


def parse_metrics_json(results_json: Path) -> Dict[str, Optional[float]]:
    out = {"SSIM": None, "PSNR": None, "LPIPS": None}
    if not results_json.exists():
        return out

    with open(results_json, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "SSIM" in data or "PSNR" in data or "LPIPS" in data:
            out["SSIM"] = data.get("SSIM")
            out["PSNR"] = data.get("PSNR")
            out["LPIPS"] = data.get("LPIPS")
            return out

        for _, v in data.items():
            if isinstance(v, dict) and ("SSIM" in v or "PSNR" in v or "LPIPS" in v):
                out["SSIM"] = v.get("SSIM")
                out["PSNR"] = v.get("PSNR")
                out["LPIPS"] = v.get("LPIPS")
                return out

    return out


def save_summary(rows: List[Dict], out_json: Path, out_csv: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)

    if rows:
        keys = [
            "name",
            "scene",
            "num_images",
            "train_time_sec",
            "train_time_min",
            "max_gpu_mem_mb",
            "SSIM",
            "PSNR",
            "LPIPS",
            "model_path",
        ]
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k) for k in keys})


def latest_saved_iteration(model_dir: Path) -> Optional[int]:
    point_cloud_dir = model_dir / "point_cloud"
    if not point_cloud_dir.exists():
        return None

    iterations = []
    for p in point_cloud_dir.glob("iteration_*"):
        try:
            iterations.append(int(p.name.replace("iteration_", "")))
        except ValueError:
            pass

    if not iterations:
        return None
    return max(iterations)


def evenly_sample_names(names: List[str], k: int) -> List[str]:
    names = sorted(names)
    if k <= 0:
        return []
    if k >= len(names):
        return names.copy()

    idxs = []
    for i in range(k):
        idx = int((i + 0.5) * len(names) / k)
        idx = min(idx, len(names) - 1)
        idxs.append(idx)
    return [names[i] for i in idxs]


def split_real_and_synth_names(scene_dir: Path) -> Tuple[List[str], List[str], List[str]]:
    images_dir = get_images_dir(scene_dir)
    all_names = sorted(
        p.name for p in images_dir.iterdir()
        if p.is_file() and p.suffix in valid_exts
    )
    test_names = read_test_names(scene_dir)
    real_names = [n for n in all_names if not n.startswith("synth_")]
    synth_names = [n for n in all_names if n.startswith("synth_")]
    real_train_names = [n for n in real_names if n not in test_names]
    return real_train_names, synth_names, test_names


def camera_to_supported_model(camera) -> Tuple[str, List[float]]:
    params = list(map(float, camera.params))

    if camera.model_name == "PINHOLE":
        return "PINHOLE", params[:4]

    if camera.model_name == "SIMPLE_PINHOLE":
        return "SIMPLE_PINHOLE", params[:3]

    if camera.model_name in {"SIMPLE_RADIAL", "RADIAL"}:
        f, cx, cy = params[:3]
        return "SIMPLE_PINHOLE", [f, cx, cy]

    if len(params) >= 4:
        return "PINHOLE", params[:4]

    raise ValueError(f"Unsupported camera model: {camera.model_name}")


def build_fewshot_gs_scene(
    source_scene: Path,
    out_scene: Path,
    kept_real_train_names: List[str],
    num_synth_keep: int,
) -> Path:
    """
    Build a new GS-ready few-shot dataset from an existing GS-ready scene.

    Keeps:
    - provided real training names
    - sampled subset of synthetic images
    - original real-only test set
    """
    ensure_gs_ready_scene(source_scene)
    source_images_dir = get_images_dir(source_scene)
    source_sparse = source_scene / "sparse" / "0"

    reconstruction = pycolmap.Reconstruction(source_sparse)

    real_train_names, synth_names, test_names = split_real_and_synth_names(source_scene)

    kept_real_train_names = [n for n in kept_real_train_names if n in real_train_names]
    if len(kept_real_train_names) < FEWSHOT_N:
        print(
            f"[WARNING] Source scene {source_scene.name} has only "
            f"{len(kept_real_train_names)} of the requested {FEWSHOT_N} real train images."
        )

    kept_synth_names = evenly_sample_names(synth_names, num_synth_keep)

    selected_names = set(kept_real_train_names) | set(kept_synth_names) | set(test_names)

    out_images_dir = out_scene / "images"
    out_sparse_dir = out_scene / "sparse" / "0"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_sparse_dir.mkdir(parents=True, exist_ok=True)

    # Clean previous outputs
    for p in out_images_dir.iterdir():
        if p.is_file() and p.suffix in valid_exts:
            p.unlink()

    for p in out_sparse_dir.glob("*"):
        if p.is_file():
            p.unlink()

    # Copy selected images
    for name in sorted(selected_names):
        src = source_images_dir / name
        if src.exists():
            shutil.copy2(src, out_images_dir / name)

    # Write cameras.txt
    camera_id_map: Dict[int, int] = {}
    camera_lines: List[str] = []
    next_camera_id = 1

    selected_images = []
    for image in sorted(reconstruction.images.values(), key=lambda im: im.name):
        if image.name not in selected_names:
            continue
        selected_images.append(image)

        if image.camera_id not in camera_id_map:
            camera = reconstruction.cameras[image.camera_id]
            model_name, params = camera_to_supported_model(camera)
            camera_id_map[image.camera_id] = next_camera_id
            params_str = " ".join(map(str, params))
            camera_lines.append(
                f"{next_camera_id} {model_name} {camera.width} {camera.height} {params_str}"
            )
            next_camera_id += 1

    with open(out_sparse_dir / "cameras.txt", "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("\n".join(camera_lines) + "\n")

    # Write images.txt
    image_lines = []
    image_id = 1
    for image in selected_images:
        pose = image.cam_from_world()
        q = pose.rotation.quat[[3, 0, 1, 2]]
        t = pose.translation
        camera_id = camera_id_map[image.camera_id]
        image_lines.append(
            f"{image_id} {' '.join(map(str, q))} {' '.join(map(str, t))} {camera_id} {image.name}"
        )
        image_id += 1

    with open(out_sparse_dir / "images.txt", "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for line in image_lines:
            f.write(line + "\n\n")

    # Write points3D.txt
    with open(out_sparse_dir / "points3D.txt", "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for point3D_id, point in sorted(reconstruction.points3D.items()):
            xyz = " ".join(map(str, point.xyz))
            rgb = " ".join(map(str, point.color.astype(int)))
            f.write(f"{point3D_id} {xyz} {rgb} {point.error}\n")

    # Real-only test set
    with open(out_sparse_dir / "test.txt", "w") as f:
        f.write("\n".join(test_names) + "\n")

    # Metadata
    with open(out_scene / "fewshot_metadata.json", "w") as f:
        json.dump(
            {
                "source_scene": str(source_scene),
                "fewshot_n": len(kept_real_train_names),
                "num_synth_keep": len(kept_synth_names),
                "real_train_names": kept_real_train_names,
                "synth_train_names": kept_synth_names,
                "test_names": test_names,
            },
            f,
            indent=2,
        )

    return out_scene


def build_conditions(common_real_names: List[str]) -> List[Dict]:
    conditions: List[Dict] = []

    # Few-shot real-only baseline (0% synthetic)
    fewshot_real_scene = GENERATED_FEWSHOT_ROOT / f"fewshot{FEWSHOT_N}_real_gs"
    conditions.append(
        {
            "name": f"fewshot{FEWSHOT_N}_real",
            "scene": fewshot_real_scene,
            "derived": True,
            "source_scene": FEWSHOT_REAL_SOURCE,
            "num_synth_keep": 0,
            "kept_real_names": common_real_names,
        }
    )

    # Few-shot + synthetic
    for model_name in MODELS:
        for ratio in SYN_RATIOS:
            num_synth_keep = int(round(FEWSHOT_N * ratio / 100.0))
            source_scene = DATASETS_ROOT / f"{model_name}_{ratio}syn_injected"
            out_scene = GENERATED_FEWSHOT_ROOT / f"fewshot{FEWSHOT_N}_{model_name}_{ratio}syn_gs"

            conditions.append(
                {
                    "name": f"fewshot{FEWSHOT_N}_{model_name}_{ratio}syn",
                    "scene": out_scene,
                    "derived": True,
                    "source_scene": source_scene,
                    "num_synth_keep": num_synth_keep,
                    "kept_real_names": common_real_names,
                }
            )

    return conditions


# ============================================================
# MAIN EXPERIMENT LOOP
# ============================================================
def main():
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    GENERATED_FEWSHOT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    if not FEWSHOT_REAL_SOURCE.exists():
        raise FileNotFoundError(f"Few-shot real source not found: {FEWSHOT_REAL_SOURCE}")

    # Pick the common FEWSHOT_N real training images once
    common_real_train_pool, _, _ = split_real_and_synth_names(FEWSHOT_REAL_SOURCE)
    common_real_names = evenly_sample_names(common_real_train_pool, FEWSHOT_N)

    print(f"[INFO] Common few-shot real train images selected: {len(common_real_names)}")

    conditions = build_conditions(common_real_names)

    for cond in conditions:
        name = cond["name"]
        scene = Path(cond["scene"]).resolve()
        model_path = RUNS_ROOT / name

        print("\n" + "=" * 80)
        print(f"CONDITION: {name}")
        print(f"SCENE:     {scene}")
        print(f"MODEL OUT: {model_path}")
        print("=" * 80)

        # Build few-shot GS scene first
        if cond.get("derived", False):
            source_scene = Path(cond["source_scene"]).resolve()
            if not source_scene.exists():
                msg = f"[WARNING] Missing source scene for {name}: {source_scene}"
                if SKIP_MISSING_SCENES:
                    print(msg)
                    continue
                raise FileNotFoundError(source_scene)

            build_fewshot_gs_scene(
                source_scene=source_scene,
                out_scene=scene,
                kept_real_train_names=cond["kept_real_names"],
                num_synth_keep=cond["num_synth_keep"],
            )

        if not scene.exists():
            msg = f"[WARNING] Scene does not exist: {scene}"
            if SKIP_MISSING_SCENES:
                print(msg)
                continue
            raise FileNotFoundError(scene)

        try:
            ensure_gs_ready_scene(scene)
        except FileNotFoundError as e:
            if SKIP_MISSING_SCENES:
                print(f"[WARNING] Skipping {name}: {e}")
                continue
            raise

        num_images = count_images(scene / "images")
        print(f"[INFO] Number of images in scene: {num_images}")

        done_iter = latest_saved_iteration(model_path)
        if done_iter is not None and done_iter >= ITERATIONS and not FORCE_RETRAIN:
            print(f"[INFO] Training already exists for {model_path} at iteration {done_iter}, skipping training.")
        else:
            train_cmd = [
            PYTHON_EXE,
            "train.py",
            "-s", str(scene),
            "-m", str(model_path),
            "--images", "images",
            "--eval",
            "-r", str(TRAIN_RESOLUTION),
            "--data_device", DATA_DEVICE,
            "--iterations", str(ITERATIONS),
            "--save_iterations", str(SAVE_ITERS),
            "--test_iterations", str(TEST_ITERS),
        ]
           

            stop_event = threading.Event()
            gpu_stats: Dict[str, float] = {"max_gpu_mem_mb": 0.0}
            monitor = threading.Thread(target=poll_gpu_memory, args=(stop_event, gpu_stats), daemon=True)

            t0 = time.time()
            monitor.start()
            try:
                run_cmd(train_cmd, GS_REPO)
            finally:
                stop_event.set()
                monitor.join()
            train_time_sec = time.time() - t0
        if done_iter is not None and done_iter >= ITERATIONS and not FORCE_RETRAIN:
            train_time_sec = 0.0
            gpu_stats = {"max_gpu_mem_mb": 0.0}

        render_iter = latest_saved_iteration(model_path)
        if render_iter is None:
            print(f"[WARNING] No trained iteration found for {model_path}, skipping render/metrics.")
            continue

        render_dir = model_path / "test" / f"ours_{render_iter}" / "renders"
        if render_dir.exists() and any(render_dir.glob("*.png")) and not FORCE_RENDER:
            print(f"[INFO] Renders already exist for {model_path}, skipping render.")
        else:
            render_cmd = [
                PYTHON_EXE,
                "render.py",
                "-m", str(model_path),
            ]
            run_cmd(render_cmd, GS_REPO)

        metrics_cmd = [
            PYTHON_EXE,
            "metrics.py",
            "-m", str(model_path),
        ]
        run_cmd(metrics_cmd, GS_REPO)

        results_json = model_path / "results.json"
        metrics = parse_metrics_json(results_json)

        row = {
            "name": name,
            "scene": str(scene),
            "num_images": num_images,
            "train_time_sec": round(train_time_sec, 2),
            "train_time_min": round(train_time_sec / 60.0, 2),
            "max_gpu_mem_mb": round(gpu_stats.get("max_gpu_mem_mb", 0.0), 2),
            "SSIM": metrics["SSIM"],
            "PSNR": metrics["PSNR"],
            "LPIPS": metrics["LPIPS"],
            "model_path": str(model_path),
        }
        summary_rows.append(row)

        print("\n[RESULT]")
        print(json.dumps(row, indent=2))

        save_summary(
            summary_rows,
            RUNS_ROOT / "summary_results.json",
            RUNS_ROOT / "summary_results.csv",
        )

    print("\nAll experiments finished.")
    print(f"Summary JSON: {RUNS_ROOT / 'summary_results.json'}")
    print(f"Summary CSV:  {RUNS_ROOT / 'summary_results.csv'}")


if __name__ == "__main__":
    main()