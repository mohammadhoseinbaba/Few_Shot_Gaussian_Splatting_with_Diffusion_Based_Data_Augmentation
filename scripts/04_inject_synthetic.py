"""Localize synthetic images in the real SfM map and build a GS-ready COLMAP scene.

Adapted from the inject_synthetic.py of github.com/mxhrnxz/Few_Shot_Gaussian_Splatting_with_Diffusion_Based_Data_Augmentation.
Differences: one run per strategy (the synthetic pool is subsampled per ratio later, in
05_run_gs_experiments.py), config-free CLI paths, the gap comes from gap.json, queries use the
map's own camera intrinsics, and per-image PnP inliers are recorded for quality filtering.

Inputs under --root (default data/outputs):
    my_scene/input/                     incomplete real frames (gap removed)
    my_scene_full/{sfm_*,feats-*.h5,global-feats-netvlad.h5}   from 02_build_real_map.py
    synth/<strategy>/input/*.jpg        synthetic images; name must contain the source real frame,
                                        e.g. outpaint_frame_0042_03.jpg (approx. real view link)
Output: <root>/<strategy>_injected/{images, sparse/0/{cameras,images,points3D,test}.txt, injection_metadata.json}
Synthetic images are only ever added to the training side; test.txt is real-only.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pycolmap
from hloc import extract_features, localize_sfm, match_features, pairs_from_retrieval
from PIL import Image

from src.common import read_inliers, camera_to_pinhole_params, compute_test_names, frame_number, in_gap, list_images, load_gap


def inject(root: Path, strategy: str, test_stride: int, min_inliers: int, num_matched: int) -> None:
    ref = root / "my_scene_full"
    ref_sfm = ref / "sfm_superpoint+superglue"
    ref_feats = ref / "feats-superpoint-n4096-r1024.h5"
    ref_retrieval = ref / "global-feats-netvlad.h5"
    real_images = root / "my_scene" / "input"
    synth_src = root / "synth" / strategy / "input"
    for req in (ref_sfm / "images.bin", ref_feats, ref_retrieval, real_images, synth_src, root / "gap.json"):
        if not req.exists():
            raise FileNotFoundError(req)

    out = root / f"{strategy}_injected"
    shutil.rmtree(out, ignore_errors=True)
    loc_dir, query_dir = out / "localization", out / "localization" / "query_images"
    img_dir, sparse = out / "images", out / "sparse" / "0"
    for d in (query_dir, img_dir, sparse):
        d.mkdir(parents=True)

    gap = load_gap(root)
    model = pycolmap.Reconstruction(ref_sfm)
    registered = sorted(model.images.values(), key=lambda im: im.name)
    test_names = compute_test_names([im.name for im in registered], gap, test_stride)
    test_set = set(test_names)

    # reference camera (single shared camera from 02_build_real_map.py)
    ref_cam_id, ref_cam = sorted(model.cameras.items())[0]
    ref_size = (ref_cam.width, ref_cam.height)

    # 1. synthetic queries; center-crop to the map's aspect ratio and resize to its resolution so
    #    the map's own intrinsics apply (generators should already output this aspect ratio)
    n_cropped = 0
    with open(loc_dir / "queries.txt", "w") as qf:
        for name in list_images(synth_src):
            with Image.open(synth_src / name) as im:
                im = im.convert("RGB")
                target = ref_size[0] / ref_size[1]
                if abs(im.width / im.height - target) >= 0.02:
                    n_cropped += 1
                    if im.width / im.height > target:
                        w = round(im.height * target)
                        im = im.crop(((im.width - w) // 2, 0, (im.width - w) // 2 + w, im.height))
                    else:
                        h = round(im.width / target)
                        im = im.crop((0, (im.height - h) // 2, im.width, (im.height - h) // 2 + h))
                im = im.resize(ref_size, Image.LANCZOS)
                qname = f"synth_{Path(name).stem}.jpg"
                im.save(query_dir / qname, quality=95)
            params = " ".join(map(str, ref_cam.params))
            qf.write(f"{qname} {ref_cam.model_name} {ref_size[0]} {ref_size[1]} {params}\n")
    n_query = len(list_images(query_dir))
    if n_query == 0:
        raise ValueError(f"no synthetic images in {synth_src}")
    if n_cropped:
        print(f"[WARN] {n_cropped} synthetic images were center-cropped to the map aspect ratio")

    # 2. localize against the full real map
    retrieval_conf = extract_features.confs["netvlad"]
    feature_conf = extract_features.confs["superpoint_aachen"]
    matcher_conf = match_features.confs["superglue"]
    pairs = loc_dir / "pairs-loc-netvlad.txt"
    results = loc_dir / "localized_poses.txt"

    q_retrieval = extract_features.main(retrieval_conf, query_dir, loc_dir)
    pairs_from_retrieval.main(q_retrieval, pairs, num_matched=min(num_matched, model.num_reg_images()),
                              db_model=ref_sfm, db_descriptors=ref_retrieval)
    q_feats = extract_features.main(feature_conf, query_dir, loc_dir)
    q_matches = match_features.main(matcher_conf, pairs, features=q_feats, features_ref=ref_feats,
                                    matches=loc_dir / "matches-loc-superglue.h5")
    localize_sfm.main(reference_sfm=ref_sfm, queries=loc_dir / "queries.txt", retrieval=pairs,
                      features=q_feats, matches=q_matches, results=results, covisibility_clustering=True)

    # 3. select what to inject
    inliers = read_inliers(results)
    localized = {}
    for line in results.read_text().splitlines():
        t = line.split()
        if len(t) == 8:
            localized[t[0]] = [float(x) for x in t[1:]]

    injected, excluded_leak, excluded_quality = {}, {}, {}
    for qname, pose in sorted(localized.items()):
        src_idx = frame_number(qname)
        src_name = f"frame_{src_idx:04d}.jpg" if src_idx is not None else None
        n_in = inliers.get(qname)
        if src_name in test_set:
            excluded_leak[qname] = src_name
        elif min_inliers and (n_in is None or n_in < min_inliers):
            excluded_quality[qname] = n_in
        else:
            injected[qname] = pose
            shutil.copy2(query_dir / qname, img_dir / qname)

    # 4. real training side: incomplete real set (gap excluded), registered frames only
    available = set(list_images(real_images))
    kept_real = [im for im in registered if im.name in available and not in_gap(im.name, gap)]
    for im in kept_real:
        shutil.copy2(real_images / im.name, img_dir / im.name)

    # 5. COLMAP text model
    cam_lines = []
    for cid, cam in sorted(model.cameras.items()):
        fx, fy, cx, cy = camera_to_pinhole_params(cam)
        cam_lines.append(f"{cid} PINHOLE {cam.width} {cam.height} {fx} {fy} {cx} {cy}")
    entries = []
    for im in kept_real:
        pose = im.cam_from_world()
        entries.append((im.name, list(pose.rotation.quat[[3, 0, 1, 2]]), list(pose.translation), im.camera_id))
    for qname, pose in injected.items():
        entries.append((qname, pose[:4], pose[4:], ref_cam_id))
    entries.sort()

    (sparse / "cameras.txt").write_text(
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n" + "\n".join(cam_lines) + "\n")
    with open(sparse / "images.txt", "w") as f:
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i, (name, q, t, cid) in enumerate(entries, start=1):
            f.write(f"{i} {' '.join(map(str, q))} {' '.join(map(str, t))} {cid} {name}\n\n")
    with open(sparse / "points3D.txt", "w") as f:
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for pid, p in sorted(model.points3D.items()):
            f.write(f"{pid} {' '.join(map(str, p.xyz))} {' '.join(map(str, p.color.astype(int)))} {p.error}\n")
    (sparse / "test.txt").write_text("\n".join(test_names) + "\n")

    meta = {
        "strategy": strategy, "test_stride": test_stride, "min_inliers": min_inliers,
        "gap": [gap["start"], gap["end"]],
        "num_registered_real": len(registered), "num_real_train_side": len(kept_real),
        "num_real_test": len(test_names), "test_names": test_names,
        "num_synth_queries": n_query, "num_synth_localized": len(localized),
        "num_synth_injected": len(injected),
        "localization_success_rate": round(len(localized) / n_query, 4),
        "excluded_test_leakage": excluded_leak, "excluded_low_inliers": excluded_quality,
        "inliers": inliers,
        "injected_synth_to_source": {q: f"frame_{frame_number(q):04d}.jpg" if frame_number(q) else None for q in injected},
    }
    (out / "injection_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[{strategy}] localized {len(localized)}/{n_query}; injected {len(injected)}; "
          f"leak-excluded {len(excluded_leak)}; quality-excluded {len(excluded_quality)}; "
          f"real train side {len(kept_real)}; test {len(test_names)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/outputs"))
    ap.add_argument("--strategy", nargs="+", required=True, help="strategy folder name(s) under <root>/synth/")
    ap.add_argument("--test-stride", type=int, default=8)
    ap.add_argument("--min-inliers", type=int, default=0, help="drop synthetic frames with fewer PnP inliers (0 = keep all)")
    ap.add_argument("--num-matched", type=int, default=20)
    args = ap.parse_args()
    for s in args.strategy:
        inject(args.root, s, args.test_stride, args.min_inliers, args.num_matched)


if __name__ == "__main__":
    main()
