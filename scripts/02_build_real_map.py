"""Build the reference SfM map from the FULL real frame set with HLoc (needs a GPU).

Pipeline: NetVLAD retrieval -> SuperPoint features -> SuperGlue matching -> COLMAP triangulation.

Writes into <scene>/ (default data/outputs/my_scene_full):
    global-feats-netvlad.h5
    feats-superpoint-n4096-r1024.h5
    pairs-netvlad.txt, matches-superpoint+superglue.h5
    sfm_superpoint+superglue/        COLMAP model (cameras/images/points3D .bin)

These names are what 04_inject_synthetic.py expects as the localization reference.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import pycolmap
from hloc import extract_features, match_features, pairs_from_retrieval, reconstruction


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=Path, default=Path("data/outputs/my_scene_full"),
                    help="folder containing input/ with the full real frames")
    ap.add_argument("--num-matched", type=int, default=40, help="retrieval neighbours per image")
    ap.add_argument("--camera-model", default="SIMPLE_RADIAL",
                    help="COLMAP camera model (one shared camera is estimated)")
    ap.add_argument("--matcher", default="superglue", choices=["superglue", "superglue-fast", "lightglue"],
                    help="use lightglue if SuperGlue is unavailable (matcher name goes into the output filenames)")
    args = ap.parse_args()

    images = args.scene / "input"
    if not images.is_dir() or not any(images.iterdir()):
        raise SystemExit(f"no images in {images}; run 01_extract_frames.py first")
    n_images = sum(1 for p in images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"mapping {n_images} images from {images}")

    retrieval_conf = extract_features.confs["netvlad"]
    feature_conf = extract_features.confs["superpoint_aachen"]  # -> feats-superpoint-n4096-r1024
    matcher_key = {"superglue": "superglue", "superglue-fast": "superglue-fast", "lightglue": "superpoint+lightglue"}[args.matcher]
    matcher_conf = match_features.confs[matcher_key]

    sfm_dir = args.scene / f"sfm_superpoint+{args.matcher}"
    pairs = args.scene / "pairs-netvlad.txt"
    matches = args.scene / f"matches-superpoint+{args.matcher}.h5"

    retrieval_path = extract_features.main(retrieval_conf, images, args.scene)
    features = extract_features.main(feature_conf, images, args.scene)
    pairs_from_retrieval.main(retrieval_path, pairs, num_matched=min(args.num_matched, n_images - 1))
    matches = match_features.main(matcher_conf, pairs, feature_conf["output"], args.scene, matches=matches)

    model = reconstruction.main(
        sfm_dir, images, pairs, features, matches,
        camera_mode=pycolmap.CameraMode.SINGLE,
        image_options=dict(camera_model=args.camera_model),
    )
    if model is None:
        raise SystemExit("reconstruction failed (no model). Check frame overlap / blur.")

    registered = model.num_reg_images()
    print(f"registered {registered}/{n_images} images, {model.num_points3D()} 3D points")
    if registered < 0.95 * n_images:
        print("[WARN] <95% registered; the held-out test set and baseline will be weaker")
    # names of registered frames: shared by generation, injection and GS scripts (src/common.py: load_split)
    (args.scene / "registered.json").write_text(json.dumps(sorted(im.name for im in model.images.values())))
    print(f"map: {sfm_dir}\nfeatures: {features}\nretrieval: {retrieval_path}")


if __name__ == "__main__":
    main()
