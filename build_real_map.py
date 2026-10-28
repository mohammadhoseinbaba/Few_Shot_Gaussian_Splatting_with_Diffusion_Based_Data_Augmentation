from pathlib import Path
import json
import shutil

import pycolmap
from PIL import Image
from hloc import (
    extract_features,
    match_features,
    pairs_from_retrieval,
    localize_sfm,
)

# ------------------------------------------------------------------
# INJECTION DATASET:
# - reference map comes from the full real dataset
# - final mixed dataset uses the incomplete real dataset + synthetic images
# - synthetic images NEVER go to the test set
# ------------------------------------------------------------------
project_root = Path("/mnt/d/project_root")
outputs_root = project_root / "GaussianSplatting" / "outputs"

MODELS = ["modelA", "modelB", "modelC", "modelD"]
SYN_LABELS = [25, 50, 100, 200]

TEST_STRIDE = 8
REMOVE_REAL_FRAME_START = 50
REMOVE_REAL_FRAME_END = 200

valid_exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

for model_name in MODELS:
    for SYN_LABEL in SYN_LABELS:
        # ----------------------------------------------------------
        # Incomplete real dataset: these real images are copied into
        # the final mixed dataset
        # ----------------------------------------------------------
        real_dataset = outputs_root / "my_scene"
        real_images = real_dataset / "input"

        # ----------------------------------------------------------
        # Full real dataset: used only as reference for localization
        # ----------------------------------------------------------
        reference_dataset = outputs_root / "my_scene_full"
        reference_sfm = reference_dataset / "sfm_superpoint+superglue"
        ref_features_path = reference_dataset / "feats-superpoint-n4096-r1024.h5"
        ref_retrieval_path = reference_dataset / "global-feats-netvlad.h5"

        # ----------------------------------------------------------
        # Synthetic source images
        # ----------------------------------------------------------
        synth_source_images = outputs_root / f"{model_name}_{SYN_LABEL}syn" / "input"

        # ----------------------------------------------------------
        # Output injected dataset
        # ----------------------------------------------------------
        injected_dataset = outputs_root / f"{model_name}_{SYN_LABEL}syn_injected"
        loc_dir = injected_dataset / "localization"
        query_images = loc_dir / "query_images"
        injected_images = injected_dataset / "images"
        injected_sparse = injected_dataset / "sparse" / "0"

        loc_pairs = loc_dir / "pairs-loc-netvlad.txt"
        query_list_path = loc_dir / "synthetic_queries.txt"
        localize_results = loc_dir / "localized_poses.txt"
        injected_pose_txt = injected_dataset / "injected_poses.txt"
        injection_metadata_path = injected_dataset / "injection_metadata.json"

        # ----------------------------------------------------------
        # Sanity checks
        # ----------------------------------------------------------
        for required in [
            reference_sfm / "images.bin",
            ref_features_path,
            ref_retrieval_path,
            synth_source_images,
            real_images,
        ]:
            if not required.exists():
                raise FileNotFoundError(required)

        for folder in [loc_dir, query_images, injected_images, injected_sparse]:
            folder.mkdir(parents=True, exist_ok=True)

        # Clean only current injection outputs
        for folder in [query_images, injected_images]:
            for old_image in folder.iterdir():
                if old_image.is_file() and old_image.suffix in valid_exts:
                    old_image.unlink()

        for old_model_file in list(injected_sparse.glob("*.txt")) + list(injected_sparse.glob("*.bin")):
            old_model_file.unlink()

        retrieval_conf = extract_features.confs["netvlad"]
        feature_conf = extract_features.confs["superpoint_aachen"]
        matcher_conf = match_features.confs["superglue"]

        # ----------------------------------------------------------
        # 1. Real-only test split from the full reference model
        # ----------------------------------------------------------
        def frame_number(name: str):
            stem = Path(name).stem
            if not stem.startswith("frame_"):
                return None
            try:
                return int(stem.split("_")[-1])
            except ValueError:
                return None

        def is_removed_original_frame(name: str):
            idx = frame_number(name)
            return idx is not None and REMOVE_REAL_FRAME_START <= idx <= REMOVE_REAL_FRAME_END

        ref_model = pycolmap.Reconstruction(reference_sfm)
        registered_real_images = sorted(ref_model.images.values(), key=lambda im: im.name)
        registered_real_names = [image.name for image in registered_real_images]

        removed_real_names = {name for name in registered_real_names if is_removed_original_frame(name)}
        real_test_candidates = [name for name in registered_real_names if name not in removed_real_names]
        real_test_names = real_test_candidates[::TEST_STRIDE]

        # ----------------------------------------------------------
        # 2. Prepare synthetic queries for HLoc
        #    Add synth_ prefix only to avoid filename collisions
        # ----------------------------------------------------------
        synth_mapping = {}
        for src in sorted(p for p in synth_source_images.iterdir() if p.suffix in valid_exts):
            query_name = f"synth_{src.name}"
            shutil.copy2(src, query_images / query_name)
            synth_mapping[src.name] = query_name

        if not synth_mapping:
            raise ValueError(f"No synthetic frame found in {synth_source_images}")

        # ----------------------------------------------------------
        # 3. Localize synthetic images with respect to the full map
        # ----------------------------------------------------------
        synth_retrieval_path = extract_features.main(retrieval_conf, query_images, loc_dir, overwrite=True)
        num_matched = min(20, ref_model.num_reg_images())

        pairs_from_retrieval.main(
            descriptors=synth_retrieval_path,
            output=loc_pairs,
            num_matched=num_matched,
            db_model=reference_sfm,
            db_descriptors=ref_retrieval_path,
        )

        synth_features_path = extract_features.main(feature_conf, query_images, loc_dir, overwrite=True)
        loc_matches_path = match_features.main(
            matcher_conf,
            loc_pairs,
            features=synth_features_path,
            features_ref=ref_features_path,
            matches=loc_dir / "matches-loc-superglue.h5",
            overwrite=True,
        )

        with open(query_list_path, "w") as f:
            for img_path in sorted(query_images.iterdir()):
                if img_path.suffix not in valid_exts:
                    continue
                with Image.open(img_path) as img:
                    width, height = img.size
                focal = max(width, height) * 1.2
                cx = width / 2.0
                cy = height / 2.0
                f.write(f"{img_path.name} SIMPLE_PINHOLE {width} {height} {focal} {cx} {cy}\n")

        localize_sfm.main(
            reference_sfm=reference_sfm,
            queries=query_list_path,
            retrieval=loc_pairs,
            features=synth_features_path,
            matches=loc_matches_path,
            results=localize_results,
            covisibility_clustering=True,
        )

        # ----------------------------------------------------------
        # 4. Build final injected dataset
        # ----------------------------------------------------------
        def camera_to_pinhole_params(camera):
            params = list(map(float, camera.params))
            if camera.model_name == "PINHOLE":
                return params[:4]
            if camera.model_name in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
                f, cx, cy = params[:3]
                return [f, f, cx, cy]
            if len(params) >= 4:
                return params[:4]
            raise ValueError(f"Camera model cannot be converted to PINHOLE: {camera.model_name}")

        # Copy real images from the incomplete real set.
        # IMPORTANT: filenames in my_scene/input must match original names
        # from my_scene_full/input.
        real_camera_id_map = {}
        kept_real_images = []
        available_real_names = {p.name for p in real_images.iterdir() if p.is_file() and p.suffix in valid_exts}

        for image in registered_real_images:
            if image.name in removed_real_names:
                continue
            if image.name not in available_real_names:
                continue
            src = real_images / image.name
            shutil.copy2(src, injected_images / image.name)
            kept_real_images.append(image)

        # Read localized synthetic poses
        localized = {}
        with open(localize_results) as f:
            for line in f:
                tokens = line.split()
                if len(tokens) != 8:
                    continue
                query_name = tokens[0]
                localized[query_name] = {
                    "pose": [float(x) for x in tokens[1:]],
                }

        # Inject ALL localized synthetic images into training.
        # Synthetic images never go to test.txt.
        injected_synth = {}
        for query_name, loc in sorted(localized.items()):
            shutil.copy2(query_images / query_name, injected_images / query_name)
            injected_synth[query_name] = loc

        # Write cameras.txt
        camera_lines = []
        next_camera_id = 1
        for old_id, camera in sorted(ref_model.cameras.items()):
            real_camera_id_map[old_id] = next_camera_id
            fx, fy, cx, cy = camera_to_pinhole_params(camera)
            camera_lines.append(f"{next_camera_id} PINHOLE {camera.width} {camera.height} {fx} {fy} {cx} {cy}")
            next_camera_id += 1

        query_camera_id_by_spec = {}
        query_camera_id_by_name = {}
        with open(query_list_path) as f:
            for line in f:
                name, model_name, width, height, focal, cx, cy = line.split()
                spec = (int(width), int(height), float(focal), float(cx), float(cy))
                if spec not in query_camera_id_by_spec:
                    query_camera_id_by_spec[spec] = next_camera_id
                    camera_lines.append(
                        f"{next_camera_id} PINHOLE {spec[0]} {spec[1]} {spec[2]} {spec[2]} {spec[3]} {spec[4]}"
                    )
                    next_camera_id += 1
                query_camera_id_by_name[name] = query_camera_id_by_spec[spec]

        with open(injected_sparse / "cameras.txt", "w") as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write("\n".join(camera_lines) + "\n")

        # Write images.txt
        image_entries = []
        for image in kept_real_images:
            pose = image.cam_from_world()
            q = pose.rotation.quat[[3, 0, 1, 2]]
            t = pose.translation
            image_entries.append((image.name, list(q), list(t), real_camera_id_map[image.camera_id]))

        for query_name, loc in injected_synth.items():
            vals = loc["pose"]
            image_entries.append((query_name, vals[:4], vals[4:], query_camera_id_by_name[query_name]))

        image_lines = []
        for image_id, (name, q, t, camera_id) in enumerate(sorted(image_entries), start=1):
            image_lines.append(
                f"{image_id} {' '.join(map(str, q))} {' '.join(map(str, t))} {camera_id} {name}"
            )

        with open(injected_sparse / "images.txt", "w") as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
            f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
            for line in image_lines:
                f.write(line + "\n\n")

        # Write points3D.txt using the reference full map
        with open(injected_sparse / "points3D.txt", "w") as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
            for point3D_id, point in sorted(ref_model.points3D.items()):
                xyz = " ".join(map(str, point.xyz))
                rgb = " ".join(map(str, point.color.astype(int)))
                f.write(f"{point3D_id} {xyz} {rgb} {point.error}\n")

        # Test set: REAL IMAGES ONLY
        with open(injected_sparse / "test.txt", "w") as f:
            f.write("\n".join(real_test_names) + "\n")

        with open(injected_pose_txt, "w") as f:
            for line in image_lines:
                tokens = line.split()
                f.write(tokens[9] + " " + " ".join(tokens[1:8]) + "\n")

        metadata = {
            "mode": "injection",
            "model_name": model_name,
            "syn_label": SYN_LABEL,
            "test_stride": TEST_STRIDE,
            "num_real_registered_in_full_map": len(registered_real_images),
            "removed_original_frame_start": REMOVE_REAL_FRAME_START,
            "removed_original_frame_end": REMOVE_REAL_FRAME_END,
            "num_real_removed_from_reference_range": len(removed_real_names),
            "num_real_kept_in_injected_dataset": len(kept_real_images),
            "num_real_test": len(real_test_names),
            "num_synth_available": len(synth_mapping),
            "num_synth_injected": len(injected_synth),
            "test_names": real_test_names,
        }

        with open(injection_metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print("\nInjection dataset completed")
        print(f"Model: {model_name}")
        print(f"Synthetic ratio label: {SYN_LABEL}")
        print(f"Reference full-map registered real images: {len(registered_real_images)}")
        print(f"Reference removed frame range count: {len(removed_real_names)} ({REMOVE_REAL_FRAME_START}-{REMOVE_REAL_FRAME_END})")
        print(f"Real images copied from incomplete dataset: {len(kept_real_images)}")
        print(f"Real test frames: {len(real_test_names)}")
        print(f"Available synthetic images: {len(synth_mapping)}")
        print(f"Synthetic images injected into training: {len(injected_synth)}")
        print(f"Injection dataset: {injected_dataset}")
        print(f"COLMAP text model: {injected_sparse}")
        print(f"Metadata: {injection_metadata_path}")