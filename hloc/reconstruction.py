import argparse
import multiprocessing
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import pycolmap

from . import logger
from .triangulation import (
    OutputCapture,
    estimation_and_geometric_verification,
    import_features,
    import_matches,
    parse_option_args,
)
from .utils.database import COLMAPDatabase


def create_empty_db(database_path: Path):
    if database_path.exists():
        logger.warning("The database already exists, deleting it.")
        database_path.unlink()
    logger.info("Creating an empty database...")
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    db.commit()
    db.close()


def import_images(
    image_dir: Path,
    database_path: Path,
    camera_mode: pycolmap.CameraMode,
    image_list: Optional[List[str]] = None,
    options: Optional[Dict[str, Any]] = None,
):
    logger.info("Importing images into the database...")
    if options is None:
        options = {}
    images = list(image_dir.iterdir())
    if len(images) == 0:
        raise IOError(f"No images found in {image_dir}.")
    with pycolmap.ostream():
        pycolmap.import_images(
            database_path,
            image_dir,
            camera_mode,
            image_list=image_list or [],
            options=options,
        )

def import_images_multicam(
    database_path: Path,
    camera_mode: pycolmap.CameraMode,
    image_list: Optional[List[str]] = None,
    cameras: Optional[List[Dict[str, Any]]] = None,
):
    logger.info("Importing images into the database...")
    db = COLMAPDatabase.connect(database_path)
    # For each camera
    for i, cam in enumerate(cameras):
        # Add the camera to the database
        camera_id = db.add_camera(1, cam["width"], cam["height"], 
                                  [cam["params"]["fx"], 
                                   cam["params"]["fy"], 
                                   cam["params"]["cx"], 
                                   cam["params"]["cy"]],
                                   prior_focal_length=True)
        # For each image taken with this camera
        for image_file_name in image_list:
            # Add the image to the database, associating it with this camera
            # db.add_image(image_name, camera_id, prior_q, prior_t)
            if cam["name"] in image_file_name:
                db.add_image(image_file_name, camera_id)

    db.commit()
    db.close()

def import_images_nuscenes_multicam(
    database_path: Path,
    camera_mode: pycolmap.CameraMode,
    image_list: Optional[List[str]] = None,
    cam_intrinsic_list: Optional[List[str]] = None,
    cam_extrinsic_list: Optional[List[str]] = None
):
    logger.info("Importing images into the database...")
    db = COLMAPDatabase.connect(database_path)
    # For each camera
    cams = ['CAM_FRONT__', 'CAM_FRONT_LEFT__', 'CAM_FRONT_RIGHT__']
    for i, cam in enumerate(cams):
        # Add the camera to the database
        camera_id = db.add_camera(1, 1600, 900, 
                                  [cam_intrinsic_list[i][0][0], 
                                   cam_intrinsic_list[i][1][1], 
                                   cam_intrinsic_list[i][0][2], 
                                   cam_intrinsic_list[i][1][2]])
        # For each image taken with this camera
        for j, image_file_name in enumerate(image_list):
            # Add the image to the database, associating it with this camera
            # db.add_image(image_name, camera_id, prior_q, prior_t)
            if cam in image_file_name:
                db.add_image(image_file_name, camera_id)

    db.commit()
    db.close()

def import_images_kitti_stereocam(
    database_path: Path,
    camera_mode: pycolmap.CameraMode,
    image_list: Optional[List[str]] = None,
    cam_intrinsic_list: Optional[List[str]] = None,
    cam_extrinsic_list: Optional[List[str]] = None
):
    logger.info("Importing images into the database...")
    db = COLMAPDatabase.connect(database_path)
    # For each camera
    cams = ['left', 'right']
    for i, cam in enumerate(cams):
        # Add the camera to the database
        camera_id = db.add_camera(1, 1241, 376, 
                                  [cam_intrinsic_list[i][0][0], 
                                   cam_intrinsic_list[i][1][1],
                                   cam_intrinsic_list[i][0][2], 
                                   cam_intrinsic_list[i][1][2]])
        # For each image taken with this camera
        for j, image_file_name in enumerate(image_list):
            # Add the image to the database, associating it with this camera
            # db.add_image(image_name, camera_id, prior_q, prior_t)
            if cam in image_file_name:
                db.add_image(image_file_name, camera_id)

    db.commit()
    db.close()


def get_image_ids(database_path: Path) -> Dict[str, int]:
    db = COLMAPDatabase.connect(database_path)
    images = {}
    for name, image_id in db.execute("SELECT name, image_id FROM images;"):
        images[name] = image_id
    db.close()
    return images


def run_reconstruction(
    sfm_dir: Path,
    database_path: Path,
    image_dir: Path,
    verbose: bool = False,
    options: Optional[Dict[str, Any]] = None,
) -> pycolmap.Reconstruction:
    models_path = sfm_dir / "models"
    models_path.mkdir(exist_ok=True, parents=True)
    logger.info("Running 3D reconstruction...")
    if options is None:
        options = {}
    options = {"num_threads": min(multiprocessing.cpu_count(), 16), **options}
    with OutputCapture(verbose):
        with pycolmap.ostream():
            reconstructions = pycolmap.incremental_mapping(
                database_path, image_dir, models_path, options=options
            )

    if len(reconstructions) == 0:
        logger.error("Could not reconstruct any model!")
        return None
    logger.info(f"Reconstructed {len(reconstructions)} model(s).")

    largest_index = None
    largest_num_images = 0
    for index, rec in reconstructions.items():
        num_images = rec.num_reg_images()
        if num_images > largest_num_images:
            largest_index = index
            largest_num_images = num_images
    assert largest_index is not None
    logger.info(
        f"Largest model is #{largest_index} " f"with {largest_num_images} images."
    )

    for filename in ["images.bin", "cameras.bin", "points3D.bin"]:
        if (sfm_dir / filename).exists():
            (sfm_dir / filename).unlink()
        shutil.move(str(models_path / str(largest_index) / filename), str(sfm_dir))
    return reconstructions[largest_index]


def main(
    sfm_dir: Path,
    image_dir: Path,
    pairs: Path,
    features: Path,
    matches: Path,
    camera_mode: pycolmap.CameraMode = pycolmap.CameraMode.AUTO,
    verbose: bool = False,
    skip_geometric_verification: bool = False,
    min_match_score: Optional[float] = None,
    image_list: Optional[List[str]] = None,
    cameras: Optional[List[Dict[str, Any]]] = None,
    image_options: Optional[Dict[str, Any]] = None,
    mapper_options: Optional[Dict[str, Any]] = None,
    apply_glomap: bool = False
) -> pycolmap.Reconstruction:
    assert features.exists(), features
    assert pairs.exists(), pairs
    assert matches.exists(), matches

    sfm_dir.mkdir(parents=True, exist_ok=True)
    database = sfm_dir / "database.db"

    create_empty_db(database)
    if len(cameras) > 1 :
        import_images_multicam(database, camera_mode, image_list, cameras)
    else :
        import_images(image_dir, database, camera_mode, image_list, image_options)
    image_ids = get_image_ids(database)
    import_features(image_ids, database, features)
    import_matches(
        image_ids,
        database,
        pairs,
        matches,
        min_match_score,
        skip_geometric_verification,
    )

    if not skip_geometric_verification:
        estimation_and_geometric_verification(database, pairs, verbose)

    if apply_glomap :
        logger.info(
                f"Colmap reconstruction is skipped. Will apply glomap for reconstruction..."
            )
        return None
    else :
        reconstruction = run_reconstruction(
            sfm_dir, database, image_dir, verbose, mapper_options
        )
        if reconstruction is not None:
            logger.info(
                f"Reconstruction statistics:\n{reconstruction.summary()}"
                + f"\n\tnum_input_images = {len(image_ids)}"
            )
        return reconstruction

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sfm_dir", type=Path, required=True)
    parser.add_argument("--image_dir", type=Path, required=True)

    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)

    parser.add_argument(
        "--camera_mode",
        type=str,
        default="AUTO",
        choices=list(pycolmap.CameraMode.__members__.keys()),
    )
    parser.add_argument("--skip_geometric_verification", action="store_true")
    parser.add_argument("--min_match_score", type=float)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument(
        "--image_options",
        nargs="+",
        default=[],
        help="List of key=value from {}".format(pycolmap.ImageReaderOptions().todict()),
    )
    parser.add_argument(
        "--mapper_options",
        nargs="+",
        default=[],
        help="List of key=value from {}".format(
            pycolmap.IncrementalMapperOptions().todict()
        ),
    )
    args = parser.parse_args().__dict__

    image_options = parse_option_args(
        args.pop("image_options"), pycolmap.ImageReaderOptions()
    )
    mapper_options = parse_option_args(
        args.pop("mapper_options"), pycolmap.IncrementalMapperOptions()
    )

    main(**args, image_options=image_options, mapper_options=mapper_options)
