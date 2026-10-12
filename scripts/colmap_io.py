from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

MAX_IMAGE_ID = 2**31 - 1


def blob_array(blob, dtype, shape=(-1,)):
    if blob is None:
        return np.zeros(shape, dtype=dtype)
    return np.frombuffer(blob, dtype=dtype).reshape(*shape)


def pair_id_to_image_ids(pair_id: int) -> tuple[int, int]:
    id2 = pair_id % MAX_IMAGE_ID
    id1 = (pair_id - id2) // MAX_IMAGE_ID
    return int(id1), int(id2)


def image_ids_to_pair_id(id1: int, id2: int) -> int:
    if id1 > id2:
        id1, id2 = id2, id1
    return id1 * MAX_IMAGE_ID + id2


def read_colmap_features(database_path: Path) -> dict:
    con = sqlite3.connect(str(database_path))
    con.row_factory = sqlite3.Row
    images = {}
    for row in con.execute("SELECT image_id, name FROM images"):
        images[int(row["image_id"])] = row["name"]

    keypoints = {}
    for row in con.execute("SELECT image_id, rows, cols, data FROM keypoints"):
        arr = blob_array(row["data"], np.float32, (row["rows"], row["cols"]))
        keypoints[int(row["image_id"])] = arr[:, :2].astype(np.float64)

    matches = {}
    try:
        for row in con.execute("SELECT pair_id, rows, cols, data FROM matches"):
            arr = blob_array(row["data"], np.uint32, (row["rows"], row["cols"]))
            matches[int(row["pair_id"])] = arr
    except sqlite3.OperationalError:
        matches = {}

    inliers = {}
    try:
        for row in con.execute("SELECT pair_id, rows, cols, data FROM two_view_geometries"):
            arr = blob_array(row["data"], np.uint32, (row["rows"], row["cols"]))
            inliers[int(row["pair_id"])] = arr
    except sqlite3.OperationalError:
        inliers = {}

    name_to_id = {name: image_id for image_id, name in images.items()}
    con.close()
    return {
        "images": images,
        "name_to_id": name_to_id,
        "keypoints": keypoints,
        "matches": matches,
        "inliers": inliers,
    }
