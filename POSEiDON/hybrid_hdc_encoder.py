

from typing import Dict, Any
import json
import numpy as np

from angle_hdc_dict import hdc_angle_dict
from angle_hdc_encoder import encode_pose_from_row


def load_grid_dict(path: str) -> Dict[str, Any]:
    
    with open(path, "r") as f:
        data = json.load(f)

    D = int(data["D"])


    joint_hv_np: Dict[str, np.ndarray] = {}
    for name, hv_list in data["joint_hv"].items():
        hv = np.array(hv_list, dtype=np.int8)
        if hv.shape[0] != D:
            raise ValueError(f"Joint HV for {name} has dim {hv.shape[0]}, expected {D}")
        joint_hv_np[name] = hv


    pos_x_hv_np: Dict[int, np.ndarray] = {}
    for ix_str, hv_list in data["pos_x_hv"].items():
        ix = int(ix_str)
        hv = np.array(hv_list, dtype=np.int8)
        if hv.shape[0] != D:
            raise ValueError(f"pos_x HV for ix={ix} has dim {hv.shape[0]}, expected {D}")
        pos_x_hv_np[ix] = hv

    pos_y_hv_np: Dict[int, np.ndarray] = {}
    for iy_str, hv_list in data["pos_y_hv"].items():
        iy = int(iy_str)
        hv = np.array(hv_list, dtype=np.int8)
        if hv.shape[0] != D:
            raise ValueError(f"pos_y HV for iy={iy} has dim {hv.shape[0]}, expected {D}")
        pos_y_hv_np[iy] = hv

    data["joint_hv_np"] = joint_hv_np
    data["pos_x_hv_np"] = pos_x_hv_np
    data["pos_y_hv_np"] = pos_y_hv_np
    data["D"] = D

    return data


def bipolar_sign(vec: np.ndarray) -> np.ndarray:
    
    s = (vec >= 0).astype(np.int8)
    s[s == 0] = -1
    return s


ANGLE_JOINT_TO_GRID_NAME = {
    "left_shoulder":  "LEFT_SHOULDER",
    "right_shoulder": "RIGHT_SHOULDER",
    "left_elbow":     "LEFT_ELBOW",
    "right_elbow":    "RIGHT_ELBOW",
    "left_hip":       "LEFT_HIP",
    "right_hip":      "RIGHT_HIP",
    "left_knee":      "LEFT_KNEE",
    "right_knee":     "RIGHT_KNEE",
}


JOINT_POS_COLUMNS = {
    "left_shoulder":  ("left_shoulder_x",  "left_shoulder_y"),
    "right_shoulder": ("right_shoulder_x", "right_shoulder_y"),
    "left_elbow":     ("left_elbow_x",     "left_elbow_y"),
    "right_elbow":    ("right_elbow_x",    "right_elbow_y"),
    "left_hip":       ("left_hip_x",       "left_hip_y"),
    "right_hip":      ("right_hip_x",      "right_hip_y"),
    "left_knee":      ("left_knee_x",      "left_knee_y"),
    "right_knee":     ("right_knee_x",     "right_knee_y"),
}


def encode_grid_from_row(
    row: "pd.Series",
    grid_dict: Dict[str, Any],
) -> np.ndarray:
    
    D = grid_dict["D"]
    grid_x = int(grid_dict["grid_x"])
    grid_y = int(grid_dict["grid_y"])


    if ("img_width" not in row) or ("img_height" not in row):

        return np.zeros(D, dtype=np.int8)

    img_w = float(row["img_width"])
    img_h = float(row["img_height"])
    if img_w <= 1e-6 or img_h <= 1e-6:
        return np.zeros(D, dtype=np.int8)

    joint_hv_np = grid_dict["joint_hv_np"]
    pos_x_hv_np = grid_dict["pos_x_hv_np"]
    pos_y_hv_np = grid_dict["pos_y_hv_np"]

    acc = np.zeros(D, dtype=np.int32)
    any_joint_used = False

    for angle_joint_name, (col_x, col_y) in JOINT_POS_COLUMNS.items():
        if (col_x not in row) or (col_y not in row):

            continue

        x = float(row[col_x])
        y = float(row[col_y])

        if not np.isfinite(x) or not np.isfinite(y):
            continue


        ix = int(x / img_w * grid_x)
        iy = int(y / img_h * grid_y)
        ix = max(0, min(grid_x - 1, ix))
        iy = max(0, min(grid_y - 1, iy))


        grid_joint_name = ANGLE_JOINT_TO_GRID_NAME.get(angle_joint_name, None)
        if grid_joint_name is None or grid_joint_name not in joint_hv_np:
            continue

        J = joint_hv_np[grid_joint_name]
        PX = pos_x_hv_np.get(ix, None)
        PY = pos_y_hv_np.get(iy, None)

        if PX is None or PY is None:
            continue


        H_joint = (
            J.astype(np.int16) *
            PX.astype(np.int16) *
            PY.astype(np.int16)
        ).astype(np.int32)

        acc += H_joint
        any_joint_used = True

    if not any_joint_used:
        return np.zeros(D, dtype=np.int8)

    return bipolar_sign(acc)


def encode_hybrid_from_row(
    row: "pd.Series",
    grid_dict: Dict[str, Any],
    angle_dict: Dict[str, Any] = hdc_angle_dict,
) -> np.ndarray:


    H_angle = encode_pose_from_row(row, hdc_dict=angle_dict).astype(np.int8)


    H_grid = encode_grid_from_row(row, grid_dict=grid_dict).astype(np.int8)


    if not np.any(H_grid != 0):
        return H_angle


    H_bind = (H_angle.astype(np.int16) * H_grid.astype(np.int16)).astype(np.int32)


    H_hybrid = bipolar_sign(H_bind).astype(np.int8)

    return H_hybrid


if __name__ == "__main__":

    pass
