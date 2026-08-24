"""
hybrid_hdc_encoder.py

Hybrid HDC encoder that combines:
  - Angle-based HDC (from angle_hdc_dict + angle_hdc_encoder)
  - Grid-based HDC (from build_hdc_dictionary.py JSON)

Hybrid HV:
    H_hybrid = sign( H_angle + H_grid )

If the CSV row does NOT contain the expected joint position
columns (e.g., left_shoulder_x, left_shoulder_y, img_width, img_height),
the grid part is skipped and we just return H_angle.
"""

from typing import Dict, Any
import json
import numpy as np

from angle_hdc_dict import hdc_angle_dict
from angle_hdc_encoder import encode_pose_from_row  # angle-only encoder


# ---------------------------------------------------------------------
# Load grid dictionary
# ---------------------------------------------------------------------

def load_grid_dict(path: str) -> Dict[str, Any]:
    """
    Load the grid HDC dictionary created by build_hdc_dictionary.py.

    Expected JSON structure:
      {
        "D": 4096,
        "seed": ...,
        "grid_x": 18,
        "grid_y": 32,
        "joint_names": [...],
        "joint_hv": { joint_name: [-1,+1,...] },
        "pos_x_hv": { "0": [-1,+1,...], ... },
        "pos_y_hv": { "0": [-1,+1,...], ... }
      }

    Returns the dict with NumPy arrays added:
      - "joint_hv_np": { name: np.ndarray(int8) }
      - "pos_x_hv_np": { ix: np.ndarray(int8) }
      - "pos_y_hv_np": { iy: np.ndarray(int8) }
    """
    with open(path, "r") as f:
        data = json.load(f)

    D = int(data["D"])

    # Convert joint hypervectors
    joint_hv_np: Dict[str, np.ndarray] = {}
    for name, hv_list in data["joint_hv"].items():
        hv = np.array(hv_list, dtype=np.int8)
        if hv.shape[0] != D:
            raise ValueError(f"Joint HV for {name} has dim {hv.shape[0]}, expected {D}")
        joint_hv_np[name] = hv

    # Convert pos_x / pos_y hypervectors, keys are string indices
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


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def bipolar_sign(vec: np.ndarray) -> np.ndarray:
    """
    Map real-valued vector to bipolar {-1, +1} (int8).
      >= 0  -> +1
      <  0  -> -1
    """
    s = (vec >= 0).astype(np.int8)
    s[s == 0] = -1
    return s


# Map from angle-encoder joint names to grid joint names
# (angle encoder joint names: left_shoulder, right_shoulder, etc.)
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

# Position columns expected in the CSV for each joint
# (If these columns are missing, we skip the grid part.)
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


# ---------------------------------------------------------------------
# Grid encoder (from a row)
# ---------------------------------------------------------------------

def encode_grid_from_row(
    row: "pd.Series",      # type: ignore[name-defined]
    grid_dict: Dict[str, Any],
) -> np.ndarray:
    """
    Encode grid-based HDC from joint positions in a CSV row.

    Requires that the row contains:
      - img_width, img_height
      - <joint>_x, <joint>_y for joints in JOINT_POS_COLUMNS

    If any critical columns are missing, returns a zero vector (all 0s).
    """
    D = grid_dict["D"]
    grid_x = int(grid_dict["grid_x"])
    grid_y = int(grid_dict["grid_y"])

    # Ensure we have image size
    if ("img_width" not in row) or ("img_height" not in row):
        # No image size -> can't compute grid location
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
            # Missing position for this joint -> skip
            continue

        x = float(row[col_x])
        y = float(row[col_y])

        if not np.isfinite(x) or not np.isfinite(y):
            continue

        # Convert to grid indices [0, grid_x-1] and [0, grid_y-1]
        ix = int(x / img_w * grid_x)
        iy = int(y / img_h * grid_y)
        ix = max(0, min(grid_x - 1, ix))
        iy = max(0, min(grid_y - 1, iy))

        # Joint symbol HV from grid dict
        grid_joint_name = ANGLE_JOINT_TO_GRID_NAME.get(angle_joint_name, None)
        if grid_joint_name is None or grid_joint_name not in joint_hv_np:
            continue

        J = joint_hv_np[grid_joint_name]          # (D,)
        PX = pos_x_hv_np.get(ix, None)            # (D,)
        PY = pos_y_hv_np.get(iy, None)            # (D,)

        if PX is None or PY is None:
            continue

        # Binding: J * PX * PY (all in {-1,+1})
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


# ---------------------------------------------------------------------
# Hybrid encoder (angles + grid)
# ---------------------------------------------------------------------

def encode_hybrid_from_row(
    row: "pd.Series",      # type: ignore[name-defined]
    grid_dict: Dict[str, Any],
    angle_dict: Dict[str, Any] = hdc_angle_dict,
) -> np.ndarray:
    """
    Encode a single CSV row into a hybrid hypervector using BINDING:

        H_hybrid = sign( H_angle ⊗ H_grid )

    If grid information is missing, returns H_angle.
    """

    # Angle HV: always available if angles exist
    H_angle = encode_pose_from_row(row, hdc_dict=angle_dict).astype(np.int8)

    # Grid HV: may be missing
    H_grid = encode_grid_from_row(row, grid_dict=grid_dict).astype(np.int8)

    # If grid HV is all zeros → fallback to angle HV only
    if not np.any(H_grid != 0):
        return H_angle

    # Binding = elementwise multiply (since HVs are ±1)
    H_bind = (H_angle.astype(np.int16) * H_grid.astype(np.int16)).astype(np.int32)

    # Convert back to bipolar ±1
    H_hybrid = bipolar_sign(H_bind).astype(np.int8)

    return H_hybrid


if __name__ == "__main__":
    # Tiny sanity check (this won't fully run without a real row & grid dict)
    print("hybrid_hdc_encoder module loaded.")
