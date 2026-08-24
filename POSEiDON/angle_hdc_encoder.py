"""
hdc_angle_encoder.py

Encoder for angle-based HDC pose representation.

Uses:
  - 8 joint symbol hypervectors (from hdc_angle_dict.py)
  - 18 angle-bin hypervectors over 0–180 degrees (10° each)

Given a set of 8 joint angles (in degrees), it produces a
single bipolar hypervector (±1, int8) representing the pose.
"""

from typing import Dict, Any
import numpy as np

from angle_hdc_dict import hdc_angle_dict


# -----------------------------
# Helpers
# -----------------------------

def angle_to_bin(theta_deg: float,
                 bin_size_deg: float,
                 num_bins: int) -> int:
    """
    Map an angle [0,180] to a bin index (hard binning).
    Still available if you want to use it elsewhere,
    but the main encoder now uses soft interpolation.
    """
    if np.isnan(theta_deg):
        return -1

    theta = max(0.0, min(180.0 - 1e-6, float(theta_deg)))
    b = int(theta // bin_size_deg)

    return max(0, min(num_bins - 1, b))


def soft_angle_bin_hv(theta_deg: float,
                      bin_size_deg: float,
                      angle_bins: Dict[int, np.ndarray]) -> np.ndarray:
    """
    Map an angle (possibly continuous) to a *soft* angle hypervector
    by linearly interpolating between neighboring bins.

    - If theta is exactly on a bin, returns that bin's vector.
    - If theta is between bins k and k+1, returns:
          (1 - alpha) * angle_bins[k] + alpha * angle_bins[k+1],
      where alpha in [0,1] is the fractional position.

    Returned vector is float32; the final pose hypervector is
    binarized with bipolar_sign at the end.
    """
    if theta_deg is None or np.isnan(theta_deg):
        return None

    num_bins = len(angle_bins)
    max_angle = bin_size_deg * num_bins

    theta = float(np.clip(theta_deg, 0.0, max_angle - 1e-6))
    pos = theta / bin_size_deg  # in [0, num_bins)
    lower = int(np.floor(pos))
    upper = min(lower + 1, num_bins - 1)
    alpha = pos - lower  # in [0,1)

    v_lower = angle_bins[lower].astype(np.float32)
    if upper == lower:
        return v_lower

    v_upper = angle_bins[upper].astype(np.float32)
    hv = (1.0 - alpha) * v_lower + alpha * v_upper
    return hv


def bipolar_sign(vector: np.ndarray) -> np.ndarray:
    """Convert real vector -> bipolar ±1."""
    return np.where(vector >= 0, 1, -1).astype(np.int8)


# -----------------------------
# Encoder
# -----------------------------

JOINT_COLUMN_TO_SYMBOL = {
    "left_shoulder_angle":  "left_shoulder",
    "right_shoulder_angle": "right_shoulder",
    "left_elbow_angle":     "left_elbow",
    "right_elbow_angle":    "right_elbow",
    "left_hip_angle":       "left_hip",
    "right_hip_angle":      "right_hip",
    "left_knee_angle":      "left_knee",
    "right_knee_angle":     "right_knee",
}


def encode_pose_angles(
    joint_angles: Dict[str, float],
    hdc_dict: Dict[str, Any] = hdc_angle_dict,
) -> np.ndarray:

    dimension = hdc_dict["dimension"]
    symbols = hdc_dict["symbols"]
    angle_bins = hdc_dict["angle_bins"]
    bin_size = hdc_dict["angle_bin_size_deg"]
    num_bins = len(angle_bins)

    # Use float32 accumulation for interpolation + summation
    acc = np.zeros(dimension, dtype=np.float32)
    valid = False

    for col, sym in JOINT_COLUMN_TO_SYMBOL.items():
        if col not in joint_angles:
            continue
        a = joint_angles[col]
        if a is None or np.isnan(a):
            continue

        # Soft angle encoding instead of hard bin assignment
        L = soft_angle_bin_hv(a, bin_size, angle_bins)
        if L is None:
            continue

        S = symbols[sym].astype(np.float32)

        # Bind joint symbol with angle HV and accumulate
        Hjoint = S * L
        acc += Hjoint
        valid = True

    if not valid:
        return np.zeros(dimension, dtype=np.int8)

    return bipolar_sign(acc)


def encode_pose_from_row(row, hdc_dict=hdc_angle_dict):
    """Encode directly from a Pandas row."""
    joint_angles = {}
    for col in JOINT_COLUMN_TO_SYMBOL:
        if col in row:
            joint_angles[col] = float(row[col])
    return encode_pose_angles(joint_angles, hdc_dict)
