# build_hdc_dictionary.py
import json
from pathlib import Path
import numpy as np
from scipy.stats import qmc  # make sure scipy is installed


# ========= CONFIG =========
D = 4096           # dimensionality
GRID_X = 9           # number of bins along x (width, horizontal)
GRID_Y = 16           # number of bins along y (height, vertical)
SEED = 12345          # for reproducibility

OUT_JSON = Path(r"SobolHybrid\hdc_pose_dict_4096.json")
OUT_NPZ  = OUT_JSON.with_suffix(".npz")   # -> ...\hdc_pose_dict_7000.npz
# ==========================

rng = np.random.default_rng(SEED)

def sobol_bipolar_vectors(num_vecs: int,
                          dim: int,
                          seed: int = 0,
                          threshold: float = 0.5,
                          scramble: bool = True) -> np.ndarray:
    """
    Generate `num_vecs` bipolar hypervectors of length `dim` using a Sobol sequence.
    Each coordinate is:
        -1 if sobol_value < threshold
        +1 otherwise

    Returns: ndarray of shape (num_vecs, dim), dtype int8 in {-1, +1}.
    """
    engine = qmc.Sobol(d=dim, scramble=scramble, seed=seed)
    # Samples in [0, 1), shape (num_vecs, dim)
    samples = engine.random(n=num_vecs)

    bipolar = np.where(samples < threshold, -1, 1).astype(np.int8)
    return bipolar


def random_bipolar(d: int) -> np.ndarray:
    """Random bipolar hypervector in {-1, +1}^D with ~50/50 distribution."""
    return np.where(rng.standard_normal(d) >= 0, 1, -1).astype(np.int8)


# MediaPipe / ML Kit 33 landmark names, in index order
JOINT_NAMES = [
    "NOSE",                 # 0
    "LEFT_EYE_INNER",       # 1
    "LEFT_EYE",             # 2
    "LEFT_EYE_OUTER",       # 3
    "RIGHT_EYE_INNER",      # 4
    "RIGHT_EYE",            # 5
    "RIGHT_EYE_OUTER",      # 6
    "LEFT_EAR",             # 7
    "RIGHT_EAR",            # 8
    "LEFT_MOUTH",           # 9
    "RIGHT_MOUTH",          # 10
    "LEFT_SHOULDER",        # 11
    "RIGHT_SHOULDER",       # 12
    "LEFT_ELBOW",           # 13
    "RIGHT_ELBOW",          # 14
    "LEFT_WRIST",           # 15
    "RIGHT_WRIST",          # 16
    "LEFT_PINKY",           # 17
    "RIGHT_PINKY",          # 18
    "LEFT_INDEX",           # 19
    "RIGHT_INDEX",          # 20
    "LEFT_THUMB",           # 21
    "RIGHT_THUMB",          # 22
    "LEFT_HIP",             # 23
    "RIGHT_HIP",            # 24
    "LEFT_KNEE",            # 25
    "RIGHT_KNEE",           # 26
    "LEFT_ANKLE",           # 27
    "RIGHT_ANKLE",          # 28
    "LEFT_HEEL",            # 29
    "RIGHT_HEEL",           # 30
    "LEFT_FOOT_INDEX",      # 31
    "RIGHT_FOOT_INDEX"      # 32
]


def build_joint_hvs() -> dict:
    """Return dict: joint_name -> np.ndarray (length D, -1 or +1)."""
    joint_hv = {}
    for name in JOINT_NAMES:
        hv = random_bipolar(D)
        joint_hv[name] = hv
    return joint_hv


#def build_pos_x_hvs(grid_x: int) -> dict:
    """
    Return dict: key 'ix' -> np.ndarray (length D) for X-axis positions.
    ix in [0, grid_x-1], along image width.
    """
    pos_x = {}
    for ix in range(grid_x):
        hv = random_bipolar(D)
        pos_x[str(ix)] = hv
    return pos_x


#def build_pos_y_hvs(grid_y: int) -> dict:
    """
    Return dict: key 'iy' -> np.ndarray (length D) for Y-axis positions.
    iy in [0, grid_y-1], along image height.
    """
    pos_y = {}
    for iy in range(grid_y):
        hv = random_bipolar(D)
        pos_y[str(iy)] = hv
    return pos_y

def build_pos_x_hvs_sobol(grid_x: int,
                          dim: int = D,
                          seed: int = SEED,
                          threshold: float = 0.5) -> dict:
    """
    Sobol-based position hypervectors for X-axis.
    Returns: {ix: hv} where hv is length-D bipolar vector.
    """
    # One Sobol sequence of length grid_x in D dimensions
    sobol_mat = sobol_bipolar_vectors(
        num_vecs=grid_x,
        dim=dim,
        seed=seed,
        threshold=threshold,
        scramble=True
    )

    pos_x = {}
    for ix in range(grid_x):
        pos_x[ix] = sobol_mat[ix]
    return pos_x


def build_pos_y_hvs_sobol(grid_y: int,
                          dim: int = D,
                          seed: int = SEED + 1000,
                          threshold: float = 0.5) -> dict:
    """
    Sobol-based position hypervectors for Y-axis.
    Uses a different seed so X/Y spaces are not identical.
    Returns: {iy: hv}.
    """
    sobol_mat = sobol_bipolar_vectors(
        num_vecs=grid_y,
        dim=dim,
        seed=seed,
        threshold=threshold,
        scramble=True
    )

    pos_y = {}
    for iy in range(grid_y):
        pos_y[iy] = sobol_mat[iy]
    return pos_y

def main():
    joint_hv = build_joint_hvs()
    pos_x_hv = build_pos_x_hvs_sobol(GRID_X, dim=D, seed=SEED, threshold=0.33)
    pos_y_hv = build_pos_y_hvs_sobol(GRID_Y, dim=D, seed=SEED, threshold=0.77)

    # ---------- JSON (same structure as before) ----------
    out_json = {
        "D": D,
        "seed": SEED,
        "grid_x": GRID_X,            # number of X bins (width)
        "grid_y": GRID_Y,            # number of Y bins (height)
        "joint_names": JOINT_NAMES,  # index -> name mapping
        # convert arrays to lists for JSON
        "joint_hv": {name: hv.tolist() for name, hv in joint_hv.items()},
        "pos_x_hv": {ix: hv.tolist() for ix, hv in pos_x_hv.items()},
        "pos_y_hv": {iy: hv.tolist() for iy, hv in pos_y_hv.items()},
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out_json, f)
    print(f"Saved HDC dictionary JSON to: {OUT_JSON}")

    # ---------- NPZ (compact, binary) ----------
    npz_data = {
        "D": np.array([D], dtype=np.int32),
        "seed": np.array([SEED], dtype=np.int32),
        "grid_x": np.array([GRID_X], dtype=np.int32),
        "grid_y": np.array([GRID_Y], dtype=np.int32),
        # you can reconstruct JOINT_NAMES from JSON, but keep them here too
        "joint_names": np.array(JOINT_NAMES, dtype=object),
    }

    # Add joint hypervectors: key "joint_<NAME>"
    for name, hv in joint_hv.items():
        key = f"joint_{name}"
        npz_data[key] = hv

    # Add position hypervectors
    for ix, hv in pos_x_hv.items():
        key = f"pos_x_{ix}"
        npz_data[key] = hv
    for iy, hv in pos_y_hv.items():
        key = f"pos_y_{iy}"
        npz_data[key] = hv

    np.savez(OUT_NPZ, **npz_data)
    print(f"Saved HDC dictionary NPZ to: {OUT_NPZ}")


if __name__ == "__main__":
    main()
