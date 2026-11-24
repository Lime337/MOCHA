
import json
from pathlib import Path
import numpy as np
from scipy.stats import qmc


D = 4096
GRID_X = 9
GRID_Y = 16
SEED = 12345

OUT_JSON = Path(r"SobolHybrid\hdc_pose_dict_4096.json")
OUT_NPZ  = OUT_JSON.with_suffix(".npz")


rng = np.random.default_rng(SEED)

def sobol_bipolar_vectors(num_vecs: int,
                          dim: int,
                          seed: int = 0,
                          threshold: float = 0.5,
                          scramble: bool = True) -> np.ndarray:
    
    engine = qmc.Sobol(d=dim, scramble=scramble, seed=seed)

    samples = engine.random(n=num_vecs)

    bipolar = np.where(samples < threshold, -1, 1).astype(np.int8)
    return bipolar


def random_bipolar(d: int) -> np.ndarray:
    
    return np.where(rng.standard_normal(d) >= 0, 1, -1).astype(np.int8)


JOINT_NAMES = [
    "NOSE",
    "LEFT_EYE_INNER",
    "LEFT_EYE",
    "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER",
    "RIGHT_EYE",
    "RIGHT_EYE_OUTER",
    "LEFT_EAR",
    "RIGHT_EAR",
    "LEFT_MOUTH",
    "RIGHT_MOUTH",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_PINKY",
    "RIGHT_PINKY",
    "LEFT_INDEX",
    "RIGHT_INDEX",
    "LEFT_THUMB",
    "RIGHT_THUMB",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_HEEL",
    "RIGHT_HEEL",
    "LEFT_FOOT_INDEX",
    "RIGHT_FOOT_INDEX"
]


def build_joint_hvs() -> dict:
    
    joint_hv = {}
    for name in JOINT_NAMES:
        hv = random_bipolar(D)
        joint_hv[name] = hv
    return joint_hv


    """
    Return dict: key 'ix' -> np.ndarray (length D) for X-axis positions.
    ix in [0, grid_x-1], along image width.
    """
    pos_x = {}
    for ix in range(grid_x):
        hv = random_bipolar(D)
        pos_x[str(ix)] = hv
    return pos_x


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


    out_json = {
        "D": D,
        "seed": SEED,
        "grid_x": GRID_X,
        "grid_y": GRID_Y,
        "joint_names": JOINT_NAMES,

        "joint_hv": {name: hv.tolist() for name, hv in joint_hv.items()},
        "pos_x_hv": {ix: hv.tolist() for ix, hv in pos_x_hv.items()},
        "pos_y_hv": {iy: hv.tolist() for iy, hv in pos_y_hv.items()},
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out_json, f)
    print(f'Saved HDC dictionary JSON to: {OUT_JSON}')


    npz_data = {
        "D": np.array([D], dtype=np.int32),
        "seed": np.array([SEED], dtype=np.int32),
        "grid_x": np.array([GRID_X], dtype=np.int32),
        "grid_y": np.array([GRID_Y], dtype=np.int32),

        "joint_names": np.array(JOINT_NAMES, dtype=object),
    }


    for name, hv in joint_hv.items():
        key = f"joint_{name}"
        npz_data[key] = hv


    for ix, hv in pos_x_hv.items():
        key = f"pos_x_{ix}"
        npz_data[key] = hv
    for iy, hv in pos_y_hv.items():
        key = f"pos_y_{iy}"
        npz_data[key] = hv

    np.savez(OUT_NPZ, **npz_data)
    print(f'Saved HDC dictionary NPZ to: {OUT_NPZ}')


if __name__ == "__main__":
    main()
