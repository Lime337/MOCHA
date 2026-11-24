

import numpy as np


DIMENSION = 4096
SEED = 42


ANGLE_MIN_DEG = 0.0
ANGLE_MAX_DEG = 180.0
ANGLE_BIN_SIZE_DEG = 20.0

NUM_ANGLE_BINS = int((ANGLE_MAX_DEG - ANGLE_MIN_DEG) / ANGLE_BIN_SIZE_DEG)


D = DIMENSION
ANGLE_SEED = 12345


JOINT_SYMBOL_NAMES = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
]


def _random_bipolar_vector(dimension: int, rng: np.random.Generator) -> np.ndarray:
    
    hv = rng.choice([-1, 1], size=dimension)
    return hv.astype(np.int8)


def _create_symbol_vectors(dimension: int, rng: np.random.Generator) -> dict:
    
    symbols = {}
    for name in JOINT_SYMBOL_NAMES:
        symbols[name] = _random_bipolar_vector(dimension, rng)
    return symbols


def _create_angle_bin_vectors_random(
    dimension: int,
    num_bins: int,
    rng: np.random.Generator,
) -> dict:
    
    angle_bins = {}
    for b in range(num_bins):
        angle_bins[b] = _random_bipolar_vector(dimension, rng)
    return angle_bins


def _create_angle_bin_vectors_unary(
    dimension: int,
    num_bins: int,
    rng: np.random.Generator,
    angle_min_deg: float = ANGLE_MIN_DEG,
    angle_max_deg: float = ANGLE_MAX_DEG,
) -> tuple[dict, float]:
    
    angle_span = angle_max_deg - angle_min_deg
    if num_bins <= 0 or angle_span <= 0:
        raise ValueError("Invalid angle configuration for unary angle bins.")

    bin_size = angle_span / num_bins


    ref = rng.random(dimension)

    angle_bins = {}

    for b in range(num_bins):

        angle_start = angle_min_deg + b * bin_size
        angle_end = angle_start + bin_size

        angle_center = 0.5 * (angle_start + angle_end)


        t = (angle_center - angle_min_deg) / angle_span
        t = float(np.clip(t, 0.0, 1.0))

        hv = np.where(ref < t, 1, -1).astype(np.int8)
        angle_bins[b] = hv

    return angle_bins, bin_size


def generate_hdc_angle_dict(
    dimension: int = DIMENSION,
    num_angle_bins: int = NUM_ANGLE_BINS,
    seed: int = SEED,
) -> dict:
    
    rng = np.random.default_rng(seed)

    symbols = _create_symbol_vectors(dimension, rng)


    angle_bins = _create_angle_bin_vectors_random(
        dimension=dimension,
        num_bins=num_angle_bins,
        rng=rng,
    )
    angle_bin_size = 180.0 / num_angle_bins

    return {
        "dimension": dimension,
        "symbols": symbols,
        "angle_bins": angle_bins,
        "angle_bin_size_deg": angle_bin_size,
    }


hdc_angle_dict = generate_hdc_angle_dict()


def save_hdc_angle_dict_npz(hdc_dict: dict, path: str):
    
    npz_data = {
        "dimension": np.array([hdc_dict["dimension"]], dtype=np.int32),
        "angle_bin_size_deg": np.array(
            [hdc_dict["angle_bin_size_deg"]], dtype=np.float32
        ),
    }

    for name, hv in hdc_dict["symbols"].items():
        key = f"symbol_{name}"
        npz_data[key] = hv

    for b, hv in hdc_dict["angle_bins"].items():
        key = f"angle_bin_{b}"
        npz_data[key] = hv

    np.savez(path, **npz_data)


def hdc_angle_dict_to_json_friendly(hdc_dict: dict) -> dict:
    
    json_dict = {
        "dimension": hdc_dict["dimension"],
        "angle_bin_size_deg": float(hdc_dict["angle_bin_size_deg"]),
        "symbols": {},
        "angle_bins": {},
    }

    for name, hv in hdc_dict["symbols"].items():
        json_dict["symbols"][name] = hv.tolist()

    for b, hv in hdc_dict["angle_bins"].items():
        json_dict["angle_bins"][str(b)] = hv.tolist()

    return json_dict
