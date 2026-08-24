"""
hdc_angle_dict.py

Generates HDC hypervectors for:
  - 8 joint symbols (shoulder, elbow, hip, knee on left/right)
  - 18 angle bins for 0–180 degrees (10 degrees per bin)

Dimensionality: D = 8000 (±1 bipolar vectors).

By default:
  - Joint symbols are random bipolar vectors.
  - Angle-bin vectors are independent random bipolar vectors.
    (Continuity over angle is handled in the encoder via
     soft interpolation, not in the dictionary itself.)

You can import `hdc_angle_dict` from this file, or call
`generate_hdc_angle_dict()` to regenerate with different seeds/params.
"""

import numpy as np

# -----------------------------
# Configuration
# -----------------------------
DIMENSION = 4096
SEED = 42  # fix for reproducibility

# Angle range and binning
ANGLE_MIN_DEG = 0.0
ANGLE_MAX_DEG = 180.0
ANGLE_BIN_SIZE_DEG = 20.0

NUM_ANGLE_BINS = int((ANGLE_MAX_DEG - ANGLE_MIN_DEG) / ANGLE_BIN_SIZE_DEG)

# Optional extra constants (not strictly required, but kept for clarity)
D = DIMENSION
ANGLE_SEED = 12345

# Names for the 8 joint angle symbols
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


# -----------------------------
# Core helpers
# -----------------------------
def _random_bipolar_vector(dimension: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate a bipolar hypervector of shape (dimension,)
    with values in {-1, +1}, approximately 50/50.
    """
    hv = rng.choice([-1, 1], size=dimension)
    return hv.astype(np.int8)


def _create_symbol_vectors(dimension: int, rng: np.random.Generator) -> dict:
    """Create random symbol hypervectors for each joint angle."""
    symbols = {}
    for name in JOINT_SYMBOL_NAMES:
        symbols[name] = _random_bipolar_vector(dimension, rng)
    return symbols


def _create_angle_bin_vectors_random(
    dimension: int,
    num_bins: int,
    rng: np.random.Generator,
) -> dict:
    """
    Create independent random hypervectors for angle bins.

    This is the default strategy. Smoothness over angle is handled
    in the encoder (soft interpolation) rather than here.
    """
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
    """
    (Optional / experimental) Create unary/thermometer-style hypervectors
    for angle bins.

    We first sample a reference vector ref[i] ~ U[0,1) of length `dimension`.

    For each bin b, we compute its center angle, map it to a normalized
    scalar t in [0,1], and then set:

        hv_b[i] = +1 if ref[i] < t else -1

    So:
      - Small angles (small t) have mostly -1 entries.
      - Large angles (large t) have mostly +1 entries.
      - Nearby bins share many coordinates (high similarity).

    Not used by default, as it can over-smooth and hurt accuracy
    depending on the classifier; kept here for experimentation.
    """
    angle_span = angle_max_deg - angle_min_deg
    if num_bins <= 0 or angle_span <= 0:
        raise ValueError("Invalid angle configuration for unary angle bins.")

    bin_size = angle_span / num_bins

    # Reference vector in [0,1)
    ref = rng.random(dimension)

    angle_bins = {}

    for b in range(num_bins):
        # Angle interval for bin b: [angle_start, angle_end)
        angle_start = angle_min_deg + b * bin_size
        angle_end = angle_start + bin_size
        # Use the bin center as the representative angle
        angle_center = 0.5 * (angle_start + angle_end)

        # Normalize to [0,1]
        t = (angle_center - angle_min_deg) / angle_span
        t = float(np.clip(t, 0.0, 1.0))

        hv = np.where(ref < t, 1, -1).astype(np.int8)
        angle_bins[b] = hv

    return angle_bins, bin_size


# -----------------------------
# Public API
# -----------------------------
def generate_hdc_angle_dict(
    dimension: int = DIMENSION,
    num_angle_bins: int = NUM_ANGLE_BINS,
    seed: int = SEED,
) -> dict:
    """
    Generate the HDC angle dictionary with:
      - Random joint symbol vectors
      - Random (independent) angle-bin vectors

    Continuity/smoothness over angle is provided in the encoder
    (via soft interpolation between bins), not in the dictionary.
    """
    rng = np.random.default_rng(seed)

    symbols = _create_symbol_vectors(dimension, rng)

    # Default: random angle-bin vectors
    angle_bins = _create_angle_bin_vectors_random(
        dimension=dimension,
        num_bins=num_angle_bins,
        rng=rng,
    )
    angle_bin_size = 180.0 / num_angle_bins  # -> 10 degrees

    return {
        "dimension": dimension,
        "symbols": symbols,
        "angle_bins": angle_bins,
        "angle_bin_size_deg": angle_bin_size,
    }


# Default generated version
hdc_angle_dict = generate_hdc_angle_dict()


# -----------------------------
# Optional helpers
# -----------------------------
def save_hdc_angle_dict_npz(hdc_dict: dict, path: str):
    """Save symbol and angle vectors to .npz file."""
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
    """Convert hypervectors to pure Python lists for JSON export."""
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
