
import os
import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.metrics import accuracy_score

# ============================================================
# POSEiDON + MOCHA REVERSIBILITY USING SOBOL / LD ENCODING
# ============================================================
# Reconstructs the supplied SobolHybrid POSEiDON pipeline for:
#   D = 1024, 2048, 4096
#
# For each D:
#   1) generate the Sobol spatial dictionary using the supplied settings,
#   2) generate the matching angle dictionary,
#   3) encode/balance/split the same 4-class pose dataset,
#   4) build unthresholded class accumulators,
#   5) inject 30% mislabeled incremental pose contributions,
#   6) remove those registered contributions algebraically,
#   7) verify exact restoration.
#
# Published LD/Sobol accuracies are printed only as a comparison:
#   1K = 93.09%, 2K = 97.35%, 4K = 97.79%

DIMENSIONS = [1024, 2048, 4096]

PUBLISHED_LD = {
    1024: 0.9309,
    2048: 0.9735,
    4096: 0.9779,
}

GRID_X = 9
GRID_Y = 16

GRID_SEED = 12345
ANGLE_SEED = 42

# These match the supplied Sobol dictionary code.
SOBOL_X_THRESHOLD = 0.33
SOBOL_Y_THRESHOLD = 0.77

NUM_ANGLE_BINS = 9   # 180 deg / 20 deg from supplied Sobol angle dictionary

VAL_FRACTION = 0.20
SPLIT_SEED = 123

MISLABEL_RATE = 0.30
CORRUPTION_SEED = 2026

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# FILE DISCOVERY
# ============================================================
def find_pose_csv(root):
    preferred = ["4extracted_angles.csv"]
    for name in preferred:
        for dirpath, _, filenames in os.walk(root):
            if name in filenames:
                return os.path.join(dirpath, name)

    hits = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(".csv") and "extracted_angles" in f.lower():
                hits.append(os.path.join(dirpath, f))

    if not hits:
        raise FileNotFoundError(
            "Could not find 4extracted_angles.csv anywhere beneath the script folder."
        )

    hits.sort(key=lambda p: (p.count(os.sep), len(p)))
    return hits[0]

# ============================================================
# SOBOL DICTIONARY
# ============================================================
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
    "RIGHT_FOOT_INDEX",
]

ANGLE_SYMBOL_NAMES = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
]

ANGLE_COL_TO_SYMBOL = {
    "left_shoulder_angle": "left_shoulder",
    "right_shoulder_angle": "right_shoulder",
    "left_elbow_angle": "left_elbow",
    "right_elbow_angle": "right_elbow",
    "left_hip_angle": "left_hip",
    "right_hip_angle": "right_hip",
    "left_knee_angle": "left_knee",
    "right_knee_angle": "right_knee",
}

JOINT_POS_COLUMNS = {
    "left_shoulder": ("left_shoulder_x", "left_shoulder_y"),
    "right_shoulder": ("right_shoulder_x", "right_shoulder_y"),
    "left_elbow": ("left_elbow_x", "left_elbow_y"),
    "right_elbow": ("right_elbow_x", "right_elbow_y"),
    "left_hip": ("left_hip_x", "left_hip_y"),
    "right_hip": ("right_hip_x", "right_hip_y"),
    "left_knee": ("left_knee_x", "left_knee_y"),
    "right_knee": ("right_knee_x", "right_knee_y"),
}

ANGLE_TO_GRID = {
    "left_shoulder": "LEFT_SHOULDER",
    "right_shoulder": "RIGHT_SHOULDER",
    "left_elbow": "LEFT_ELBOW",
    "right_elbow": "RIGHT_ELBOW",
    "left_hip": "LEFT_HIP",
    "right_hip": "RIGHT_HIP",
    "left_knee": "LEFT_KNEE",
    "right_knee": "RIGHT_KNEE",
}

def random_bipolar_from_rng(rng, D):
    # Matches supplied grid code:
    # np.where(rng.standard_normal(d) >= 0, 1, -1)
    return np.where(rng.standard_normal(D) >= 0, 1, -1).astype(np.int8)

def sobol_bipolar_vectors(num_vecs, D, seed, threshold):
    engine = qmc.Sobol(d=D, scramble=True, seed=seed)
    samples = engine.random(n=num_vecs)
    return np.where(samples < threshold, -1, 1).astype(np.int8)

def build_grid_dict_sobol(D):
    # Matches supplied hcd_pose_dictionary Sobol version.
    rng = np.random.default_rng(GRID_SEED)

    joint_hv = {}
    for name in JOINT_NAMES:
        joint_hv[name] = random_bipolar_from_rng(rng, D)

    xmat = sobol_bipolar_vectors(
        GRID_X, D, GRID_SEED, SOBOL_X_THRESHOLD
    )
    ymat = sobol_bipolar_vectors(
        GRID_Y, D, GRID_SEED + 1000, SOBOL_Y_THRESHOLD
    )

    return {
        "D": D,
        "grid_x": GRID_X,
        "grid_y": GRID_Y,
        "joint_hv": joint_hv,
        "pos_x_hv": {i: xmat[i] for i in range(GRID_X)},
        "pos_y_hv": {i: ymat[i] for i in range(GRID_Y)},
    }

def build_angle_dict(D):
    # Matches supplied Sobol angle_hdc_dict:
    # same RNG is used first for symbols and then for angle-bin vectors.
    rng = np.random.default_rng(ANGLE_SEED)

    symbols = {
        name: rng.choice([-1, 1], size=D).astype(np.int8)
        for name in ANGLE_SYMBOL_NAMES
    }

    bins = {
        b: rng.choice([-1, 1], size=D).astype(np.int8)
        for b in range(NUM_ANGLE_BINS)
    }

    return {
        "dimension": D,
        "symbols": symbols,
        "angle_bins": bins,
        "angle_bin_size_deg": 180.0 / NUM_ANGLE_BINS,
    }

# ============================================================
# ENCODER
# ============================================================
def bipolar_sign(v):
    return np.where(v >= 0, 1, -1).astype(np.int8)

def soft_angle_bin_hv(theta_deg, bin_size_deg, angle_bins):
    if theta_deg is None or pd.isna(theta_deg):
        return None

    num_bins = len(angle_bins)
    max_angle = bin_size_deg * num_bins

    theta = float(np.clip(theta_deg, 0.0, max_angle - 1e-6))
    pos = theta / bin_size_deg
    lower = int(np.floor(pos))
    upper = min(lower + 1, num_bins - 1)
    alpha = pos - lower

    v_lower = angle_bins[lower].astype(np.float32)
    if upper == lower:
        return v_lower

    v_upper = angle_bins[upper].astype(np.float32)
    return (1.0 - alpha) * v_lower + alpha * v_upper

def encode_angle(row, angle_dict):
    D = angle_dict["dimension"]
    acc = np.zeros(D, dtype=np.float32)
    valid = False

    for col, sym in ANGLE_COL_TO_SYMBOL.items():
        if col not in row.index:
            continue

        angle = row[col]
        if pd.isna(angle):
            continue

        L = soft_angle_bin_hv(
            float(angle),
            angle_dict["angle_bin_size_deg"],
            angle_dict["angle_bins"],
        )

        if L is None:
            continue

        S = angle_dict["symbols"][sym].astype(np.float32)
        acc += S * L
        valid = True

    if not valid:
        return np.zeros(D, dtype=np.int8)

    return bipolar_sign(acc)

def encode_grid(row, grid_dict):
    D = grid_dict["D"]

    if "img_width" not in row.index or "img_height" not in row.index:
        return np.zeros(D, dtype=np.int8)

    img_w = float(row["img_width"])
    img_h = float(row["img_height"])

    if img_w <= 1e-6 or img_h <= 1e-6:
        return np.zeros(D, dtype=np.int8)

    acc = np.zeros(D, dtype=np.int32)
    valid = False

    for angle_joint_name, (cx, cy) in JOINT_POS_COLUMNS.items():
        if cx not in row.index or cy not in row.index:
            continue

        x = row[cx]
        y = row[cy]

        if pd.isna(x) or pd.isna(y):
            continue

        ix = int(float(x) / img_w * GRID_X)
        iy = int(float(y) / img_h * GRID_Y)

        ix = max(0, min(GRID_X - 1, ix))
        iy = max(0, min(GRID_Y - 1, iy))

        grid_joint_name = ANGLE_TO_GRID[angle_joint_name]

        J = grid_dict["joint_hv"][grid_joint_name]
        PX = grid_dict["pos_x_hv"][ix]
        PY = grid_dict["pos_y_hv"][iy]

        H_joint = (
            J.astype(np.int16)
            * PX.astype(np.int16)
            * PY.astype(np.int16)
        ).astype(np.int32)

        acc += H_joint
        valid = True

    if not valid:
        return np.zeros(D, dtype=np.int8)

    return bipolar_sign(acc)

def encode_hybrid(row, angle_dict, grid_dict):
    # Matches supplied hybrid_hdc_encoder:
    # H_hybrid = sign(H_angle * H_grid)
    H_angle = encode_angle(row, angle_dict)
    H_grid = encode_grid(row, grid_dict)

    if not np.any(H_grid != 0):
        return H_angle

    H_bind = (
        H_angle.astype(np.int16)
        * H_grid.astype(np.int16)
    ).astype(np.int32)

    return bipolar_sign(H_bind)

# ============================================================
# BALANCING / SPLIT
# ============================================================
def balance_and_split(X, y):
    rng = np.random.default_rng(SPLIT_SEED)

    label_to_indices = {}
    for i, label in enumerate(y):
        label_to_indices.setdefault(label, []).append(i)

    min_size = min(len(v) for v in label_to_indices.values())

    X_train_list = []
    y_train_list = []
    X_val_list = []
    y_val_list = []

    for label, indices in label_to_indices.items():
        indices = np.asarray(indices, dtype=np.int32)
        rng.shuffle(indices)
        indices = indices[:min_size]

        n_val = max(1, int(VAL_FRACTION * min_size))

        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        X_train_list.append(X[train_idx])
        y_train_list.extend([label] * len(train_idx))

        X_val_list.append(X[val_idx])
        y_val_list.extend([label] * len(val_idx))

    X_train = np.vstack(X_train_list)
    X_val = np.vstack(X_val_list)
    y_train = np.asarray(y_train_list, dtype=object)
    y_val = np.asarray(y_val_list, dtype=object)

    # Matches supplied trainer's final shuffles.
    perm_train = rng.permutation(len(X_train))
    perm_val = rng.permutation(len(X_val))

    return (
        X_train[perm_train],
        y_train[perm_train],
        X_val[perm_val],
        y_val[perm_val],
        sorted(label_to_indices.keys()),
    )

# ============================================================
# CLASS ACCUMULATORS / INFERENCE
# ============================================================
def build_accumulators(X, y, labels):
    D = X.shape[1]
    acc = {
        label: np.zeros(D, dtype=np.int32)
        for label in labels
    }

    for hv, lab in zip(X, y):
        acc[str(lab)] += hv.astype(np.int32)

    return acc

def predict(X, accumulators):
    labels = sorted(accumulators.keys())

    class_hvs = np.stack([
        bipolar_sign(accumulators[label])
        for label in labels
    ]).astype(np.float32)

    class_hvs /= (
        np.linalg.norm(class_hvs, axis=1, keepdims=True) + 1e-8
    )

    Xf = X.astype(np.float32)
    Xf /= np.linalg.norm(Xf, axis=1, keepdims=True) + 1e-8

    scores = Xf @ class_hvs.T
    pred_idx = np.argmax(scores, axis=1)

    return np.asarray(
        [labels[i] for i in pred_idx],
        dtype=object,
    )

# ============================================================
# RUN ONE DIMENSION
# ============================================================
def run_dimension(D, df):
    print(f"[D={D}] Building Sobol dictionaries...")
    grid_dict = build_grid_dict_sobol(D)
    angle_dict = build_angle_dict(D)

    print(f"[D={D}] Encoding poses...")
    X_list = []
    y_list = []

    for _, row in df.iterrows():
        hv = encode_hybrid(row, angle_dict, grid_dict)
        if not np.any(hv != 0):
            continue

        X_list.append(hv)
        y_list.append(str(row["label"]))

    X = np.stack(X_list).astype(np.int8)
    y = np.asarray(y_list, dtype=object)

    X_train, y_train, X_val, y_val, labels = balance_and_split(X, y)

    base_accumulators = build_accumulators(
        X_train, y_train, labels
    )

    baseline_pred = predict(X_val, base_accumulators)
    baseline_acc = accuracy_score(y_val, baseline_pred)

    rng = np.random.default_rng(CORRUPTION_SEED)

    n_bad = max(
        1,
        int(round(MISLABEL_RATE * len(X_train)))
    )

    bad_positions = rng.choice(
        len(X_train),
        size=n_bad,
        replace=False,
    )

    corrupted_accumulators = {
        label: acc.copy()
        for label, acc in base_accumulators.items()
    }

    update_log = []

    for pos in bad_positions:
        hv = X_train[int(pos)].astype(np.int32)
        true_label = str(y_train[int(pos)])

        choices = [
            label for label in labels
            if label != true_label
        ]

        wrong_label = str(rng.choice(choices))

        corrupted_accumulators[wrong_label] += hv

        update_log.append(
            (wrong_label, hv.copy())
        )

    corrupted_pred = predict(
        X_val,
        corrupted_accumulators,
    )

    corrupted_acc = accuracy_score(
        y_val,
        corrupted_pred,
    )

    # MOCHA algebraic removal.
    restored_accumulators = {
        label: acc.copy()
        for label, acc in corrupted_accumulators.items()
    }

    for wrong_label, hv in reversed(update_log):
        restored_accumulators[wrong_label] -= hv

    restored_pred = predict(
        X_val,
        restored_accumulators,
    )

    restored_acc = accuracy_score(
        y_val,
        restored_pred,
    )

    max_diff = max(
        int(np.max(np.abs(
            restored_accumulators[label]
            - base_accumulators[label]
        )))
        for label in labels
    )

    return {
        "D": D,
        "published_acc": PUBLISHED_LD[D],
        "baseline_acc": baseline_acc,
        "baseline_delta": baseline_acc - PUBLISHED_LD[D],
        "corrupted_acc": corrupted_acc,
        "restored_acc": restored_acc,
        "recovery": restored_acc - corrupted_acc,
        "restore_gap": abs(restored_acc - baseline_acc),
        "max_accumulator_diff": max_diff,
        "exact_restore": int(max_diff == 0),
        "valid_samples": len(X),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "bad_updates": n_bad,
    }

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    csv_path = find_pose_csv(BASE_DIR)

    print("[DATA] Pose CSV:", csv_path)

    df = pd.read_csv(csv_path)

    if "label" not in df.columns:
        raise ValueError(
            "4extracted_angles.csv must contain a 'label' column."
        )

    print("[DATA] Rows:", len(df))
    print(
        "[DATA] Classes:",
        df["label"].value_counts().to_dict()
    )

    rows = []

    for D in DIMENSIONS:
        print(
            f"\n================ D = {D} ================"
        )

        result = run_dimension(D, df)
        rows.append(result)

        print(
            f"D={D} | "
            f"baseline={result['baseline_acc']:.4f} "
            f"published={result['published_acc']:.4f} "
            f"delta={result['baseline_delta']:+.4f} | "
            f"corrupt={result['corrupted_acc']:.4f} "
            f"restored={result['restored_acc']:.4f} | "
            f"gap={result['restore_gap']:.4f} "
            f"maxdiff={result['max_accumulator_diff']} "
            f"exact={result['exact_restore']}"
        )

    results = pd.DataFrame(rows)

    out_csv = os.path.join(
        BASE_DIR,
        "poseidon_mocha_sobol_results.csv",
    )

    results.to_csv(out_csv, index=False)

    print(
        "\n================ PAPER TABLE VIEW ================"
    )

    table = results[
        [
            "D",
            "published_acc",
            "baseline_acc",
            "corrupted_acc",
            "restored_acc",
            "restore_gap",
            "exact_restore",
        ]
    ].copy()

    table.columns = [
        "D",
        "Published",
        "Baseline",
        "Corrupted",
        "MOCHA Restored",
        "Restore Gap",
        "Exact Restore",
    ]

    print(
        table.to_string(
            index=False,
            formatters={
                "Published": lambda x: f"{x:.4f}",
                "Baseline": lambda x: f"{x:.4f}",
                "Corrupted": lambda x: f"{x:.4f}",
                "MOCHA Restored": lambda x: f"{x:.4f}",
                "Restore Gap": lambda x: f"{x:.4f}",
            },
        )
    )

    print(
        "\n================ LATEX EXTENSION ROWS ================"
    )

    for _, r in results.iterrows():
        label = (
            "1K" if int(r["D"]) == 1024
            else "2K" if int(r["D"]) == 2048
            else "4K"
        )

        print(
            f"{label} & "
            f"{100*r['baseline_acc']:.2f} & "
            f"{100*r['corrupted_acc']:.2f} & "
            f"{100*r['restored_acc']:.2f} & "
            f"{100*r['restore_gap']:.2f} \\\\"
        )

    print(f"\n[SAVED] {out_csv}")
