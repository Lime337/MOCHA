import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from hashlib import md5

# ======================================================
# HERMES + MOCHA REVERSIBILITY ACROSS D = 512, 1024, 2048
# ======================================================
# This script extends the original HERMES AndMal experiment.
#
# For each D:
#   1) build the original clean HERMES class accumulators,
#   2) inject 30% mislabeled incremental sample contributions,
#   3) evaluate corrupted accuracy,
#   4) algebraically subtract the exact registered contributions,
#   5) verify recovery of the original accumulator state and accuracy.
#
# IMPORTANT:
# Unthresholded class accumulators are retained for MOCHA updates.
# sign() is applied only when forming class vectors for inference.

DIMENSIONS = [512, 1024, 2048]
MISLABEL_RATE = 0.30
CORRUPTION_SEED = 2026

SAMPLES_PER_FAMILY = 21
METHOD_NAME = "CSV_METHOD"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------
# Automatically find the nested AndMal folders/files
# ------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def find_file_recursive(root, filename):
    matches = []
    for dirpath, _, filenames in os.walk(root):
        if filename in filenames:
            matches.append(os.path.join(dirpath, filename))
    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename!r} anywhere under: {root}"
        )
    # Prefer the shortest path in case duplicate copies exist.
    matches.sort(key=lambda p: (len(Path(p).parts), len(p)))
    return matches[0]

BENIGN_FILE = find_file_recursive(BASE_DIR, "Ben0.csv")
ADWARE_FILE = find_file_recursive(BASE_DIR, "Adware.csv")
MALWARE_DIR = os.path.dirname(ADWARE_FILE)

MALWARE_FILES = [
    "Adware.csv",
    "Backdoor.csv",
    "Banker.csv",
    "Dropper.csv",
    "FileInfector.csv",
    "NoCategory.csv",
    "PUA.csv",
    "Ransomware.csv",
    "Riskware.csv",
    "Scareware.csv",
    "SMS.csv",
    "Spy.csv",
    "Trojan.csv",
    "Zeroday.csv",
]

# ======================================================
# DATA LOAD
# ======================================================
def load_benign(n=300):
    return pd.read_csv(BENIGN_FILE, header=None, nrows=n)

def load_malware_families():
    dfs = []
    for fname in MALWARE_FILES:
        path = os.path.join(MALWARE_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing malware family file: {path}")
        dfs.append(pd.read_csv(path, header=None, nrows=SAMPLES_PER_FAMILY))
    return pd.concat(dfs, ignore_index=True)

# ======================================================
# HDC HELPERS
# ======================================================
def seed_hash(s):
    return int(md5(s.encode()).hexdigest()[:8], 16)

def bipolar(seed, D):
    rng = np.random.RandomState(seed)
    v = rng.randint(0, 2, size=D).astype(np.int8)
    v[v == 0] = -1
    return torch.tensor(v, dtype=torch.int8, device=DEVICE)

def sign(a):
    out = torch.empty_like(a, dtype=torch.int8)
    out[a > 0] = 1
    out[a < 0] = -1
    zero = (a == 0)
    if torch.any(zero):
        out[zero] = -1
    return out

# ======================================================
# RUN ONE DIMENSION
# ======================================================
def run_dimension(D, df_ben, df_mal):
    # Dimension-specific caches.
    opcode_cache = {}
    method_cache = {}

    def opcode_vec(idx):
        name = f"OPCODE_{idx}"
        if name not in opcode_cache:
            opcode_cache[name] = bipolar(seed_hash("OPC::" + name), D)
        return opcode_cache[name]

    def method_vec():
        if METHOD_NAME not in method_cache:
            method_cache[METHOD_NAME] = bipolar(
                seed_hash("MTH::" + METHOD_NAME), D
            )
        return method_cache[METHOD_NAME]

    def encode_app(counts):
        total = sum(counts)
        if total == 0:
            return torch.zeros(D, dtype=torch.int8, device=DEVICE)

        max_c = max(counts)
        if max_c == 0:
            return torch.zeros(D, dtype=torch.int8, device=DEVICE)

        acc = torch.zeros(D, dtype=torch.float32, device=DEVICE)
        mvec = method_vec()

        for idx, c in enumerate(counts):
            if c <= 0:
                continue

            weight = np.log1p(c) / np.log1p(max_c)
            hv = opcode_vec(idx) * mvec
            acc += weight * hv.float()

        return sign(acc)

    def predict(X, acc_ben, acc_mal):
        C_ben = sign(acc_ben)
        C_mal = sign(acc_mal)

        preds = []
        for hv in X:
            sb = torch.dot(hv.float(), C_ben.float()).item()
            sm = torch.dot(hv.float(), C_mal.float()).item()
            preds.append(0 if sb > sm else 1)
        return np.asarray(preds, dtype=np.int32)

    opcode_cols = df_ben.columns[1:]

    X = []
    y = []

    for _, row in df_ben.iterrows():
        counts = [int(row[c]) for c in opcode_cols]
        X.append(encode_app(counts))
        y.append(0)

    for _, row in df_mal.iterrows():
        counts = [int(row[c]) for c in opcode_cols]
        X.append(encode_app(counts))
        y.append(1)

    # EXACT same split for every D.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    # --------------------------------------------------
    # Original clean HERMES class accumulators
    # --------------------------------------------------
    acc_ben = torch.zeros(D, dtype=torch.float32, device=DEVICE)
    acc_mal = torch.zeros(D, dtype=torch.float32, device=DEVICE)

    for hv, lab in zip(X_train, y_train):
        if lab == 0:
            acc_ben += hv.float()
        else:
            acc_mal += hv.float()

    acc_ben_base = acc_ben.clone()
    acc_mal_base = acc_mal.clone()

    baseline_pred = predict(X_test, acc_ben, acc_mal)
    baseline_acc = accuracy_score(y_test, baseline_pred)

    # --------------------------------------------------
    # Same corruption INDEX SET for every D
    # --------------------------------------------------
    rng = np.random.default_rng(CORRUPTION_SEED)
    train_idx = np.arange(len(X_train))
    n_bad = max(1, int(round(MISLABEL_RATE * len(X_train))))
    bad_idx = rng.choice(train_idx, size=n_bad, replace=False)

    update_log = []

    # Deliberately add selected training samples to WRONG class.
    for i in bad_idx:
        hv = X_train[int(i)].float()
        true_lab = int(y_train[int(i)])
        wrong_lab = 1 - true_lab

        if wrong_lab == 0:
            acc_ben += hv
        else:
            acc_mal += hv

        # MOCHA contribution registration.
        update_log.append((wrong_lab, hv.clone()))

    corrupted_pred = predict(X_test, acc_ben, acc_mal)
    corrupted_acc = accuracy_score(y_test, corrupted_pred)

    # --------------------------------------------------
    # MOCHA reversible removal
    # --------------------------------------------------
    for wrong_lab, hv in reversed(update_log):
        if wrong_lab == 0:
            acc_ben -= hv
        else:
            acc_mal -= hv

    restored_pred = predict(X_test, acc_ben, acc_mal)
    restored_acc = accuracy_score(y_test, restored_pred)

    max_ben_diff = float(
        torch.max(torch.abs(acc_ben - acc_ben_base)).item()
    )
    max_mal_diff = float(
        torch.max(torch.abs(acc_mal - acc_mal_base)).item()
    )
    max_accumulator_diff = max(max_ben_diff, max_mal_diff)

    return {
        "D": int(D),
        "bad_updates": int(n_bad),
        "mislabel_rate": float(MISLABEL_RATE),
        "baseline_acc": float(baseline_acc),
        "corrupted_acc": float(corrupted_acc),
        "restored_acc": float(restored_acc),
        "recovery": float(restored_acc - corrupted_acc),
        "restore_gap": float(abs(restored_acc - baseline_acc)),
        "max_accumulator_diff": max_accumulator_diff,
        "exact_restore": int(max_accumulator_diff == 0.0),
    }

# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    print("Using device:", DEVICE)
    print("Benign file:", BENIGN_FILE)
    print("Malware dir:", MALWARE_DIR)
    print("Dimensions:", DIMENSIONS)
    print(f"MOCHA corruption rate: {MISLABEL_RATE*100:.0f}%")

    print("\nLoading AndMal data once...")
    df_ben = load_benign(n=300)
    df_mal = load_malware_families()

    results = []

    for D in DIMENSIONS:
        print(f"\n================ D = {D} ================")
        row = run_dimension(D, df_ben, df_mal)
        results.append(row)

        print(
            f"D={D:4d} | "
            f"baseline={row['baseline_acc']:.4f} "
            f"corrupt={row['corrupted_acc']:.4f} "
            f"restored={row['restored_acc']:.4f} | "
            f"recovery={row['recovery']:+.4f} "
            f"gap={row['restore_gap']:.4f} "
            f"maxdiff={row['max_accumulator_diff']:.0f} "
            f"exact={row['exact_restore']}"
        )

    df = pd.DataFrame(results)

    out_csv = os.path.join(
        BASE_DIR,
        "hermes_mocha_all_dimensions_results.csv"
    )
    df.to_csv(out_csv, index=False)

    print("\n================ PAPER TABLE VIEW ================")

    paper = df[
        [
            "D",
            "baseline_acc",
            "corrupted_acc",
            "restored_acc",
            "recovery",
            "restore_gap",
            "exact_restore",
        ]
    ].copy()

    paper.columns = [
        "D",
        "Baseline",
        "Corrupted",
        "MOCHA Restored",
        "Recovery",
        "Restore Gap",
        "Exact Restore",
    ]

    print(
        paper.to_string(
            index=False,
            formatters={
                "Baseline": lambda x: f"{x:.4f}",
                "Corrupted": lambda x: f"{x:.4f}",
                "MOCHA Restored": lambda x: f"{x:.4f}",
                "Recovery": lambda x: f"{x:+.4f}",
                "Restore Gap": lambda x: f"{x:.4f}",
            },
        )
    )

    print("\n================ LATEX ROW VIEW ================")
    for _, r in df.iterrows():
        print(
            f"HDC $D$={int(r['D'])} & "
            f"{100*r['corrupted_acc']:.2f} & "
            f"{100*r['restored_acc']:.2f} \\\\"
        )

    print(f"\n[SAVED] {out_csv}")