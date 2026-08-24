import os
import numpy as np
import pandas as pd
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================
DATASET_CACHE_NPZ = r"pamap2_windows_w512_s256_m0p900_notemp_allsensors.npz"  # your all-sensors cache (no HR in feat_cols)

# Must match Exp1 settings you want to attribute sensors under
D = 4096
Q = 21
seed = 0
use_time = True          # means "use segment position IDs" in pooled encoder
clamp_z = 1.0
n_segments = 4
segment_stats = ("mean", "std")
use_delta = True

# ============================================================
# FAULT INJECTION (optional)
# ============================================================
ENABLE_CORRUPTION = True        # <- turn ON/OFF here
CORRUPT_SENSOR_PREFIX = "chest_"  # "hand_", "chest_", "ankle_"
CORRUPT_MODE = "drift+dropout+spike+shift+clip"
CORRUPT_SEED = 123

RUN_TAG = "faulty_" + CORRUPT_SENSOR_PREFIX.replace("_", "") if ENABLE_CORRUPTION else "clean"
OUT_DIR = f"exp_sensor_loo_exp1symbols_{RUN_TAG}"
os.makedirs(OUT_DIR, exist_ok=True)


# corruption strength (tune if needed)
CORRUPT_GAIN_STD = 0.8
CORRUPT_BIAS_STD = 1.5
CORRUPT_DROPOUT_PROB = 0.20
CORRUPT_SPIKE_PROB = 0.02
CORRUPT_SPIKE_SCALE = 8.0
CORRUPT_TIME_SHIFT_MAX = 20
CORRUPT_CLIP_K = 0.5


def corrupt_sensor_inplace(
    data_by_subject,
    feat_cols,
    sensor_prefix="chest_",              # "hand_", "chest_", or "ankle_"
    mode="drift+dropout+spike+shift+clip",
    gain_std=0.8,
    bias_std=1.5,
    dropout_prob=0.20,
    spike_prob=0.02,
    spike_scale=8.0,
    time_shift_max=20,
    clip_k=0.5,
    seed=123
):
    rng = np.random.default_rng(seed)

    idx = [i for i, c in enumerate(feat_cols) if c.startswith(sensor_prefix)]
    if len(idx) == 0:
        raise ValueError(f"No features matched sensor_prefix='{sensor_prefix}'")
    idx = np.asarray(idx, dtype=np.int32)

    out = {}
    for subj, d in data_by_subject.items():
        X = d["X"].astype(np.float32, copy=False)  # (N,T,F)
        y = d["y"]
        Xc = X.copy()

        N, T, _ = Xc.shape
        block = Xc[:, :, idx]  # view

        if "drift" in mode:
            gain = rng.normal(1.0, gain_std, size=(N, 1, len(idx))).astype(np.float32)
            bias = rng.normal(0.0, bias_std, size=(N, 1, len(idx))).astype(np.float32)
            block = block * gain + bias

        if "shift" in mode and time_shift_max > 0:
            shifts = rng.integers(-time_shift_max, time_shift_max + 1, size=(N,), dtype=np.int32)
            for i in range(N):
                s = int(shifts[i])
                if s != 0:
                    block[i] = np.roll(block[i], shift=s, axis=0)

        if "dropout" in mode and dropout_prob > 0:
            mask = (rng.random(size=(N, T, 1)) < dropout_prob).astype(np.float32)
            block = block * (1.0 - mask)

        if "spike" in mode and spike_prob > 0:
            spmask = (rng.random(size=(N, T, len(idx))) < spike_prob)
            spikes = rng.normal(0.0, spike_scale, size=(N, T, len(idx))).astype(np.float32)
            block = block + spmask.astype(np.float32) * spikes

        if "clip" in mode and clip_k is not None:
            med = np.median(block, axis=1, keepdims=True).astype(np.float32)
            block = np.clip(block, med - clip_k, med + clip_k)

        # write back corrupted block
        Xc[:, :, idx] = block.astype(np.float32, copy=False)
        out[subj] = {"X": Xc, "y": y}

    return out

# ============================================================
# DATASET CACHE IO
# ============================================================
def load_cached_dataset_npz(cache_file):
    z = np.load(cache_file, allow_pickle=True)
    feat_cols = list(z["feat_cols"].tolist())
    subjects = [str(s) for s in z["subjects"].tolist()]
    data_by_subject = {}
    for subj in subjects:
        data_by_subject[subj] = {
            "X": z[f"X_{subj}"].astype(np.float32, copy=False),  # (N,T,F)
            "y": z[f"y_{subj}"].astype(np.int32, copy=False),
        }
    return data_by_subject, feat_cols

# ============================================================
# FEATURE GROUPING (hand/chest/ankle blocks)
# ============================================================
def sensor_indices(feat_cols):
    idx_hand  = [i for i, c in enumerate(feat_cols) if c.startswith("hand_")]
    idx_chest = [i for i, c in enumerate(feat_cols) if c.startswith("chest_")]
    idx_ankle = [i for i, c in enumerate(feat_cols) if c.startswith("ankle_")]

    if not idx_hand or not idx_chest or not idx_ankle:
        raise ValueError(
            "Expected feat_cols to contain prefixes: hand_, chest_, ankle_. "
            f"Got counts: hand={len(idx_hand)} chest={len(idx_chest)} ankle={len(idx_ankle)}"
        )
    return {"hand": idx_hand, "chest": idx_chest, "ankle": idx_ankle}

def build_conditions(feat_cols):
    groups = sensor_indices(feat_cols)
    all_idx = list(range(len(feat_cols)))

    def drop(name):
        drop_set = set(groups[name])
        return [i for i in all_idx if i not in drop_set]

    return {
        "all": all_idx,
        "no_hand": drop("hand"),
        "no_chest": drop("chest"),
        "no_ankle": drop("ankle"),
    }

# ============================================================
# Exp1-consistent symbol generation
# ============================================================
def _rand_bipolar_vec(rng, D):
    v = rng.integers(0, 2, size=D, dtype=np.int8)
    return (v * 2 - 1).astype(np.int8)

def build_VAL_cim(D, Q, seed):
    rng = np.random.default_rng(seed + 12345)
    VAL = np.empty((Q, D), dtype=np.int8)
    VAL[0] = _rand_bipolar_vec(rng, D)

    flips_per_step = max(1, D // (Q - 1)) if Q > 1 else D
    for q in range(1, Q):
        VAL[q] = VAL[q - 1].copy()
        flip_idx = rng.choice(D, size=flips_per_step, replace=False)
        VAL[q][flip_idx] *= -1
    return VAL

def build_ID_feat_full(F_full, D, seed):
    rng = np.random.default_rng(seed + 999)
    ID_feat_full = (rng.integers(0, 2, size=(F_full, D), dtype=np.int8) * 2 - 1).astype(np.int8)
    return ID_feat_full

def build_ID_seg(D, n_segments, seed):
    rng_seg = np.random.default_rng(seed + 4242)
    ID_seg = (rng_seg.integers(0, 2, size=(n_segments, D), dtype=np.int8) * 2 - 1).astype(np.int8)
    return ID_seg

# ============================================================
# Normalization (same as Exp1)
# ============================================================
def compute_norm_stats(X_flat):
    X_flat = X_flat.astype(np.float32, copy=False)
    X_flat = np.where(np.isfinite(X_flat), X_flat, np.nan).astype(np.float32)

    mu = np.nanmean(X_flat, axis=0).astype(np.float32)
    sigma = np.nanstd(X_flat, axis=0).astype(np.float32)

    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    sigma = np.nan_to_num(sigma, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32)
    sigma = np.where(sigma == 0, 1.0, sigma).astype(np.float32)
    return mu, sigma

# ============================================================
# Encoder (pooled segments) - matches Exp1 logic
# ============================================================
class HDCEncoder:
    def __init__(self, D, Q, seed, use_time, clamp_z, n_segments, segment_stats, use_delta,
                 ID_feat, VAL, ID_seg=None):
        self.D = int(D)
        self.Q = int(Q)
        self.seed = int(seed)
        self.use_time = bool(use_time)
        self.clamp_z = float(clamp_z)
        self.n_segments = int(n_segments)
        self.segment_stats = tuple(segment_stats)
        self.use_delta = bool(use_delta)

        self.ID_feat = ID_feat.astype(np.int8, copy=False)  # (F,D)
        self.VAL = VAL.astype(np.int8, copy=False)          # (Q,D)
        self.ID_seg = ID_seg.astype(np.int8, copy=False) if (self.use_time and ID_seg is not None) else None

        if self.use_time and self.ID_seg is None:
            raise ValueError("use_time=True but ID_seg is None")

    def quantize(self, Z):
        Z = np.nan_to_num(Z, nan=0.0, posinf=self.clamp_z, neginf=-self.clamp_z)
        Z = np.clip(Z, -self.clamp_z, self.clamp_z)
        u = (Z + self.clamp_z) / (2 * self.clamp_z)
        q = np.floor(u * (self.Q - 1 + 1e-8)).astype(np.int32)
        return np.clip(q, 0, self.Q - 1)

    def _encode_stat_vector(self, stat_vec, mu, sigma_safe, seg_id=None):
        Z = (stat_vec - mu) / sigma_safe
        q = self.quantize(Z)
        v = self.VAL[q]                      # (F,D)
        hv = (self.ID_feat * v).sum(axis=0)  # (D,)
        if seg_id is not None:
            hv = hv * seg_id
        return hv.astype(np.int32, copy=False)

    def encode_window(self, W, mu, sigma):
        W = W.astype(np.float32, copy=False)
        W = np.where(np.isfinite(W), W, mu[None, :]).astype(np.float32)
        sigma_safe = np.where(sigma == 0, 1.0, sigma).astype(np.float32)

        T, _ = W.shape
        K = self.n_segments
        seg_len = T // K
        if seg_len <= 0:
            raise ValueError(f"win_size={T} too small for n_segments={K}")

        acc = np.zeros(self.D, dtype=np.int32)

        for k in range(K):
            s = k * seg_len
            e = (k + 1) * seg_len if k < K - 1 else T
            seg = W[s:e]

            seg_id = self.ID_seg[k] if self.use_time else None

            if "mean" in self.segment_stats:
                acc += self._encode_stat_vector(seg.mean(axis=0), mu, sigma_safe, seg_id=seg_id)
            if "std" in self.segment_stats:
                acc += self._encode_stat_vector(seg.std(axis=0), mu, sigma_safe, seg_id=seg_id)

            if self.use_delta and seg.shape[0] >= 2:
                d = np.abs(np.diff(seg, axis=0))
                acc += self._encode_stat_vector(d.mean(axis=0), mu, sigma_safe, seg_id=seg_id)

        return np.where(acc >= 0, 1, -1).astype(np.int8)

# ============================================================
# Prototypes + cosine-ish eval (same as Exp1 classifier)
# ============================================================
def train_prototypes(H, y, D):
    P = {}
    for hv, lab in zip(H, y):
        lab = int(lab)
        if lab not in P:
            P[lab] = np.zeros(D, dtype=np.int32)
        P[lab] += hv.astype(np.int32)
    labels = sorted(P.keys())
    return labels, P

def predict(H, labels, P):
    Pm = np.stack([P[l] for l in labels]).astype(np.float32)  # (C,D)
    Pn = Pm / (np.linalg.norm(Pm, axis=1, keepdims=True) + 1e-8)
    Hf = H.astype(np.float32)
    Hn = Hf / (np.linalg.norm(Hf, axis=1, keepdims=True) + 1e-8)
    scores = Hn @ Pn.T
    idx = np.argmax(scores, axis=1)
    return np.array([labels[i] for i in idx], dtype=np.int32)

def encode_dataset(encoder, X, mu, sigma):
    N = X.shape[0]
    H = np.empty((N, encoder.D), dtype=np.int8)
    for i in range(N):
        H[i] = encoder.encode_window(X[i], mu, sigma)
    return H

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print(f"[DATA] Loading: {DATASET_CACHE_NPZ}")
    data_by_subject, feat_cols = load_cached_dataset_npz(DATASET_CACHE_NPZ)
        # Optional: inject a fault into one sensor block (in-place corruption)
    if ENABLE_CORRUPTION:
        print(f"[FAULT] Injecting corruption into sensor block: {CORRUPT_SENSOR_PREFIX}")
        print(f"[FAULT] mode={CORRUPT_MODE} seed={CORRUPT_SEED}")
        data_by_subject = corrupt_sensor_inplace(
            data_by_subject=data_by_subject,
            feat_cols=feat_cols,
            sensor_prefix=CORRUPT_SENSOR_PREFIX,
            mode=CORRUPT_MODE,
            gain_std=CORRUPT_GAIN_STD,
            bias_std=CORRUPT_BIAS_STD,
            dropout_prob=CORRUPT_DROPOUT_PROB,
            spike_prob=CORRUPT_SPIKE_PROB,
            spike_scale=CORRUPT_SPIKE_SCALE,
            time_shift_max=CORRUPT_TIME_SHIFT_MAX,
            clip_k=CORRUPT_CLIP_K,
            seed=CORRUPT_SEED,
        )
    else:
        print("[FAULT] Corruption disabled (running clean dataset).")

    subjects = sorted(data_by_subject.keys(), key=int)

    conditions = build_conditions(feat_cols)

    # Build full symbol tables once (Exp1-consistent)
    F_full = len(feat_cols)
    VAL = build_VAL_cim(D=D, Q=Q, seed=seed)
    ID_feat_full = build_ID_feat_full(F_full=F_full, D=D, seed=seed)
    ID_seg = build_ID_seg(D=D, n_segments=n_segments, seed=seed) if use_time else None

    per_subject_rows = []
    summary_rows = []

    for cond_name, keep_idx in conditions.items():
        keep_idx = np.asarray(keep_idx, dtype=np.int32)
        print(f"\n=== CONDITION: {cond_name}  (F={len(keep_idx)}/{F_full}) ===")

        # Condition-specific feature IDs: subset of full IDs (best ablation fairness)
        ID_feat = ID_feat_full[keep_idx]  # (F_kept, D)

        cond_accs = []

        for test_subj in subjects:
            train_subjects = [s for s in subjects if s != test_subj]

            # Stack train
            Xtr = np.concatenate([data_by_subject[s]["X"][:, :, keep_idx] for s in train_subjects], axis=0)
            ytr = np.concatenate([data_by_subject[s]["y"] for s in train_subjects], axis=0)
            Xte = data_by_subject[test_subj]["X"][:, :, keep_idx]
            yte = data_by_subject[test_subj]["y"]

            # Train-only mu/sigma on flattened samples
            mu, sigma = compute_norm_stats(Xtr.reshape(-1, Xtr.shape[-1]))

            encoder = HDCEncoder(
                D=D, Q=Q, seed=seed, use_time=use_time, clamp_z=clamp_z,
                n_segments=n_segments, segment_stats=segment_stats, use_delta=use_delta,
                ID_feat=ID_feat, VAL=VAL, ID_seg=ID_seg
            )

            Htr = encode_dataset(encoder, Xtr, mu, sigma)
            Hte = encode_dataset(encoder, Xte, mu, sigma)

            labels, P = train_prototypes(Htr, ytr, D=D)
            yhat = predict(Hte, labels, P)
            acc = float((yhat == yte).mean())

            cond_accs.append(acc)
            per_subject_rows.append({
                "condition": cond_name,
                "test_subject": int(test_subj),
                "acc": acc,
                "test_windows": int(len(yte)),
                "train_windows": int(len(ytr)),
                "n_features": int(len(keep_idx)),
                "fault_enabled": int(1 if ENABLE_CORRUPTION else 0),
                "fault_sensor": CORRUPT_SENSOR_PREFIX if ENABLE_CORRUPTION else "",
                "fault_mode": CORRUPT_MODE if ENABLE_CORRUPTION else "",
                "fault_seed": int(CORRUPT_SEED) if ENABLE_CORRUPTION else -1
            })

            print(f"[{cond_name}] test_subj={test_subj}  acc={acc*100:.2f}%")

        mean_acc = float(np.mean(cond_accs))
        std_acc = float(np.std(cond_accs, ddof=1)) if len(cond_accs) > 1 else 0.0
        summary_rows.append({
            "condition": cond_name,
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "subjects": int(len(cond_accs)),
            "n_features": int(len(keep_idx)),
        })

    per_subject_df = pd.DataFrame(per_subject_rows).sort_values(["condition", "test_subject"])
    summary_df = pd.DataFrame(summary_rows)

    

    # deltas vs all
    if (summary_df["condition"] == "all").any():
        base = float(summary_df.loc[summary_df["condition"] == "all", "mean_acc"].iloc[0])
        summary_df["delta_vs_all"] = summary_df["mean_acc"] - base
    else:
        summary_df["delta_vs_all"] = np.nan

        # Heuristic: if one removal gives the largest improvement vs all, call it the "likely faulty sensor"
    if (summary_df["condition"] == "all").any():
        cand = summary_df[summary_df["condition"] != "all"].copy()
        if len(cand) > 0:
            best = cand.sort_values("delta_vs_all", ascending=False).iloc[0]
            likely = str(best["condition"])
            print("\n[DIAG] Likely faulty sensor (largest recovery vs all):", likely,
                  f"(delta_vs_all={float(best['delta_vs_all'])*100:.2f}%)")

    per_subject_path = os.path.join(OUT_DIR, "sensor_loo_per_subject.csv")
    summary_path = os.path.join(OUT_DIR, "sensor_loo_summary.csv")
    per_subject_df.to_csv(per_subject_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\n==============================")
    print("SENSOR LEAVE-ONE-OUT SUMMARY")
    print("==============================")
    print(summary_df.sort_values("mean_acc", ascending=False).to_string(index=False))

    if (summary_df["condition"] == "all").any():
        improves = summary_df[(summary_df["condition"] != "all") & (summary_df["delta_vs_all"] > 0)].copy()
        if len(improves) > 0:
            print("\nRemoving these IMPROVED accuracy (likely noisy sensor group):")
            print(improves.sort_values("delta_vs_all", ascending=False)[["condition","delta_vs_all","mean_acc"]].to_string(index=False))
        else:
            print("\nNo leave-one-out condition improved accuracy vs all.")

        drops = summary_df[summary_df["condition"] != "all"].copy().sort_values("delta_vs_all")
        print("\nLargest drops when removed (most informative sensor group):")
        print(drops[["condition","delta_vs_all","mean_acc"]].to_string(index=False))

    print(f"\n[SAVED] {per_subject_path}")
    print(f"[SAVED] {summary_path}")