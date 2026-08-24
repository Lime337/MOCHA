import os
import glob
import numpy as np
import pandas as pd
from collections import Counter

# ============================================================
# Noisy Sensor Experiment (APPROX KEY-UNBIND VERSION)
#   - Same pooled/segmented stats encoding (K=4 pools)
#   - Adds a "corrupted sensor" derived from real channels (noise_*)
#   - Evaluates LOSO:
#       (1) Train/test with corrupted sensor present (same as before)
#       (2) Approx algebraic removal using SENSOR KEY unbinding:
#           - Estimate the noise sensor component from H_total (and prototypes)
#             via:   F_hat = S_noise ⊙ H_total
#           - Reconstruct: H_noise_hat = S_noise ⊙ F_hat
#             (which equals H_total ideally; we instead use a projection trick below)
#
# IMPORTANT NOTE (why it's "approx"):
#   To make unbinding meaningful, we MUST encode sensors in a hierarchical way:
#       H = Σ_s  ( S_s ⊙ F_s )
#   where F_s is the per-sensor bundled HV (from its features+segments+stats),
#   and S_s is a random bipolar "sensor key".
#
#   Then:
#       S_noise ⊙ H = F_noise + Σ_{s≠noise} (S_noise ⊙ S_s) ⊙ F_s
#   The cross-terms behave like near-orthogonal noise in high D, so
#       S_noise ⊙ H ≈ F_noise
#
#   To remove noise without explicitly computing F_noise from feature indices:
#       H_clean ≈ H - (S_noise ⊙ (S_noise ⊙ H))_filtered
#   Since S_noise ⊙ S_noise = 1, naive rebinding gives H back. So we instead
#   use a "projection" estimate:
#       H_noise_hat ≈ S_noise ⊙ sign(S_noise ⊙ H)
#   which acts like extracting a stable estimate of F_noise then rebinding.
#
#   This is intentionally approximate; it won't match the exact subtraction method.
# ============================================================

# -----------------------------
# Dataset config
# -----------------------------
COLS = ["timestamp", "activity_id", "heart_rate"] + \
       [f"hand_{i}" for i in range(17)] + \
       [f"chest_{i}" for i in range(17)] + \
       [f"ankle_{i}" for i in range(17)]

KEEP_ACTS = {1, 2, 3, 4, 5, 6, 12, 13}  # 8-class set

# -----------------------------
# Utilities: dataset windowing + caching
# -----------------------------
def subject_id_from_path(path: str) -> str:
    base = os.path.basename(path)
    digits = "".join([c for c in base if c.isdigit()])
    return digits

def pruned_cols(include_temp: bool = False, wrist_only: bool = False):
    keep = []
    if include_temp:
        keep.append(0)
    keep += [1, 2, 3, 7, 8, 9, 10, 11, 12]  # 9 channels (+1 if temp)

    locs = ["hand"] if wrist_only else ["hand", "chest", "ankle"]
    cols = []
    for loc in locs:
        cols += [f"{loc}_{i}" for i in keep]
    return cols

def make_windows(df, feat_cols, win_size=512, stride=256, majority=0.9):
    df = df.reset_index(drop=True)

    feats = df[feat_cols].to_numpy(np.float32)
    labels = df["activity_id"].to_numpy(np.int32)

    X, y = [], []
    for start in range(0, len(df) - win_size + 1, stride):
        end = start + win_size

        w_labels = labels[start:end]
        vals, counts = np.unique(w_labels, return_counts=True)
        k = np.argmax(counts)

        if counts[k] / win_size < majority:
            continue

        X.append(feats[start:end])
        y.append(vals[k])

    if len(X) == 0:
        return np.empty((0, win_size, len(feat_cols)), dtype=np.float32), np.empty((0,), dtype=np.int32)

    return np.stack(X).astype(np.float32), np.array(y, dtype=np.int32)

def _cache_path(protocol_dir, win_size, stride, majority, include_temp, wrist_only, cache_dir):
    base_dir = cache_dir if cache_dir is not None else os.path.dirname(os.path.abspath(protocol_dir))
    os.makedirs(base_dir, exist_ok=True)
    maj_s = f"{majority:.3f}".replace(".", "p")
    fname = (
        f"pamap2_windows_"
        f"w{win_size}_s{stride}_m{maj_s}_"
        f"{'temp' if include_temp else 'notemp'}_"
        f"{'wrist' if wrist_only else 'allsensors'}"
        f".npz"
    )
    return os.path.join(base_dir, fname)

def save_cached_dataset_npz(cache_file, data_by_subject, feat_cols):
    payload = {
        "feat_cols": np.array(feat_cols, dtype=object),
        "subjects": np.array(sorted(data_by_subject.keys(), key=int), dtype=object),
    }
    for subj, d in data_by_subject.items():
        payload[f"X_{subj}"] = d["X"].astype(np.float32, copy=False)
        payload[f"y_{subj}"] = d["y"].astype(np.int32, copy=False)
    np.savez_compressed(cache_file, **payload)

def load_cached_dataset_npz(cache_file):
    z = np.load(cache_file, allow_pickle=True)
    feat_cols = list(z["feat_cols"].tolist())
    subjects = list(z["subjects"].tolist())
    data_by_subject = {}
    for subj in subjects:
        data_by_subject[str(subj)] = {
            "X": z[f"X_{subj}"].astype(np.float32, copy=False),
            "y": z[f"y_{subj}"].astype(np.int32, copy=False),
        }
    return data_by_subject, feat_cols

def load_pamap2_by_subject(protocol_dir=r"Protocol",
                          win_size=512,
                          stride=256,
                          majority=0.9,
                          include_temp=False,
                          wrist_only=False,
                          use_cache=True,
                          cache_dir=None):
    cache_file = _cache_path(protocol_dir, win_size, stride, majority, include_temp, wrist_only, cache_dir)

    if use_cache and os.path.exists(cache_file):
        print(f"[CACHE] Loading windowed PAMAP2 from: {cache_file}")
        data_by_subject, feat_cols = load_cached_dataset_npz(cache_file)
        print(f"[CACHE] Loaded subjects={len(data_by_subject)} total_windows={sum(v['X'].shape[0] for v in data_by_subject.values())}")
        return data_by_subject, feat_cols

    files = sorted(glob.glob(os.path.join(protocol_dir, "subject*.dat")))
    if not files:
        raise FileNotFoundError(f"No subject*.dat files found in: {protocol_dir}")

    feat_cols = pruned_cols(include_temp=include_temp, wrist_only=wrist_only)

    data_by_subject = {}
    for fp in files:
        subj = subject_id_from_path(fp)
        df = pd.read_csv(fp, sep=r"\s+", header=None, names=COLS, engine="python")

        df = df[df["activity_id"] != 0].copy()
        df = df[df["activity_id"].isin(KEEP_ACTS)].copy()

        X, y = make_windows(df, feat_cols, win_size=win_size, stride=stride, majority=majority)

        if X.shape[0] == 0:
            print(f"subject{subj}: skipped (no valid windows)")
            continue

        data_by_subject[subj] = {"X": X, "y": y}
        print(f"subject{subj}: rows={len(df):>7} windows={X.shape[0]:>5} feats={X.shape[2]}")

    total_counts = Counter()
    for subj, d in data_by_subject.items():
        total_counts.update(d["y"])
    print("\nGlobal label counts:", dict(total_counts))

    if use_cache:
        print(f"[CACHE] Saving windowed PAMAP2 to: {cache_file}")
        save_cached_dataset_npz(cache_file, data_by_subject, feat_cols)

    return data_by_subject, feat_cols


# -----------------------------
# Corrupted sensor derived from REAL sensor channels
# -----------------------------
def add_corrupted_sensor_from_real(
    data_by_subject,
    feat_cols,
    source_prefix="hand_",
    n_noise_feats=27,
    mode="drift+clip+dropout+spike",
    gain_std=0.8,
    bias_std=1.5,
    clip_k=0.5,
    dropout_prob=0.20,
    spike_prob=0.02,
    spike_scale=8.0,
    time_shift_max=20,
    seed=0
):
    rng = np.random.default_rng(seed)

    src_idx = [i for i, c in enumerate(feat_cols) if c.startswith(source_prefix)]
    if len(src_idx) == 0:
        raise ValueError(f"No features matched source_prefix='{source_prefix}'")

    src_idx = np.array(src_idx, dtype=np.int32)
    choice = rng.choice(src_idx, size=(n_noise_feats,), replace=(n_noise_feats > len(src_idx)))

    new_feat_cols = list(feat_cols) + [f"noise_{i}" for i in range(n_noise_feats)]
    out = {}

    for subj, d in data_by_subject.items():
        X = d["X"].astype(np.float32, copy=False)  # (N,T,F)
        y = d["y"]
        N, T, F = X.shape

        noise = X[:, :, choice].copy()  # (N,T,n_noise_feats)

        if "drift" in mode:
            gain = rng.normal(1.0, gain_std, size=(N, 1, n_noise_feats)).astype(np.float32)
            bias = rng.normal(0.0, bias_std, size=(N, 1, n_noise_feats)).astype(np.float32)
            noise = noise * gain + bias

        if "shift" in mode and time_shift_max > 0:
            shifts = rng.integers(-time_shift_max, time_shift_max + 1, size=(N,), dtype=np.int32)
            for i in range(N):
                s = int(shifts[i])
                if s != 0:
                    noise[i] = np.roll(noise[i], shift=s, axis=0)

        if "dropout" in mode and dropout_prob > 0:
            mask = (rng.random(size=(N, T, 1)) < dropout_prob).astype(np.float32)
            noise = noise * (1.0 - mask)

        if "spike" in mode and spike_prob > 0:
            spmask = (rng.random(size=(N, T, n_noise_feats)) < spike_prob)
            spikes = rng.normal(0.0, spike_scale, size=(N, T, n_noise_feats)).astype(np.float32)
            noise = noise + spmask.astype(np.float32) * spikes

        if "clip" in mode and clip_k is not None:
            med = np.median(noise, axis=1, keepdims=True).astype(np.float32)
            lo = med - clip_k
            hi = med + clip_k
            noise = np.clip(noise, lo, hi)

        X_aug = np.concatenate([X, noise.astype(np.float32)], axis=2).astype(np.float32, copy=False)
        out[subj] = {"X": X_aug, "y": y}

    return out, new_feat_cols


# -----------------------------
# Pooled/segmented HDC Encoder with SENSOR KEYS (hierarchical)
#   H = Σ_s ( S_s ⊙ F_s )
#   where F_s is formed from ONLY that sensor's features but uses
#   the same segment/time/stat pooling as before.
# -----------------------------
class HDCEncoderPooledSensorKeys:
    def __init__(self, D=4096, Q=21, seed=0, use_time=True, win_size=512, clamp_z=1.0,
                 n_segments=4, segment_stats=("mean", "std"), use_delta=True):
        self.D = int(D)
        self.Q = int(Q)
        self.seed = int(seed)
        self.use_time = bool(use_time)
        self.win_size = int(win_size)
        self.clamp_z = float(clamp_z)

        self.n_segments = int(n_segments)
        self.segment_stats = tuple(segment_stats)
        self.use_delta = bool(use_delta)

        self.feat_cols = None
        self.ID_feat = None
        self.VAL = None
        self.ID_seg = None

        # sensor keys: name -> (D,) int8 bipolar
        self.S_keys = {}

        self.VAL = self._precompute_val_cim()

        if self.use_time:
            rng_seg = np.random.default_rng(self.seed + 4242)
            self.ID_seg = (rng_seg.integers(0, 2, size=(self.n_segments, self.D), dtype=np.int8) * 2 - 1).astype(np.int8)

    def _rand_bipolar(self, rng):
        v = rng.integers(0, 2, size=self.D, dtype=np.int8)
        return (v * 2 - 1).astype(np.int8)

    def _precompute_val_cim(self):
        rng = np.random.default_rng(self.seed + 12345)
        VAL = np.empty((self.Q, self.D), dtype=np.int8)
        VAL[0] = self._rand_bipolar(rng)
        flips_per_step = max(1, self.D // (self.Q - 1)) if self.Q > 1 else self.D
        for q in range(1, self.Q):
            VAL[q] = VAL[q - 1].copy()
            flip_idx = rng.choice(self.D, size=flips_per_step, replace=False)
            VAL[q][flip_idx] *= -1
        return VAL

    def build_feature_ids(self, feat_cols):
        self.feat_cols = list(feat_cols)
        rng = np.random.default_rng(self.seed + 999)
        F = len(self.feat_cols)
        self.ID_feat = (rng.integers(0, 2, size=(F, self.D), dtype=np.int8) * 2 - 1).astype(np.int8)

    def build_sensor_keys(self, sensor_names):
        rng = np.random.default_rng(self.seed + 2026)
        self.S_keys = {}
        for name in sensor_names:
            self.S_keys[name] = self._rand_bipolar(rng)

    def quantize(self, Z):
        Z = np.nan_to_num(Z, nan=0.0, posinf=self.clamp_z, neginf=-self.clamp_z)
        Z = np.clip(Z, -self.clamp_z, self.clamp_z)
        u = (Z + self.clamp_z) / (2 * self.clamp_z)
        q = np.floor(u * (self.Q - 1 + 1e-8)).astype(np.int32)
        return np.clip(q, 0, self.Q - 1)

    def _encode_stat_vector_subset(self, stat_vec, mu, sigma_safe, feat_idx, seg_id=None):
        fi = feat_idx
        Z = (stat_vec[fi] - mu[fi]) / sigma_safe[fi]
        q = self.quantize(Z)
        v = self.VAL[q]                   # (|fi|,D) int8
        hv = (self.ID_feat[fi] * v).sum(axis=0)  # (D,) int32
        if seg_id is not None:
            hv = hv * seg_id
        return hv.astype(np.int32, copy=False)

    def encode_window_acc(self, W, mu, sigma, sensor_feat_idx_map):
        """
        Returns:
          acc_total: (D,) int32 where acc_total = Σ_s ( S_s ⊙ F_s )
        """
        if self.ID_feat is None:
            raise RuntimeError("Encoder has no ID_feat. Call build_feature_ids().")
        if not self.S_keys:
            raise RuntimeError("No sensor keys. Call build_sensor_keys().")

        W = W.astype(np.float32, copy=False)
        W = np.where(np.isfinite(W), W, mu[None, :]).astype(np.float32)
        sigma_safe = np.where(sigma == 0, 1.0, sigma).astype(np.float32)

        T, _ = W.shape
        K = self.n_segments
        seg_len = T // K
        if seg_len <= 0:
            raise ValueError(f"win_size={T} too small for n_segments={K}")

        acc_total = np.zeros(self.D, dtype=np.int32)

        for sensor_name, feat_idx in sensor_feat_idx_map.items():
            if feat_idx is None or len(feat_idx) == 0:
                continue
            if sensor_name not in self.S_keys:
                raise KeyError(f"Missing sensor key for '{sensor_name}'")

            F_s = np.zeros(self.D, dtype=np.int32)

            for k in range(K):
                s = k * seg_len
                e = (k + 1) * seg_len if k < K - 1 else T
                seg = W[s:e]
                seg_id = self.ID_seg[k] if self.use_time else None

                if "mean" in self.segment_stats:
                    m = seg.mean(axis=0)
                    F_s += self._encode_stat_vector_subset(m, mu, sigma_safe, feat_idx=feat_idx, seg_id=seg_id)

                if "std" in self.segment_stats:
                    sd = seg.std(axis=0)
                    F_s += self._encode_stat_vector_subset(sd, mu, sigma_safe, feat_idx=feat_idx, seg_id=seg_id)

                if self.use_delta and seg.shape[0] >= 2:
                    d = np.abs(np.diff(seg, axis=0))
                    dm = d.mean(axis=0)
                    F_s += self._encode_stat_vector_subset(dm, mu, sigma_safe, feat_idx=feat_idx, seg_id=seg_id)

            # Bind sensor key at the end: S_s ⊙ F_s
            acc_total += (F_s * self.S_keys[sensor_name].astype(np.int32))

        return acc_total


# -----------------------------
# Prototype classifier (cosine over int32 HVs)
# -----------------------------
class HDCPrototypeClassifier:
    def __init__(self, D=4096):
        self.D = int(D)
        self.prototypes = {}

    def fit(self, H_acc: np.ndarray, y: np.ndarray):
        self.prototypes = {}
        for hv, label in zip(H_acc, y):
            label = int(label)
            if label not in self.prototypes:
                self.prototypes[label] = np.zeros(self.D, dtype=np.int32)
            self.prototypes[label] += hv.astype(np.int32, copy=False)

    def predict(self, H_acc: np.ndarray) -> np.ndarray:
        labels = sorted(self.prototypes.keys())
        P = np.stack([self.prototypes[l] for l in labels]).astype(np.float32)
        Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)

        Hf = H_acc.astype(np.float32)
        Hn = Hf / (np.linalg.norm(Hf, axis=1, keepdims=True) + 1e-8)

        scores = Hn @ Pn.T
        pred_idx = np.argmax(scores, axis=1)
        return np.array([labels[i] for i in pred_idx], dtype=np.int32)


# -----------------------------
# Approximate unlearning helpers (sensor-key unbinding)
# -----------------------------
def bipolar_sign_int32(v: np.ndarray) -> np.ndarray:
    """Return int32 bipolar sign (+1/-1). Zeros -> +1."""
    out = np.ones_like(v, dtype=np.int32)
    out[v < 0] = -1
    return out

def approx_remove_sensor_from_H(H: np.ndarray, S_key_int32: np.ndarray) -> np.ndarray:
    """
    Approx remove a sensor component from an HV superposition using sensor-key unbinding.

    We estimate:
        F_hat = S ⊙ H           (approx per-sensor bundle)
        F_hat_bin = sign(F_hat) (denoise / stabilize)
        H_sensor_hat = S ⊙ F_hat_bin
        H_clean ≈ H - H_sensor_hat

    This is intentionally approximate.
    """
    F_hat = H * S_key_int32
    F_hat_bin = bipolar_sign_int32(F_hat)
    H_sensor_hat = F_hat_bin * S_key_int32
    return H - H_sensor_hat


# -----------------------------
# Helpers
# -----------------------------
def stack_subjects(data_by_subject, subjects):
    Xs, ys = [], []
    for s in subjects:
        Xs.append(data_by_subject[s]["X"])
        ys.append(data_by_subject[s]["y"])
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)

def compute_norm_stats(X_flat: np.ndarray):
    X_flat = X_flat.astype(np.float32, copy=False)
    X_flat = np.where(np.isfinite(X_flat), X_flat, np.nan).astype(np.float32)

    mu = np.nanmean(X_flat, axis=0).astype(np.float32)
    sigma = np.nanstd(X_flat, axis=0).astype(np.float32)

    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    sigma = np.nan_to_num(sigma, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32)
    sigma = np.where(sigma == 0, 1.0, sigma).astype(np.float32)
    return mu, sigma

def encode_dataset_acc(enc: HDCEncoderPooledSensorKeys, X, mu, sigma, sensor_feat_idx_map, name=""):
    N = X.shape[0]
    H = np.empty((N, enc.D), dtype=np.int32)
    for i in range(N):
        H[i] = enc.encode_window_acc(X[i], mu=mu, sigma=sigma, sensor_feat_idx_map=sensor_feat_idx_map)
        if (i + 1) % 600 == 0:
            print(f"  encoded {name}: {i+1}/{N}")
    return H


# -----------------------------
# Main Experiment: LOSO noisy vs approx removed (key-unbinding)
# -----------------------------
def loso_noisy_vs_removed_pooled_key_unbind(
    data_by_subject_base, feat_cols_base,
    # HDC pooled params
    D=4096, Q=21, seed=0, use_time=True, win_size=512, clamp_z=1.0,
    n_segments=4, segment_stats=("mean","std"), use_delta=True,
    # noise params
    source_prefix="hand_",
    n_noise_feats=27,
    noise_seed=123,
    noise_mode="drift+clip+dropout+spike",
    # results
    results_csv_path="exp_noisy_vs_removed_pooled512_keyunbind.csv",
):
    subjects = sorted(data_by_subject_base.keys(), key=int)
    rows = []
    acc_noisy_all = []
    acc_removed_all = []

    # Build noisy dataset once
    data_by_subject_noisy, feat_cols_noisy = add_corrupted_sensor_from_real(
        data_by_subject_base, feat_cols_base,
        source_prefix=source_prefix,
        n_noise_feats=n_noise_feats,
        mode=noise_mode,
        seed=noise_seed
    )

    # Build sensor index groups (this is just for encoding; removal will be key-based)
    hand_idx  = np.array([i for i, c in enumerate(feat_cols_noisy) if c.startswith("hand_")], dtype=np.int32)
    chest_idx = np.array([i for i, c in enumerate(feat_cols_noisy) if c.startswith("chest_")], dtype=np.int32)
    ankle_idx = np.array([i for i, c in enumerate(feat_cols_noisy) if c.startswith("ankle_")], dtype=np.int32)
    noise_idx = np.array([i for i, c in enumerate(feat_cols_noisy) if c.startswith("noise_")], dtype=np.int32)
    if noise_idx.size == 0:
        raise RuntimeError("No noise_* features found after augmentation.")

    sensor_feat_idx_map = {
        "hand": hand_idx,
        "chest": chest_idx,
        "ankle": ankle_idx,
        "noise": noise_idx,
    }

    # Encoder using pooled scheme but with sensor keys
    enc = HDCEncoderPooledSensorKeys(
        D=D, Q=Q, seed=seed, use_time=use_time, win_size=win_size, clamp_z=clamp_z,
        n_segments=n_segments, segment_stats=segment_stats, use_delta=use_delta
    )
    enc.build_feature_ids(feat_cols_noisy)
    enc.build_sensor_keys(sensor_names=list(sensor_feat_idx_map.keys()))

    S_noise_int32 = enc.S_keys["noise"].astype(np.int32)

    for test_subj in subjects:
        train_subjects = [s for s in subjects if s != test_subj]

        X_train, y_train = stack_subjects(data_by_subject_noisy, train_subjects)
        X_test,  y_test  = data_by_subject_noisy[test_subj]["X"], data_by_subject_noisy[test_subj]["y"]

        mu, sigma = compute_norm_stats(X_train.reshape(-1, X_train.shape[-1]))

        Htr = encode_dataset_acc(enc, X_train, mu, sigma, sensor_feat_idx_map, name=f"train(subj{test_subj})")
        Hte = encode_dataset_acc(enc, X_test,  mu, sigma, sensor_feat_idx_map, name=f"test(subj{test_subj})")

        # (1) Noisy model
        clf_noisy = HDCPrototypeClassifier(D=D)
        clf_noisy.fit(Htr, y_train)
        y_pred_noisy = clf_noisy.predict(Hte)
        acc_noisy = float((y_pred_noisy == y_test).mean())

        # (2) Approx remove noise sensor using KEY-UNBIND method (no explicit noise parts stored)
        # Clean test encodings
        Hte_clean = np.empty_like(Hte)
        for i in range(Hte.shape[0]):
            Hte_clean[i] = approx_remove_sensor_from_H(Hte[i], S_noise_int32)

        # Clean prototypes (approx) by applying the same operator to each class prototype
        proto_clean = {}
        for lab, P in clf_noisy.prototypes.items():
            proto_clean[lab] = approx_remove_sensor_from_H(P, S_noise_int32)

        clf_clean = HDCPrototypeClassifier(D=D)
        clf_clean.prototypes = proto_clean

        y_pred_clean = clf_clean.predict(Hte_clean)
        acc_removed = float((y_pred_clean == y_test).mean())

        acc_noisy_all.append(acc_noisy)
        acc_removed_all.append(acc_removed)

        print(f"[SUBJ {test_subj}] noisy={acc_noisy*100:6.2f}%  approx_removed={acc_removed*100:6.2f}%")

        rows.append({
            "test_subject": int(test_subj),
            "acc_noisy": acc_noisy,
            "acc_removed": acc_removed,
            "test_windows": int(len(y_test)),
            "train_windows": int(len(y_train)),
            "source_prefix": str(source_prefix),
            "n_noise_feats": int(n_noise_feats),
            "noise_mode": str(noise_mode),
            "noise_seed": int(noise_seed),
            "win_size": int(win_size),
            "n_segments": int(n_segments),
            "segment_stats": "-".join(segment_stats),
            "use_delta": int(1 if use_delta else 0),
            "unlearning_method": "approx_key_unbind_sign",
        })

    acc_noisy_all = np.array(acc_noisy_all, dtype=np.float32)
    acc_removed_all = np.array(acc_removed_all, dtype=np.float32)

    print("\n==============================")
    print("[SUMMARY] LOSO means (± std)")
    print("==============================")
    print(f"With corrupted sensor : {acc_noisy_all.mean()*100:.2f}% ± {acc_noisy_all.std()*100:.2f}%")
    print(f"After approx removal  : {acc_removed_all.mean()*100:.2f}% ± {acc_removed_all.std()*100:.2f}%")

    df = pd.DataFrame(rows).sort_values("test_subject")
    df.to_csv(results_csv_path, index=False)
    print(f"\n[EXP] Saved results to: {results_csv_path}")
    return df


if __name__ == "__main__":
    # ---- dataset params (pooled + 512) ----
    protocol_dir = r"Protocol"
    win_size = 512
    stride = 256
    majority = 0.9
    include_temp = False
    wrist_only = False
    use_cache = True
    cache_dir = None

    # ---- pooled HDC params (match your newer scripts) ----
    D = 4096
    Q = 21
    seed = 0
    use_time = True
    clamp_z = 1.0
    n_segments = 4
    segment_stats = ("mean", "std")
    use_delta = True

    # ---- noise params ----
    source_prefix = "hand_"
    n_noise_feats = 27
    noise_seed = 123
    noise_mode = "drift+clip+dropout+spike"

    results_csv_noise = "exp_noisy_vs_removed_pooled512_keyunbind.csv"

    data_by_subject, feat_cols = load_pamap2_by_subject(
        protocol_dir=protocol_dir,
        win_size=win_size,
        stride=stride,
        majority=majority,
        include_temp=include_temp,
        wrist_only=wrist_only,
        use_cache=use_cache,
        cache_dir=cache_dir
    )

    total_windows = sum(v["X"].shape[0] for v in data_by_subject.values())
    print("\nTotal subjects:", len(data_by_subject))
    print("Total windows:", total_windows)
    print("Pruned feature count:", len(feat_cols))

    loso_noisy_vs_removed_pooled_key_unbind(
        data_by_subject_base=data_by_subject,
        feat_cols_base=feat_cols,
        D=D, Q=Q, seed=seed,
        use_time=use_time, win_size=win_size, clamp_z=clamp_z,
        n_segments=n_segments, segment_stats=segment_stats, use_delta=use_delta,
        source_prefix=source_prefix,
        n_noise_feats=n_noise_feats,
        noise_seed=noise_seed,
        noise_mode=noise_mode,
        results_csv_path=results_csv_noise,
    )