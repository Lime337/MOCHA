
import os
import glob
import numpy as np
import pandas as pd
from collections import Counter


COLS = ["timestamp", "activity_id", "heart_rate"] + \
       [f"hand_{i}" for i in range(17)] + \
       [f"chest_{i}" for i in range(17)] + \
       [f"ankle_{i}" for i in range(17)]

KEEP_ACTS = {1, 2, 3, 4, 5, 6, 12, 13}


def subject_id_from_path(path: str) -> str:
    base = os.path.basename(path)
    digits = "".join([c for c in base if c.isdigit()])
    return digits

def pruned_cols(include_temp: bool = False, wrist_only: bool = False):
    keep = []
    if include_temp:
        keep.append(0)
    keep += [1, 2, 3, 7, 8, 9, 10, 11, 12]

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
        print(f'Loading windowed PAMAP2 from: {cache_file}')
        data_by_subject, feat_cols = load_cached_dataset_npz(cache_file)
        print(f"Loaded subjects={len(data_by_subject)} total_windows={sum((v['X'].shape[0] for v in data_by_subject.values()))}")
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
            print(f'subject{subj}: skipped (no valid windows)')
            continue

        data_by_subject[subj] = {"X": X, "y": y}
        print(f'subject{subj}: rows={len(df):>7} windows={X.shape[0]:>5} feats={X.shape[2]}')

    total_counts = Counter()
    for subj, d in data_by_subject.items():
        total_counts.update(d["y"])
    print('Global label counts:', dict(total_counts))

    if use_cache:
        print(f'Saving windowed PAMAP2 to: {cache_file}')
        save_cached_dataset_npz(cache_file, data_by_subject, feat_cols)

    return data_by_subject, feat_cols


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
        X = d["X"].astype(np.float32, copy=False)
        y = d["y"]
        N, T, F = X.shape

        noise = X[:, :, choice].copy()

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


class HDCEncoderPooled:
    def __init__(self, D=4096, Q=21, seed=0, use_time=True, win_size=512, clamp_z=1.0,
                 n_segments=4, segment_stats=("mean","std"), use_delta=True):
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

    def quantize(self, Z):
        Z = np.nan_to_num(Z, nan=0.0, posinf=self.clamp_z, neginf=-self.clamp_z)
        Z = np.clip(Z, -self.clamp_z, self.clamp_z)
        u = (Z + self.clamp_z) / (2 * self.clamp_z)
        q = np.floor(u * (self.Q - 1 + 1e-8)).astype(np.int32)
        return np.clip(q, 0, self.Q - 1)

    def _encode_stat_vector(self, stat_vec, mu, sigma_safe, seg_id=None, feat_idx=None):

        if feat_idx is None:
            Z = (stat_vec - mu) / sigma_safe
            q = self.quantize(Z)
            v = self.VAL[q]
            hv = (self.ID_feat * v).sum(axis=0)
        else:
            fi = feat_idx
            Z = (stat_vec[fi] - mu[fi]) / sigma_safe[fi]
            q = self.quantize(Z)
            v = self.VAL[q]
            hv = (self.ID_feat[fi] * v).sum(axis=0)
        if seg_id is not None:
            hv = hv * seg_id
        return hv.astype(np.int32, copy=False)

    def encode_window_acc_parts(self, W, mu, sigma, noise_feat_idx=None):
        
        if self.ID_feat is None:
            raise RuntimeError("Encoder has no ID_feat. Call build_feature_ids().")

        W = W.astype(np.float32, copy=False)
        W = np.where(np.isfinite(W), W, mu[None, :]).astype(np.float32)
        sigma_safe = np.where(sigma == 0, 1.0, sigma).astype(np.float32)

        T, F = W.shape
        K = self.n_segments
        seg_len = T // K
        if seg_len <= 0:
            raise ValueError(f"win_size={T} too small for n_segments={K}")

        acc_total = np.zeros(self.D, dtype=np.int32)
        acc_noise = np.zeros(self.D, dtype=np.int32)

        use_noise = (noise_feat_idx is not None) and (len(noise_feat_idx) > 0)

        for k in range(K):
            s = k * seg_len
            e = (k + 1) * seg_len if k < K - 1 else T
            seg = W[s:e]
            seg_id = self.ID_seg[k] if self.use_time else None

            if "mean" in self.segment_stats:
                m = seg.mean(axis=0)
                acc_total += self._encode_stat_vector(m, mu, sigma_safe, seg_id=seg_id, feat_idx=None)
                if use_noise:
                    acc_noise += self._encode_stat_vector(m, mu, sigma_safe, seg_id=seg_id, feat_idx=noise_feat_idx)

            if "std" in self.segment_stats:
                sd = seg.std(axis=0)
                acc_total += self._encode_stat_vector(sd, mu, sigma_safe, seg_id=seg_id, feat_idx=None)
                if use_noise:
                    acc_noise += self._encode_stat_vector(sd, mu, sigma_safe, seg_id=seg_id, feat_idx=noise_feat_idx)

            if self.use_delta and seg.shape[0] >= 2:
                d = np.abs(np.diff(seg, axis=0))
                dm = d.mean(axis=0)
                acc_total += self._encode_stat_vector(dm, mu, sigma_safe, seg_id=seg_id, feat_idx=None)
                if use_noise:
                    acc_noise += self._encode_stat_vector(dm, mu, sigma_safe, seg_id=seg_id, feat_idx=noise_feat_idx)

        return acc_total, acc_noise


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

def encode_dataset_acc_parts(encoder: HDCEncoderPooled, X, mu, sigma, noise_feat_idx, name=""):
    N = X.shape[0]
    H_total = np.empty((N, encoder.D), dtype=np.int32)
    H_noise = np.empty((N, encoder.D), dtype=np.int32)
    for i in range(N):
        ht, hn = encoder.encode_window_acc_parts(X[i], mu=mu, sigma=sigma, noise_feat_idx=noise_feat_idx)
        H_total[i] = ht
        H_noise[i] = hn
        if (i + 1) % 600 == 0:
            print(f'  encoded {name}: {i + 1}/{N}')
    return H_total, H_noise


def loso_noisy_vs_removed_pooled(
    data_by_subject_base, feat_cols_base,

    D=4096, Q=21, seed=0, use_time=True, win_size=512, clamp_z=1.0,
    n_segments=4, segment_stats=("mean","std"), use_delta=True,

    source_prefix="hand_",
    n_noise_feats=27,
    noise_seed=123,
    noise_mode="drift+clip+dropout+spike",

    results_csv_path="exp_noisy_vs_removed_pooled512.csv",
):
    subjects = sorted(data_by_subject_base.keys(), key=int)
    rows = []
    acc_noisy_all = []
    acc_removed_all = []


    data_by_subject_noisy, feat_cols_noisy = add_corrupted_sensor_from_real(
        data_by_subject_base, feat_cols_base,
        source_prefix=source_prefix,
        n_noise_feats=n_noise_feats,
        mode=noise_mode,
        seed=noise_seed
    )


    noise_feat_idx = np.array([i for i, c in enumerate(feat_cols_noisy) if c.startswith("noise_")], dtype=np.int32)
    if noise_feat_idx.size == 0:
        raise RuntimeError("No noise_* features found after augmentation.")


    enc = HDCEncoderPooled(
        D=D, Q=Q, seed=seed, use_time=use_time, win_size=win_size, clamp_z=clamp_z,
        n_segments=n_segments, segment_stats=segment_stats, use_delta=use_delta
    )
    enc.build_feature_ids(feat_cols_noisy)

    for test_subj in subjects:
        train_subjects = [s for s in subjects if s != test_subj]

        X_train, y_train = stack_subjects(data_by_subject_noisy, train_subjects)
        X_test,  y_test  = data_by_subject_noisy[test_subj]["X"], data_by_subject_noisy[test_subj]["y"]

        mu, sigma = compute_norm_stats(X_train.reshape(-1, X_train.shape[-1]))

        Htr_total, Htr_noise = encode_dataset_acc_parts(enc, X_train, mu, sigma, noise_feat_idx, name=f"train(subj{test_subj})")
        Hte_total, Hte_noise = encode_dataset_acc_parts(enc, X_test,  mu, sigma, noise_feat_idx, name=f"test(subj{test_subj})")


        clf_noisy = HDCPrototypeClassifier(D=D)
        clf_noisy.fit(Htr_total, y_train)
        y_pred_noisy = clf_noisy.predict(Hte_total)
        acc_noisy = float((y_pred_noisy == y_test).mean())


        labels = sorted(clf_noisy.prototypes.keys())

        P_noise = {lab: np.zeros(D, dtype=np.int32) for lab in labels}
        for hv_n, lab in zip(Htr_noise, y_train):
            P_noise[int(lab)] += hv_n

        proto_clean = {lab: (clf_noisy.prototypes[lab] - P_noise[lab]) for lab in labels}
        Hte_clean = Hte_total - Hte_noise

        clf_clean = HDCPrototypeClassifier(D=D)
        clf_clean.prototypes = proto_clean

        y_pred_clean = clf_clean.predict(Hte_clean)
        acc_removed = float((y_pred_clean == y_test).mean())

        acc_noisy_all.append(acc_noisy)
        acc_removed_all.append(acc_removed)

        print(f'[SUBJ {test_subj}] noisy={acc_noisy * 100:6.2f}%  removed={acc_removed * 100:6.2f}%')

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
            "stride": int(0),
            "n_segments": int(n_segments),
            "segment_stats": "-".join(segment_stats),
            "use_delta": int(1 if use_delta else 0),
        })

    acc_noisy_all = np.array(acc_noisy_all, dtype=np.float32)
    acc_removed_all = np.array(acc_removed_all, dtype=np.float32)

    pass
    print('LOSO mean accuracy (± standard deviation):')
    pass
    print(f'With corrupted sensor : {acc_noisy_all.mean() * 100:.2f}% ± {acc_noisy_all.std() * 100:.2f}%')
    print(f'After algebra removal : {acc_removed_all.mean() * 100:.2f}% ± {acc_removed_all.std() * 100:.2f}%')

    df = pd.DataFrame(rows).sort_values("test_subject")
    df.to_csv(results_csv_path, index=False)
    print(f'Saved results to: {results_csv_path}')
    return df


if __name__ == "__main__":

    protocol_dir = r"Protocol"
    win_size = 512
    stride = 256
    majority = 0.9
    include_temp = False
    wrist_only = False
    use_cache = True
    cache_dir = None


    D = 4096
    Q = 21
    seed = 0
    use_time = True
    clamp_z = 1.0
    n_segments = 4
    segment_stats = ("mean", "std")
    use_delta = True


    source_prefix = "hand_"
    n_noise_feats = 27
    noise_seed = 123
    noise_mode = "drift+clip+dropout+spike"

    results_csv_noise = "exp_noisy_vs_removed_pooled512.csv"

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
    print('Total subjects:', len(data_by_subject))
    print('Total windows:', total_windows)
    print('Pruned feature count:', len(feat_cols))

    loso_noisy_vs_removed_pooled(
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
