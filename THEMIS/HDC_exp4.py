
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# EXP4: "Good personalization" -> inject bad samples -> unlearn
#   - Starts from LOSO global model for each subject
#   - Personalizes online using Exp2 margin rule
#   - After a chosen point, injects BAD samples (label noise and/or feature noise)
#   - Accuracy drops on the (clean) held-out test set
#   - Then "unlearn" ONLY the updates caused by those bad samples to recover
#
# This is meant to be a variation combining Exp2 + Exp3 ideas.
# ============================================================

# ----------------------------
# CONFIG (match your pooled pipeline)
# ----------------------------
DATASET_CACHE_NPZ = r"pamap2_windows_w512_s256_m0p900_notemp_allsensors.npz"
LOSO_MODELS_NPZ   = r"pamap2_loso_models_w512_s256_m0p900_notemp_allsensors_D4096_Q21_seed0_time_cz1p0.npz"
ENCODER_DICT_NPZ  = r"pamap2_encoder_dict_w512_s256_m0p900_notemp_allsensors_D4096_Q21_seed0_time_cz1p0_seg4_statsmean-std_dlt.npz"

OUT_DIR = "exp4_poison_unlearn_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Personalization protocol (same style as Exp2)
PERS_FRACTION = 0.70
ORDER = "shuffle"         # "chronological" or "shuffle"
SPLIT_SEED = 42
EVAL_EVERY = 50

# Margin update knobs (same defaults as your pooled Exp2)
ALPHA_WRONG   = 10
ALPHA_CORRECT = 2
MARGIN_TAU    = 0.02

# "Bad sample" injection knobs
# Choose either/both:
#  (A) label flips: with probability P_LABEL_FLIP, replace y_true with random wrong label
#  (B) feature noise: add N(0, sigma) to the window before encoding (only during bad phase)
BAD_START_MODE = "checkpoint"  # "checkpoint" or "fixed_seen"
BAD_START_SEEN_FIXED = 200     # used if BAD_START_MODE="fixed_seen"
BAD_LEN = 250                  # number of personalization windows to corrupt after bad start
P_LABEL_FLIP = 0.50            # probability to flip label during bad phase (0..1)
FEATURE_NOISE_STD = 0.0        # e.g., 0.25 or 0.50; 0 disables feature noise

# Unlearning: how to undo bad-phase updates
UNLEARN_STRATEGY = "bad_only"  # "bad_only" or "last_k_updates"
LAST_K_UPDATES = 200           # used if UNLEARN_STRATEGY="last_k_updates"
EVAL_EVERY_UNDO = 25

# ----------------------------
# Helper: meta normalization
# ----------------------------
def _meta_to_dict(meta):
    if isinstance(meta, dict):
        return meta
    if isinstance(meta, np.ndarray):
        try:
            meta = meta.tolist()
        except Exception:
            pass
    if isinstance(meta, list) and len(meta) == 1 and isinstance(meta[0], dict):
        return meta[0]
    if isinstance(meta, list) and all(isinstance(x, (tuple, list)) and len(x) == 2 for x in meta):
        return dict(meta)
    if hasattr(meta, "item"):
        try:
            v = meta.item()
            if isinstance(v, dict):
                return v
        except Exception:
            pass
    raise TypeError(f"Could not normalize meta to dict. Got type={type(meta)} value={meta}")

# ----------------------------
# Dataset cache IO
# ----------------------------
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

# ----------------------------
# Encoder dict IO (pooled)
# ----------------------------
def load_encoder_dict_npz(path):
    z = np.load(path, allow_pickle=True)
    feat_cols = list(z["feat_cols"].tolist())
    meta = _meta_to_dict(z["meta"].tolist())
    ID_feat = z["ID_feat"].astype(np.int8, copy=False)   # (F,D)
    VAL     = z["VAL"].astype(np.int8, copy=False)       # (Q,D)
    ID_seg  = z["ID_seg"].astype(np.int8, copy=False) if "ID_seg" in z.files else None  # (K,D)
    return feat_cols, meta, ID_feat, VAL, ID_seg

# ----------------------------
# LOSO models IO
# ----------------------------
def load_loso_models_npz(path):
    z = np.load(path, allow_pickle=True)
    feat_cols = list(z["feat_cols"].tolist())
    subjects = [str(s) for s in z["subjects"].tolist()]
    meta = _meta_to_dict(z["meta"].tolist())

    models = {}
    for subj in subjects:
        mu = z[f"mu_{subj}"].astype(np.float32, copy=False)
        sigma = z[f"sigma_{subj}"].astype(np.float32, copy=False)
        labels = z[f"labels_{subj}"].astype(np.int32, copy=False).tolist()
        P = {int(lab): z[f"P_{subj}_{int(lab)}"].astype(np.int32, copy=False) for lab in labels}
        models[subj] = {"mu": mu, "sigma": sigma, "labels": labels, "P": P}
    return feat_cols, models, meta

# ----------------------------
# Pooled HDC Encoder (same as Exp2/Exp3)
# ----------------------------
class HDCEncoderPooled:
    def __init__(self, D=4096, Q=21, seed=0, use_time=True, win_size=256, clamp_z=1.0,
                 n_segments=4, segment_stats=("mean",), use_delta=False):
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

    def set_encoding_dict(self, feat_cols, ID_feat, VAL, ID_seg=None):
        self.feat_cols = list(feat_cols)
        self.ID_feat = ID_feat.astype(np.int8, copy=False)
        self.VAL = VAL.astype(np.int8, copy=False)
        self.D = int(self.ID_feat.shape[1])
        self.Q = int(self.VAL.shape[0])
        self.ID_seg = ID_seg.astype(np.int8, copy=False) if (self.use_time and ID_seg is not None) else None

        if self.use_time and self.ID_seg is None:
            raise ValueError("use_time=True but encoder dict has no ID_seg.")
        if self.use_time and self.ID_seg.shape[0] != self.n_segments:
            raise ValueError(f"ID_seg has K={self.ID_seg.shape[0]} but n_segments={self.n_segments}")

    def quantize(self, Z):
        Z = np.nan_to_num(Z, nan=0.0, posinf=self.clamp_z, neginf=-self.clamp_z)
        Z = np.clip(Z, -self.clamp_z, self.clamp_z)
        u = (Z + self.clamp_z) / (2 * self.clamp_z)
        q = np.floor(u * (self.Q - 1 + 1e-8)).astype(np.int32)
        return np.clip(q, 0, self.Q - 1)

    def _encode_stat_vector(self, stat_vec, mu, sigma_safe, seg_id=None):
        Z = (stat_vec - mu) / sigma_safe
        q = self.quantize(Z)
        v = self.VAL[q]                     # (F,D)
        hv = (self.ID_feat * v).sum(axis=0) # (D,)
        if seg_id is not None:
            hv = hv * seg_id
        return hv.astype(np.int32, copy=False)

    def encode_window(self, W, mu, sigma):
        if self.ID_feat is None or self.VAL is None:
            raise RuntimeError("Encoder dictionary not initialized.")

        W = W.astype(np.float32, copy=False)
        W = np.where(np.isfinite(W), W, mu[None, :]).astype(np.float32)

        sigma_safe = np.where(sigma == 0, 1.0, sigma).astype(np.float32)

        T, F = W.shape
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

# ----------------------------
# Prototype prediction / eval
# ----------------------------
def _stack_P(labels, P_dict):
    return np.stack([P_dict[int(l)] for l in labels], axis=0)

def scores_for(hv, labels, P_dict):
    Pm = _stack_P(labels, P_dict).astype(np.float32)     # (C,D)
    hvf = hv.astype(np.float32)
    dots = Pm @ hvf
    norms = np.linalg.norm(Pm, axis=1) + 1e-8
    return dots / norms

def eval_on_set(H, y, labels, P_dict):
    Pm = _stack_P(labels, P_dict).astype(np.float32)
    norms = np.linalg.norm(Pm, axis=1) + 1e-8
    scores = (H.astype(np.float32) @ Pm.T) / norms[None, :]
    pred_idx = np.argmax(scores, axis=1)
    y_pred = np.array([labels[i] for i in pred_idx], dtype=np.int32)
    return float((y_pred == y).mean())

# ----------------------------
# Online update + logging
# ----------------------------
UPD_WRONG = 0
UPD_LOW_MARGIN_CORRECT = 1

def apply_update(P, labels, hv_i32, y_true, y_hat, margin):
    """
    Applies the Exp2 margin rule update.
    Returns (did_update, update_type, alpha_used).
    """
    if y_hat != y_true:
        P[y_true] += ALPHA_WRONG * hv_i32
        P[y_hat]  -= ALPHA_WRONG * hv_i32
        return True, UPD_WRONG, ALPHA_WRONG
    else:
        if margin < MARGIN_TAU:
            P[y_true] += ALPHA_CORRECT * hv_i32
            return True, UPD_LOW_MARGIN_CORRECT, ALPHA_CORRECT
    return False, -1, 0

def undo_update(P, hv_i32, y_true, y_hat, upd_type, alpha):
    """
    Inverts the update applied by apply_update().
    """
    if upd_type == UPD_WRONG:
        P[y_true] -= alpha * hv_i32
        P[y_hat]  += alpha * hv_i32
    elif upd_type == UPD_LOW_MARGIN_CORRECT:
        P[y_true] -= alpha * hv_i32
    else:
        raise RuntimeError(f"Unknown upd_type={upd_type}")

def maybe_corrupt_label(rng, y_true, labels):
    if P_LABEL_FLIP <= 0:
        return y_true, False
    if rng.random() >= P_LABEL_FLIP:
        return y_true, False
    # pick a wrong label uniformly
    choices = [int(l) for l in labels if int(l) != int(y_true)]
    if not choices:
        return y_true, False
    return int(rng.choice(choices)), True

def maybe_corrupt_window(rng, W):
    if FEATURE_NOISE_STD <= 0:
        return W, False
    noise = rng.normal(0.0, FEATURE_NOISE_STD, size=W.shape).astype(np.float32)
    return (W + noise).astype(np.float32), True

# ----------------------------
# Core routine for one subject
# ----------------------------
def run_exp4_subject(subj, data_by_subject, encoder, loso_models):
    X = data_by_subject[subj]["X"]
    y = data_by_subject[subj]["y"]

    fold = loso_models[subj]
    mu, sigma = fold["mu"], fold["sigma"]
    labels = fold["labels"]
    P = {lab: fold["P"][lab].copy() for lab in labels}  # start from global/LOSO model

    # deterministic subject rng (for corruption + shuffle)
    rng = np.random.default_rng(1000 + int(subj))

    # subject split (same as Exp2 style)
    idx_perm = np.arange(len(y), dtype=np.int32)
    if ORDER == "shuffle":
        rng_shuf = np.random.default_rng(SPLIT_SEED + int(subj))
        rng_shuf.shuffle(idx_perm)

    n_pers = int(round(PERS_FRACTION * len(idx_perm)))
    pers_idx = idx_perm[:n_pers].copy()
    test_idx = idx_perm[n_pers:].copy()

    X_pers, y_pers = X[pers_idx], y[pers_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Encode clean test set once
    H_test = np.empty((len(y_test), encoder.D), dtype=np.int8)
    for i in range(len(y_test)):
        H_test[i] = encoder.encode_window(X_test[i], mu, sigma)

    baseline_acc = eval_on_set(H_test, y_test, labels, P)

    # Track best checkpoint during clean personalization (pre-bad)
    best_acc = baseline_acc
    best_seen = 0
    best_updates = 0
    best_P = {lab: P[lab].copy() for lab in labels}

    # Decide when to start "bad" phase
    bad_start_seen = None

    # Update logs (like Exp2/3 but with bad flags)
    upd_seen_t = []      # t index in personalization stream
    upd_ytrue  = []
    upd_yhat   = []
    upd_alpha  = []
    upd_type   = []
    upd_is_bad = []      # 0/1: did this update happen during bad phase?
    upd_lbl_flip = []    # 0/1: was label flipped for that sample?
    upd_feat_noise = []  # 0/1: was feature noise applied?

    # Curve logs
    curve = []
    updates = 0

    # ----------------------------
    # Online personalization stream
    # ----------------------------
    for t in range(len(y_pers)):
        # Determine bad phase start
        if bad_start_seen is None:
            if BAD_START_MODE == "fixed_seen" and (t + 1) == int(BAD_START_SEEN_FIXED):
                bad_start_seen = int(t + 1)
            elif BAD_START_MODE == "checkpoint":
                # start bad phase right after we hit a "good" checkpoint improvement (first time best_acc improves by >=0.5%)
                # If you want "after peak", you can instead trigger at some best_seen.
                pass

        in_bad = (bad_start_seen is not None) and ((t + 1) >= bad_start_seen) and ((t + 1) < (bad_start_seen + BAD_LEN))

        # If BAD_START_MODE="checkpoint": trigger bad right after we observe a clear improvement
        if BAD_START_MODE == "checkpoint" and bad_start_seen is None:
            # We will set bad_start_seen the first time best_acc exceeds baseline by 0.5%
            if (best_acc - baseline_acc) >= 0.005:
                bad_start_seen = int(t + 1)  # begin bad immediately after we got "good personalization"
                in_bad = True

        # Sample (maybe corrupt)
        W = X_pers[t]
        y_true_clean = int(y_pers[t])

        lbl_flip = False
        feat_noise = False

        if in_bad:
            y_true_eff, lbl_flip = maybe_corrupt_label(rng, y_true_clean, labels)
            W_eff, feat_noise = maybe_corrupt_window(rng, W)
        else:
            y_true_eff = y_true_clean
            W_eff = W

        hv = encoder.encode_window(W_eff, mu, sigma)
        sc = scores_for(hv, labels, P)
        bi = int(np.argmax(sc))
        si = int(np.argsort(sc)[-2]) if len(sc) >= 2 else bi
        y_hat = int(labels[bi])
        margin = float(sc[bi] - sc[si]) if len(sc) >= 2 else 0.0

        hv_i32 = hv.astype(np.int32)

        did, typ, a = apply_update(P, labels, hv_i32, y_true_eff, y_hat, margin)
        if did:
            updates += 1
            upd_seen_t.append(int(t))
            upd_ytrue.append(int(y_true_eff))
            upd_yhat.append(int(y_hat))
            upd_alpha.append(int(a))
            upd_type.append(int(typ))
            upd_is_bad.append(int(1 if in_bad else 0))
            upd_lbl_flip.append(int(1 if lbl_flip else 0))
            upd_feat_noise.append(int(1 if feat_noise else 0))

        # Evaluate periodically (always on CLEAN test)
        if (t + 1) % EVAL_EVERY == 0 or (t + 1) == len(y_pers):
            acc = eval_on_set(H_test, y_test, labels, P)

            # Update best checkpoint ONLY if we are still in clean phase
            if not in_bad and (acc > best_acc + 1e-4):
                best_acc = acc
                best_seen = int(t + 1)
                best_updates = int(updates)
                best_P = {lab: P[lab].copy() for lab in labels}

            curve.append({
                "subject": subj,
                "seen": int(t + 1),
                "updates": int(updates),
                "test_acc_clean": float(acc),
                "baseline_acc": float(baseline_acc),
                "best_acc_cleanphase": float(best_acc),
                "best_seen_cleanphase": int(best_seen),
                "in_bad_phase": int(1 if in_bad else 0),
                "bad_start_seen": int(bad_start_seen) if bad_start_seen is not None else -1,
            })

            print(
                f"[SUBJ {subj}] seen={t+1:4d} updates={updates:4d} "
                f"acc(clean_test)={acc*100:6.2f}% "
                f"(baseline={baseline_acc*100:6.2f} best_clean={best_acc*100:6.2f}) "
                f"{'BAD' if in_bad else 'OK'}"
            )

    # After stream, record acc after bad phase took effect
    acc_after_stream = eval_on_set(H_test, y_test, labels, P)

    # ----------------------------
    # UNLEARN / ROLLBACK
    # ----------------------------
    # Start from final P (after bad samples)
    P_bad = {lab: P[lab].copy() for lab in labels}

    upd_seen_t = np.array(upd_seen_t, dtype=np.int32)
    upd_ytrue  = np.array(upd_ytrue, dtype=np.int32)
    upd_yhat   = np.array(upd_yhat, dtype=np.int32)
    upd_alpha  = np.array(upd_alpha, dtype=np.int32)
    upd_type   = np.array(upd_type, dtype=np.int32)
    upd_is_bad = np.array(upd_is_bad, dtype=np.int32)

    if UNLEARN_STRATEGY == "bad_only":
        undo_mask = (upd_is_bad == 1)
    elif UNLEARN_STRATEGY == "last_k_updates":
        k = int(min(LAST_K_UPDATES, len(upd_seen_t)))
        undo_mask = np.zeros(len(upd_seen_t), dtype=bool)
        if k > 0:
            undo_mask[-k:] = True
    else:
        raise ValueError(f"Unknown UNLEARN_STRATEGY={UNLEARN_STRATEGY}")

    undo_idx = np.where(undo_mask)[0].astype(np.int32)
    U = int(len(undo_idx))

    # We'll undo in reverse chronological order of *updates*
    undo_curve = []
    undone = 0

    # Evaluate before undo
    acc0 = eval_on_set(H_test, y_test, labels, P_bad)
    undo_curve.append({
        "subject": subj,
        "undone_updates": 0,
        "remaining_to_undo": U,
        "test_acc_clean": float(acc0),
    })

    # Need X_pers indexed by t (seen index), not by update index
    for j in range(U - 1, -1, -1):
        u = int(undo_idx[j])  # index into update log arrays
        t = int(upd_seen_t[u])
        y_true_eff = int(upd_ytrue[u])
        y_hat = int(upd_yhat[u])
        a = int(upd_alpha[u])
        typ = int(upd_type[u])

        # Re-encode the SAME (potentially corrupted) sample that produced the update:
        # To do that exactly, we'd need to replay the corruption randomness.
        # Instead, we store only whether corruption was applied, not the exact noise vector.
        # So for strict invertibility under feature noise, set FEATURE_NOISE_STD=0.
        # Label flips are perfectly invertible since hv is from the window, not the label.
        W = X_pers[t]
        W_eff = W
        if FEATURE_NOISE_STD > 0:
            # WARNING: cannot reconstruct exact noise without storing it. We'll skip exact invert in this case.
            # We still keep the demo running (undo will be approximate).
            pass

        hv = encoder.encode_window(W_eff, mu, sigma).astype(np.int32)
        undo_update(P_bad, hv, y_true_eff, y_hat, typ, a)

        undone += 1
        if undone % int(EVAL_EVERY_UNDO) == 0 or undone == U:
            accu = eval_on_set(H_test, y_test, labels, P_bad)
            undo_curve.append({
                "subject": subj,
                "undone_updates": int(undone),
                "remaining_to_undo": int(U - undone),
                "test_acc_clean": float(accu),
            })
            print(f"[SUBJ {subj}] UNDO {undone}/{U} acc(clean_test)={accu*100:.2f}%")

    acc_after_unlearn = eval_on_set(H_test, y_test, labels, P_bad)

    # Summary for this subject
    summary = {
        "subject": subj,
        "baseline_acc": float(baseline_acc),
        "best_cleanphase_acc": float(best_acc),
        "best_cleanphase_seen": int(best_seen),
        "acc_after_bad_stream": float(acc_after_stream),
        "bad_start_seen": int(bad_start_seen) if bad_start_seen is not None else -1,
        "bad_len": int(BAD_LEN),
        "num_updates_total": int(len(upd_seen_t)),
        "num_updates_to_undo": int(U),
        "acc_after_unlearn": float(acc_after_unlearn),
        "recovered_vs_best": float(acc_after_unlearn - best_acc),
        "drop_due_to_bad": float(acc_after_stream - best_acc),
        "unlearn_gain": float(acc_after_unlearn - acc_after_stream),
        "UNLEARN_STRATEGY": UNLEARN_STRATEGY,
        "P_LABEL_FLIP": float(P_LABEL_FLIP),
        "FEATURE_NOISE_STD": float(FEATURE_NOISE_STD),
    }

    return curve, undo_curve, summary

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    print(f"[DATA] Loading cached windows from: {DATASET_CACHE_NPZ}")
    data_by_subject, feat_cols_data = load_cached_dataset_npz(DATASET_CACHE_NPZ)
    subjects_data = sorted(data_by_subject.keys(), key=int)
    print(f"[DATA] subjects={len(subjects_data)} total_windows={sum(v['X'].shape[0] for v in data_by_subject.values())}")

    print(f"[LOSO] Loading LOSO models from: {LOSO_MODELS_NPZ}")
    feat_cols_model, loso_models, meta = load_loso_models_npz(LOSO_MODELS_NPZ)
    print("[LOSO] meta:", meta)

    print(f"[ENC] Loading pooled encoder dict from: {ENCODER_DICT_NPZ}")
    feat_cols_enc, meta_enc, ID_feat, VAL, ID_seg = load_encoder_dict_npz(ENCODER_DICT_NPZ)
    print("[ENC] meta:", meta_enc)

    if feat_cols_data != feat_cols_model or feat_cols_data != feat_cols_enc:
        raise ValueError("Feature columns mismatch between dataset / LOSO / encoder dict NPZs.")

    # Pull encoder settings from meta
    D = int(meta.get("D", 4096))
    Q = int(meta.get("Q", 21))
    seed = int(meta.get("seed", 0))
    use_time = bool(meta.get("use_time", True))
    win_size = int(meta.get("win_size", 256))
    clamp_z = float(meta.get("clamp_z", 1.0))

    n_segments = int(meta_enc.get("n_segments", 4))
    segment_stats = tuple(meta_enc.get("segment_stats", ("mean",)))
    use_delta = bool(meta_enc.get("use_delta", False))

    if use_time and ID_seg is None:
        raise ValueError("use_time=True but encoder dict has no ID_seg saved.")

    encoder = HDCEncoderPooled(
        D=D, Q=Q, seed=seed, use_time=use_time, win_size=win_size, clamp_z=clamp_z,
        n_segments=n_segments, segment_stats=segment_stats, use_delta=use_delta
    )
    encoder.set_encoding_dict(feat_cols_data, ID_feat=ID_feat, VAL=VAL, ID_seg=ID_seg)

    curve_rows = []
    undo_rows = []
    summary_rows = []

    subjects = sorted(set(subjects_data) & set(loso_models.keys()), key=int)
    print(f"[EXP4] subjects to run: {subjects}")

    for subj in subjects:
        curve, undo_curve, summary = run_exp4_subject(subj, data_by_subject, encoder, loso_models)
        curve_rows.extend(curve)
        undo_rows.extend(undo_curve)
        summary_rows.append(summary)

    curve_df = pd.DataFrame(curve_rows)
    undo_df = pd.DataFrame(undo_rows)
    summary_df = pd.DataFrame(summary_rows)

    curve_path = os.path.join(OUT_DIR, "exp4_personalize_then_poison_curve.csv")
    undo_path = os.path.join(OUT_DIR, "exp4_unlearn_curve.csv")
    summary_path = os.path.join(OUT_DIR, "exp4_summary.csv")

    curve_df.to_csv(curve_path, index=False)
    undo_df.to_csv(undo_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"\n[SAVED] curve:   {curve_path}")
    print(f"[SAVED] undo:    {undo_path}")
    print(f"[SAVED] summary: {summary_path}")

        # ----------------------------
    # NEW: 4-stage summary plot (mean over subjects)
    # ----------------------------
    if len(summary_df) > 0:
        # Consolas font (falls back if not available)
        plt.rcParams["font.family"] = "Consolas"
        plt.rcParams["font.size"] = 12

        # Stage values (mean over subjects)
        y_vals = [
            float(summary_df["baseline_acc"].mean()),
            float(summary_df["best_cleanphase_acc"].mean()),
            float(summary_df["acc_after_bad_stream"].mean()),
            float(summary_df["acc_after_unlearn"].mean()),
        ]
        x_labels = ["Baseline", "Personalized\n(best)", "Corrupted", "Unlearned"]

        fig = plt.figure(figsize=(7.2, 3.2), dpi=200)
        ax = plt.gca()

        ax.plot(range(len(y_vals)), y_vals, marker="o", linewidth=2)

        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(max(0.0, min(y_vals) - 0.03), min(1.0, max(y_vals) + 0.03))
        ax.grid(True, alpha=0.3)

        # Annotate each point with value
        for i, v in enumerate(y_vals):
            ax.text(i, v + 0.004, f"{v:.3f}", ha="center", va="bottom")

        ax.set_title("Personalization → Corruption → Selective Unlearning (mean over subjects)")

        png_path = os.path.join(OUT_DIR, "exp4_4stage_summary.png")
        pdf_path = os.path.join(OUT_DIR, "exp4_4stage_summary.pdf")
        plt.tight_layout()
        plt.savefig(png_path, bbox_inches="tight")
        plt.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)

        print(f"[SAVED] plot: {png_path}")
        print(f"[SAVED] plot: {pdf_path}")

       

    if len(summary_df) > 0:
        print("\nSummary (mean over subjects):")
        print(" baseline_acc:          ", float(summary_df["baseline_acc"].mean()))
        print(" best_cleanphase_acc:   ", float(summary_df["best_cleanphase_acc"].mean()))
        print(" acc_after_bad_stream:  ", float(summary_df["acc_after_bad_stream"].mean()))
        print(" acc_after_unlearn:     ", float(summary_df["acc_after_unlearn"].mean()))
        print(" mean drop_due_to_bad:  ", float(summary_df["drop_due_to_bad"].mean()))
        print(" mean unlearn_gain:     ", float(summary_df["unlearn_gain"].mean()))
