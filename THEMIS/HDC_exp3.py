import os
import numpy as np
import pandas as pd

# ============================================================
# CONFIG (edit these)
# ============================================================
# Should match your NEW pooled tuned window experiment
DATASET_CACHE_NPZ = r"pamap2_windows_w512_s256_m0p900_notemp_wrist.npz"
LOSO_MODELS_NPZ   = r"pamap2_loso_models_w512_s256_m0p900_notemp_wrist_D4096_Q21_seed0_time_cz1p0.npz"
ENCODER_DICT_NPZ  = r"pamap2_encoder_dict_w512_s256_m0p900_notemp_wrist_D4096_Q21_seed0_time_cz1p0_seg4_statsmean-std_dlt.npz"
# DATASET_CACHE_NPZ = r"pamap2_windows_w512_s256_m0p700_notemp_wrist.npz"
# LOSO_MODELS_NPZ   = r"pamap2_loso_sgd_w512_s256_m0p700_notemp_wrist_losslog_loss_a0.0001_seed0.npz"
# ENCODER_DICT_NPZ  = r"pamap2_encoder_dict_w512_s256_m0p900_notemp_wrist_D4096_Q21_seed0_time_cz1p0_seg4_statsmean-std_dlt.npz"
# This is the NPZ produced by your pooled Exp2 script
PERSONALIZED_MODELS_NPZ = r"exp2_outputs_pooled\exp2_personalized_models_pooled_pf70_shuffle_seed42_aw10_ac2_tau0p02.npz"

OUT_DIR = "exp3_unlearn_outputs_pooled"
os.makedirs(OUT_DIR, exist_ok=True)

# Rollback evaluation cadence
EVAL_EVERY_UPDATES = 25      # evaluate after undoing every N updates
ALSO_EVAL_AT_EDGES = True    # always include {0, U} points


# ============================================================
# META NORMALIZATION
# ============================================================
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


# ============================================================
# DATASET / ENCODER / LOSO IO
# ============================================================
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


def load_encoder_dict_npz_pooled(path):
    z = np.load(path, allow_pickle=True)
    feat_cols = list(z["feat_cols"].tolist())
    meta = _meta_to_dict(z["meta"].tolist())
    ID_feat = z["ID_feat"].astype(np.int8, copy=False)   # (F,D)
    VAL     = z["VAL"].astype(np.int8, copy=False)       # (Q,D)
    ID_seg  = z["ID_seg"].astype(np.int8, copy=False) if "ID_seg" in z.files else None  # (K,D)
    return feat_cols, meta, ID_feat, VAL, ID_seg


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


# ============================================================
# POOLED HDC ENCODER (segments + stats + optional delta)
# ============================================================
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
        self.ID_seg = None  # (K,D) when use_time=True

    def set_encoding_dict(self, feat_cols, ID_feat, VAL, ID_seg=None):
        self.feat_cols = list(feat_cols)
        self.ID_feat = ID_feat.astype(np.int8, copy=False)   # (F,D)
        self.VAL = VAL.astype(np.int8, copy=False)           # (Q,D)
        self.D = int(self.ID_feat.shape[1])
        self.Q = int(self.VAL.shape[0])
        self.ID_seg = ID_seg.astype(np.int8, copy=False) if (self.use_time and ID_seg is not None) else None

        if self.use_time and self.ID_seg is None:
            raise ValueError("use_time=True but encoder dict has no ID_seg saved.")
        if self.use_time and self.ID_seg.shape[0] != self.n_segments:
            raise ValueError(
                f"ID_seg has K={self.ID_seg.shape[0]} but encoder expects n_segments={self.n_segments}."
            )

    def quantize(self, Z):
        Z = np.nan_to_num(Z, nan=0.0, posinf=self.clamp_z, neginf=-self.clamp_z)
        Z = np.clip(Z, -self.clamp_z, self.clamp_z)
        u = (Z + self.clamp_z) / (2 * self.clamp_z)
        q = np.floor(u * (self.Q - 1 + 1e-8)).astype(np.int32)
        return np.clip(q, 0, self.Q - 1)

    def _encode_stat_vector(self, stat_vec, mu, sigma_safe, seg_id=None):
        Z = (stat_vec - mu) / sigma_safe         # (F,)
        q = self.quantize(Z)                     # (F,)
        v = self.VAL[q]                          # (F,D)
        hv = (self.ID_feat * v).sum(axis=0)      # (D,)
        if seg_id is not None:
            hv = hv * seg_id
        return hv.astype(np.int32, copy=False)

    def encode_window(self, W, mu, sigma):
        if self.ID_feat is None or self.VAL is None:
            raise RuntimeError("Encoder dictionary not initialized (ID_feat/VAL missing).")

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
            seg = W[s:e]  # (segT, F)

            seg_id = self.ID_seg[k] if self.use_time else None

            if "mean" in self.segment_stats:
                acc += self._encode_stat_vector(seg.mean(axis=0), mu, sigma_safe, seg_id=seg_id)
            if "std" in self.segment_stats:
                acc += self._encode_stat_vector(seg.std(axis=0), mu, sigma_safe, seg_id=seg_id)

            if self.use_delta:
                if seg.shape[0] >= 2:
                    d = np.abs(np.diff(seg, axis=0))
                    acc += self._encode_stat_vector(d.mean(axis=0), mu, sigma_safe, seg_id=seg_id)

        return np.where(acc >= 0, 1, -1).astype(np.int8)


# ============================================================
# PROTOTYPE PRED / EVAL
# ============================================================
def _stack_P(labels, P_dict):
    return np.stack([P_dict[int(l)] for l in labels], axis=0)

def eval_on_set(H, y, labels, P_dict):
    Pm = _stack_P(labels, P_dict).astype(np.float32)   # (C,D)
    norms = np.linalg.norm(Pm, axis=1) + 1e-8
    scores = (H.astype(np.float32) @ Pm.T) / norms[None, :]
    pred_idx = np.argmax(scores, axis=1)
    y_pred = np.array([labels[i] for i in pred_idx], dtype=np.int32)
    return float((y_pred == y).mean())


# ============================================================
# LOAD personalized models/logs saved by pooled Exp2
# ============================================================
def load_personalized_models_npz(path):
    z = np.load(path, allow_pickle=True)

    feat_cols = list(z["feat_cols"].tolist())
    subjects = [str(s) for s in z["subjects"].tolist()]
    exp_meta = _meta_to_dict(z["exp_meta"].tolist())

    models = {}
    for subj in subjects:
        labels = z[f"labels_{subj}"].astype(np.int32, copy=False).tolist()
        P_best = {int(lab): z[f"Pbest_{subj}_{int(lab)}"].astype(np.int32, copy=False) for lab in labels}

        best_acc = float(z[f"best_acc_{subj}"][0])
        best_seen = int(z[f"best_seen_{subj}"][0])
        best_updates = int(z[f"best_updates_{subj}"][0])

        idx_perm = z[f"idxperm_{subj}"].astype(np.int32, copy=False)
        pers_idx = z[f"pers_idx_{subj}"].astype(np.int32, copy=False)
        test_idx = z[f"test_idx_{subj}"].astype(np.int32, copy=False)

        upd_t     = z[f"upd_t_{subj}"].astype(np.int32, copy=False)
        upd_ytrue = z[f"upd_ytrue_{subj}"].astype(np.int32, copy=False)
        upd_yhat  = z[f"upd_yhat_{subj}"].astype(np.int32, copy=False)
        upd_alpha = z[f"upd_alpha_{subj}"].astype(np.int32, copy=False)
        upd_type  = z[f"upd_type_{subj}"].astype(np.int32, copy=False)

        models[subj] = {
            "labels": labels,
            "P_best": P_best,
            "best_acc": best_acc,
            "best_seen": best_seen,
            "best_updates": best_updates,
            "idx_perm": idx_perm,
            "pers_idx": pers_idx,
            "test_idx": test_idx,
            "upd_t": upd_t,
            "upd_ytrue": upd_ytrue,
            "upd_yhat": upd_yhat,
            "upd_alpha": upd_alpha,
            "upd_type": upd_type,
        }

    return feat_cols, subjects, exp_meta, models


# ============================================================
# EXP3: rollback/unlearn personalization
# ============================================================
UPD_WRONG = 0
UPD_LOW_MARGIN_CORRECT = 1

def unlearn_subject(subj, data_by_subject, encoder, loso_models, saved):
    X = data_by_subject[subj]["X"]
    y = data_by_subject[subj]["y"]

    fold = loso_models[subj]
    mu, sigma = fold["mu"], fold["sigma"]
    labels = fold["labels"]

    # Use EXACT same split used in Exp2
    test_idx = saved["test_idx"]
    pers_idx = saved["pers_idx"]

    X_test, y_test = X[test_idx], y[test_idx]
    X_pers, y_pers = X[pers_idx], y[pers_idx]

    # Encode test set once
    H_test = np.empty((len(y_test), encoder.D), dtype=np.int8)
    for i in range(len(y_test)):
        H_test[i] = encoder.encode_window(X_test[i], mu, sigma)

    # Baseline prototypes (LOSO)
    P_base = {lab: fold["P"][lab].copy() for lab in labels}
    baseline_acc = eval_on_set(H_test, y_test, labels, P_base)

    # Start from best personalized prototypes
    P = {lab: saved["P_best"][lab].copy() for lab in labels}
    best_acc_check = eval_on_set(H_test, y_test, labels, P)

    # Only undo updates that happened BEFORE the best checkpoint time
    best_seen = saved["best_seen"]
    upd_t     = saved["upd_t"]
    upd_ytrue = saved["upd_ytrue"]
    upd_yhat  = saved["upd_yhat"]
    upd_alpha = saved["upd_alpha"]
    upd_type  = saved["upd_type"]

    mask = (upd_t < best_seen)
    upd_t2     = upd_t[mask]
    upd_ytrue2 = upd_ytrue[mask]
    upd_yhat2  = upd_yhat[mask]
    upd_alpha2 = upd_alpha[mask]
    upd_type2  = upd_type[mask]

    U = int(len(upd_t2))

    print(f"\n[SUBJ {subj}] baseline_acc={baseline_acc*100:.2f}% best_acc(saved)={saved['best_acc']*100:.2f}% best_acc(recalc)={best_acc_check*100:.2f}%")
    print(f"[SUBJ {subj}] rollback updates available (<=best_seen): {U}")

    # Decide rollback evaluation points
    eval_points = set()
    if ALSO_EVAL_AT_EDGES:
        eval_points.add(0)
        eval_points.add(U)
    step = max(1, int(EVAL_EVERY_UPDATES))
    for k in range(0, U + 1, step):
        eval_points.add(k)
    eval_points = sorted(eval_points)

    records = []
    eval_ptr = 0
    target_k = eval_points[eval_ptr]

    undone = 0

    # Evaluate initial (still personalized)
    if target_k == 0:
        acc0 = eval_on_set(H_test, y_test, labels, P)
        records.append({
            "subject": subj,
            "undone_updates": 0,
            "remaining_updates": U,
            "unlearn_fraction": 0.0 if U > 0 else 1.0,
            "test_acc": acc0,
        })
        print(f"[SUBJ {subj}] undone=0/{U} acc={acc0*100:.2f}%")
        eval_ptr += 1
        if eval_ptr < len(eval_points):
            target_k = eval_points[eval_ptr]

    # Rollback in reverse order
    for j in range(U - 1, -1, -1):
        t = int(upd_t2[j])
        y_true = int(upd_ytrue2[j])
        y_hat  = int(upd_yhat2[j])
        a      = int(upd_alpha2[j])
        typ    = int(upd_type2[j])

        hv = encoder.encode_window(X_pers[t], mu, sigma).astype(np.int32)

        if typ == UPD_WRONG:
            # forward: P[y_true]+=a*hv; P[y_hat]-=a*hv
            P[y_true] -= a * hv
            P[y_hat]  += a * hv
        elif typ == UPD_LOW_MARGIN_CORRECT:
            # forward: P[y_true]+=a*hv
            P[y_true] -= a * hv
        else:
            raise RuntimeError(f"Unknown update type {typ} for subject {subj}")

        undone += 1

        if eval_ptr < len(eval_points) and undone == target_k:
            acc = eval_on_set(H_test, y_test, labels, P)
            frac = float(undone) / float(U) if U > 0 else 1.0
            records.append({
                "subject": subj,
                "undone_updates": undone,
                "remaining_updates": U - undone,
                "unlearn_fraction": frac,
                "test_acc": acc,
            })
            print(f"[SUBJ {subj}] undone={undone}/{U} acc={acc*100:.2f}%")
            eval_ptr += 1
            if eval_ptr < len(eval_points):
                target_k = eval_points[eval_ptr]

    fully_unlearned_acc = records[-1]["test_acc"] if records else best_acc_check
    return baseline_acc, best_acc_check, fully_unlearned_acc, U, records


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print(f"[EXP3] Loading personalized models/logs from: {PERSONALIZED_MODELS_NPZ}")
    feat_cols_p, subjects_p, exp_meta, pers_models = load_personalized_models_npz(PERSONALIZED_MODELS_NPZ)

    print(f"[DATA] Loading cached windows from: {DATASET_CACHE_NPZ}")
    data_by_subject, feat_cols_data = load_cached_dataset_npz(DATASET_CACHE_NPZ)

    print(f"[LOSO] Loading LOSO models from: {LOSO_MODELS_NPZ}")
    feat_cols_model, loso_models, meta = load_loso_models_npz(LOSO_MODELS_NPZ)
    print("[LOSO] meta:", meta)

    print(f"[ENC] Loading pooled encoder dict from: {ENCODER_DICT_NPZ}")
    feat_cols_enc, meta_enc, ID_feat, VAL, ID_seg = load_encoder_dict_npz_pooled(ENCODER_DICT_NPZ)
    print("[ENC] meta:", meta_enc)

    # Consistency checks
    if feat_cols_data != feat_cols_model or feat_cols_data != feat_cols_enc or feat_cols_data != feat_cols_p:
        raise ValueError("Feature columns mismatch between dataset/model/encoder/personalized NPZs.")

    # Pull encoder settings from meta (LOSO + encoder dict)
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
        raise ValueError("use_time=True but pooled encoder dict has no ID_seg saved.")

    encoder = HDCEncoderPooled(
        D=D, Q=Q, seed=seed, use_time=use_time, win_size=win_size, clamp_z=clamp_z,
        n_segments=n_segments, segment_stats=segment_stats, use_delta=use_delta
    )
    encoder.set_encoding_dict(feat_cols_data, ID_feat=ID_feat, VAL=VAL, ID_seg=ID_seg)

    subjects = sorted(set(subjects_p) & set(loso_models.keys()) & set(data_by_subject.keys()), key=int)
    print(f"[EXP3] subjects to run: {subjects}")

    curve_rows = []
    summary_rows = []

    for subj in subjects:
        base_acc, best_acc, unlearn_acc, U, records = unlearn_subject(
            subj, data_by_subject, encoder, loso_models, pers_models[subj]
        )
        curve_rows.extend(records)
        summary_rows.append({
            "subject": subj,
            "baseline_acc": base_acc,
            "best_personalized_acc": best_acc,
            "fully_unlearned_acc": unlearn_acc,
            "best_minus_baseline": best_acc - base_acc,
            "unlearned_minus_baseline": unlearn_acc - base_acc,
            "rollback_updates": U,
            "eval_every_updates": EVAL_EVERY_UPDATES,
            "n_segments": n_segments,
            "segment_stats": str(segment_stats),
            "use_delta": use_delta,
            "D": D, "Q": Q, "seed": seed, "use_time": use_time, "win_size": win_size, "clamp_z": clamp_z,
        })

    curve_df = pd.DataFrame(curve_rows)
    summary_df = pd.DataFrame(summary_rows)

    curve_path = os.path.join(OUT_DIR, "exp3_unlearn_curve_pooled.csv")
    summary_path = os.path.join(OUT_DIR, "exp3_unlearn_summary_pooled.csv")

    curve_df.to_csv(curve_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"\n[SAVED] curve:   {curve_path}")
    print(f"[SAVED] summary: {summary_path}")

    if len(summary_df) > 0:
        print("\nSummary (mean over subjects):")
        print(" baseline_acc:", float(summary_df["baseline_acc"].mean()))
        print(" best_acc:    ", float(summary_df["best_personalized_acc"].mean()))
        print(" unlearn_acc: ", float(summary_df["fully_unlearned_acc"].mean()))