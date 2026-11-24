import os
import numpy as np
import pandas as pd


DATASET_CACHE_NPZ = r"pamap2_windows_w512_s256_m0p900_notemp_wrist.npz"


LOSO_MODELS_NPZ   = r"pamap2_loso_models_w512_s256_m0p900_notemp_wrist_D4096_Q21_seed0_time_cz1p0.npz"


ENCODER_DICT_NPZ  = r"pamap2_encoder_dict_w512_s256_m0p900_notemp_wrist_D4096_Q21_seed0_time_cz1p0_seg4_statsmean-std_dlt.npz"


OUT_DIR = "exp2_outputs_pooled"
os.makedirs(OUT_DIR, exist_ok=True)


PERS_FRACTION = 0.70
ORDER = "shuffle"
SPLIT_SEED = 42
EVAL_EVERY = 50


PATIENCE_EVALS = 5
MIN_DELTA = 1e-4
REPORT_BEST = True


ALPHA_WRONG   = 10
ALPHA_CORRECT = 2
MARGIN_TAU    = 0.02


SAVE_PERSONALIZED_MODELS = True
PERSONALIZED_MODELS_NPZ = os.path.join(
    OUT_DIR,
    f"exp2_personalized_models_pooled_pf{int(PERS_FRACTION*100)}_{ORDER}_seed{SPLIT_SEED}_"
    f"aw{ALPHA_WRONG}_ac{ALPHA_CORRECT}_tau{str(MARGIN_TAU).replace('.','p')}.npz"
)


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


def load_encoder_dict_npz(path):
    z = np.load(path, allow_pickle=True)
    feat_cols = list(z["feat_cols"].tolist())
    meta = z["meta"].tolist()
    ID_feat = z["ID_feat"].astype(np.int8, copy=False)
    VAL     = z["VAL"].astype(np.int8, copy=False)
    ID_seg  = z["ID_seg"].astype(np.int8, copy=False) if "ID_seg" in z.files else None
    return feat_cols, meta, ID_feat, VAL, ID_seg


def load_loso_models_npz(path):
    z = np.load(path, allow_pickle=True)
    feat_cols = list(z["feat_cols"].tolist())
    subjects = [str(s) for s in z["subjects"].tolist()]
    meta = z["meta"].tolist()
    models = {}
    for subj in subjects:
        mu = z[f"mu_{subj}"].astype(np.float32, copy=False)
        sigma = z[f"sigma_{subj}"].astype(np.float32, copy=False)
        labels = z[f"labels_{subj}"].astype(np.int32, copy=False).tolist()
        P = {int(lab): z[f"P_{subj}_{int(lab)}"].astype(np.int32, copy=False) for lab in labels}
        models[subj] = {"mu": mu, "sigma": sigma, "labels": labels, "P": P}
    return feat_cols, models, meta


class HDCEncoder:
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
            raise ValueError("use_time=True but encoder dict has no ID_seg saved.")
        if self.use_time and self.ID_seg.shape[0] != self.n_segments:
            raise ValueError(
                f"ID_seg has K={self.ID_seg.shape[0]} but encoder expects n_segments={self.n_segments}. "
                f"Fix n_segments to match the encoder dict meta."
            )

    def quantize(self, Z):
        Z = np.nan_to_num(Z, nan=0.0, posinf=self.clamp_z, neginf=-self.clamp_z)
        Z = np.clip(Z, -self.clamp_z, self.clamp_z)
        u = (Z + self.clamp_z) / (2 * self.clamp_z)
        q = np.floor(u * (self.Q - 1 + 1e-8)).astype(np.int32)
        return np.clip(q, 0, self.Q - 1)

    def _encode_stat_vector(self, stat_vec, mu, sigma_safe, seg_id=None):
        Z = (stat_vec - mu) / sigma_safe
        q = self.quantize(Z)
        v = self.VAL[q]
        hv = (self.ID_feat * v).sum(axis=0)
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
            seg = W[s:e]

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


def _stack_P(labels, P_dict):
    return np.stack([P_dict[int(l)] for l in labels], axis=0)

def scores_for(hv, labels, P_dict):
    Pm = _stack_P(labels, P_dict).astype(np.float32)
    hvf = hv.astype(np.float32)
    dots = Pm @ hvf
    norms = np.linalg.norm(Pm, axis=1) + 1e-8
    return dots / norms

def eval_on_set(H, y, labels, P_dict):
    P = _stack_P(labels, P_dict).astype(np.float32)
    norms = np.linalg.norm(P, axis=1) + 1e-8
    scores = (H.astype(np.float32) @ P.T) / norms[None, :]
    pred_idx = np.argmax(scores, axis=1)
    y_pred = np.array([labels[i] for i in pred_idx], dtype=np.int32)
    return float((y_pred == y).mean())


def save_personalized_models_npz(path, feat_cols, models_dict, exp_meta):
    payload = {
        "feat_cols": np.array(list(feat_cols), dtype=object),
        "subjects": np.array(sorted(models_dict.keys(), key=int), dtype=object),
        "exp_meta": np.array(dict(exp_meta), dtype=object),
    }

    for subj, m in models_dict.items():
        payload[f"labels_{subj}"] = np.array(m["labels"], dtype=np.int32)

        for lab in m["labels"]:
            payload[f"Pbest_{subj}_{int(lab)}"] = m["P_best"][int(lab)].astype(np.int32, copy=False)

        payload[f"best_acc_{subj}"] = np.array([float(m["best_acc"])], dtype=np.float32)
        payload[f"best_seen_{subj}"] = np.array([int(m["best_seen"])], dtype=np.int32)
        payload[f"best_updates_{subj}"] = np.array([int(m["best_updates"])], dtype=np.int32)

        payload[f"idxperm_{subj}"] = m["idx_perm"].astype(np.int32, copy=False)
        payload[f"pers_idx_{subj}"] = m["pers_idx"].astype(np.int32, copy=False)
        payload[f"test_idx_{subj}"] = m["test_idx"].astype(np.int32, copy=False)

        payload[f"upd_t_{subj}"]     = m["upd_t"].astype(np.int32, copy=False)
        payload[f"upd_ytrue_{subj}"] = m["upd_ytrue"].astype(np.int32, copy=False)
        payload[f"upd_yhat_{subj}"]  = m["upd_yhat"].astype(np.int32, copy=False)
        payload[f"upd_alpha_{subj}"] = m["upd_alpha"].astype(np.int32, copy=False)
        payload[f"upd_type_{subj}"]  = m["upd_type"].astype(np.int32, copy=False)

    np.savez_compressed(path, **payload)


UPD_WRONG = 0
UPD_LOW_MARGIN_CORRECT = 1

def run_personalization_for_subject(subj, data_by_subject, encoder, loso_models):
    X = data_by_subject[subj]["X"]
    y = data_by_subject[subj]["y"]

    fold = loso_models[subj]
    mu, sigma = fold["mu"], fold["sigma"]
    labels = fold["labels"]

    P = {lab: fold["P"][lab].copy() for lab in labels}

    idx_perm = np.arange(len(y), dtype=np.int32)
    if ORDER == "shuffle":
        rng = np.random.default_rng(SPLIT_SEED + int(subj))
        rng.shuffle(idx_perm)

    n_pers = int(round(PERS_FRACTION * len(idx_perm)))
    pers_idx = idx_perm[:n_pers].copy()
    test_idx = idx_perm[n_pers:].copy()

    X_pers, y_pers = X[pers_idx], y[pers_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f'[SUBJ {subj}] pers={len(y_pers)} test={len(y_test)}')


    H_test = np.empty((len(y_test), encoder.D), dtype=np.int8)
    for i in range(len(y_test)):
        H_test[i] = encoder.encode_window(X_test[i], mu, sigma)

    baseline_acc = eval_on_set(H_test, y_test, labels, P)
    print(f'[SUBJ {subj}] baseline test acc: {baseline_acc * 100:.2f}%')

    records = []
    updates = 0

    best_acc = baseline_acc
    best_seen = 0
    best_updates = 0
    best_P = {lab: P[lab].copy() for lab in labels}
    no_improve = 0

    upd_t = []
    upd_ytrue = []
    upd_yhat = []
    upd_alpha = []
    upd_type = []

    for t in range(len(y_pers)):
        hv = encoder.encode_window(X_pers[t], mu, sigma)
        y_true = int(y_pers[t])

        sc = scores_for(hv, labels, P)
        bi = int(np.argmax(sc))
        si = int(np.argsort(sc)[-2])
        y_hat = int(labels[bi])
        margin = float(sc[bi] - sc[si])

        hv_i32 = hv.astype(np.int32)

        if y_hat != y_true:
            P[y_true] += ALPHA_WRONG * hv_i32
            P[y_hat]  -= ALPHA_WRONG * hv_i32
            updates += 1

            upd_t.append(t)
            upd_ytrue.append(y_true)
            upd_yhat.append(y_hat)
            upd_alpha.append(ALPHA_WRONG)
            upd_type.append(UPD_WRONG)

        else:
            if margin < MARGIN_TAU:
                P[y_true] += ALPHA_CORRECT * hv_i32
                updates += 1

                upd_t.append(t)
                upd_ytrue.append(y_true)
                upd_yhat.append(y_hat)
                upd_alpha.append(ALPHA_CORRECT)
                upd_type.append(UPD_LOW_MARGIN_CORRECT)

        if (t + 1) % EVAL_EVERY == 0 or (t + 1) == len(y_pers):
            acc = eval_on_set(H_test, y_test, labels, P)

            improved = (acc - best_acc) > MIN_DELTA
            if improved:
                best_acc = acc
                best_seen = t + 1
                best_updates = updates
                best_P = {lab: P[lab].copy() for lab in labels}
                no_improve = 0
            else:
                no_improve += 1

            records.append({
                "subject": subj,
                "windows_seen": t + 1,
                "updates": updates,
                "test_acc": acc,
                "best_acc_so_far": best_acc,
                "best_seen": best_seen,
                "best_updates": best_updates,
                "no_improve_evals": no_improve,
            })

            print(f'[SUBJ {subj}] seen={t + 1:4d} updates={updates:4d} test_acc={acc * 100:6.2f}% best={best_acc * 100:6.2f}% no_improve={no_improve}/{PATIENCE_EVALS}')

            if no_improve >= PATIENCE_EVALS:
                print(f'[SUBJ {subj}] early stop (no improvement for {PATIENCE_EVALS} evals).')
                break

    final_acc = best_acc if REPORT_BEST else (records[-1]["test_acc"] if records else baseline_acc)

    save_bundle = {
        "labels": labels,
        "P_best": best_P,
        "best_acc": float(best_acc),
        "best_seen": int(best_seen),
        "best_updates": int(best_updates),
        "idx_perm": idx_perm,
        "pers_idx": pers_idx,
        "test_idx": test_idx,
        "upd_t": np.array(upd_t, dtype=np.int32),
        "upd_ytrue": np.array(upd_ytrue, dtype=np.int32),
        "upd_yhat": np.array(upd_yhat, dtype=np.int32),
        "upd_alpha": np.array(upd_alpha, dtype=np.int32),
        "upd_type": np.array(upd_type, dtype=np.int32),
    }

    return baseline_acc, final_acc, records, save_bundle


if __name__ == "__main__":
    print(f'Loading cached windows from: {DATASET_CACHE_NPZ}')
    data_by_subject, feat_cols_data = load_cached_dataset_npz(DATASET_CACHE_NPZ)
    subjects_data = sorted(data_by_subject.keys(), key=int)
    print(f"subjects={len(subjects_data)} total_windows={sum((v['X'].shape[0] for v in data_by_subject.values()))}")

    print(f'Loading LOSO models from: {LOSO_MODELS_NPZ}')
    feat_cols_model, loso_models, meta = load_loso_models_npz(LOSO_MODELS_NPZ)
    meta = _meta_to_dict(meta)
    print('meta:', meta)

    if feat_cols_data != feat_cols_model:
        raise ValueError("Feature columns in dataset cache do not match LOSO model cache. "
                         "Make sure you used the same wrist_only/include_temp/sensors config.")


    D = int(meta.get("D", 4096))
    Q = int(meta.get("Q", 21))
    seed = int(meta.get("seed", 0))
    use_time = bool(meta.get("use_time", True))
    win_size = int(meta.get("win_size", 256))
    clamp_z = float(meta.get("clamp_z", 1.0))

    print(f'Loading encoder dict from: {ENCODER_DICT_NPZ}')
    feat_cols_enc, meta_enc, ID_feat, VAL, ID_seg = load_encoder_dict_npz(ENCODER_DICT_NPZ)
    meta_enc = _meta_to_dict(meta_enc)

    if feat_cols_enc != feat_cols_data:
        raise ValueError("Feature columns in encoder dict do not match dataset cache.")


    n_segments = int(meta_enc.get("n_segments", 4))
    segment_stats = tuple(meta_enc.get("segment_stats", ("mean",)))
    use_delta = bool(meta_enc.get("use_delta", False))


    if int(meta_enc.get("D", -1)) != D or int(meta_enc.get("Q", -1)) != Q:
        raise ValueError("Encoder dict D/Q does not match LOSO model meta.")
    if bool(meta_enc.get("use_time", True)) != use_time:
        raise ValueError("Encoder dict use_time does not match LOSO model meta.")
    if use_time and ID_seg is None:
        raise ValueError("use_time=True but encoder dict has no ID_seg saved.")

    print(f'pooled meta: n_segments={n_segments} segment_stats={segment_stats} use_delta={use_delta}')

    encoder = HDCEncoder(
        D=D, Q=Q, seed=seed, use_time=use_time, win_size=win_size, clamp_z=clamp_z,
        n_segments=n_segments, segment_stats=segment_stats, use_delta=use_delta
    )
    encoder.set_encoding_dict(feat_cols_data, ID_feat=ID_feat, VAL=VAL, ID_seg=ID_seg)

    curve_rows = []
    summary_rows = []
    personalized_models = {}

    for subj in subjects_data:
        if subj not in loso_models:
            print(f'[WARN] No LOSO model found for subject {subj}; skipping.')
            continue

        base_acc, final_acc, records, save_bundle = run_personalization_for_subject(
            subj, data_by_subject, encoder, loso_models
        )

        summary_rows.append({
            "subject": subj,
            "baseline_acc": base_acc,
            "final_acc": final_acc,
            "delta_acc": final_acc - base_acc,
            "best_seen": save_bundle["best_seen"],
            "best_updates": save_bundle["best_updates"],
            "stopped_seen": records[-1]["windows_seen"] if records else 0,
            "stopped_updates": records[-1]["updates"] if records else 0,
            "pers_fraction": PERS_FRACTION,
            "order": ORDER,
            "eval_every": EVAL_EVERY,
            "alpha_wrong": ALPHA_WRONG,
            "alpha_correct": ALPHA_CORRECT,
            "margin_tau": MARGIN_TAU,
            "patience_evals": PATIENCE_EVALS,
            "min_delta": MIN_DELTA,
            "n_segments": n_segments,
            "segment_stats": str(segment_stats),
            "use_delta": use_delta,
            "D": D, "Q": Q, "seed": seed, "use_time": use_time, "clamp_z": clamp_z,
        })

        curve_rows.extend(records)
        personalized_models[str(subj)] = save_bundle


    curve_df = pd.DataFrame(curve_rows)
    summary_df = pd.DataFrame(summary_rows)

    curve_path = os.path.join(OUT_DIR, "exp2_curve_pooled.csv")
    summary_path = os.path.join(OUT_DIR, "exp2_summary_pooled.csv")

    curve_df.to_csv(curve_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f'curve:   {curve_path}')
    print(f'summary: {summary_path}')

    if len(summary_df) > 0:
        print('Summary (mean over subjects):')
        print(' baseline_acc:', float(summary_df['baseline_acc'].mean()))
        print(' final_acc:   ', float(summary_df['final_acc'].mean()))
        print(' delta_acc:   ', float(summary_df['delta_acc'].mean()))


    if SAVE_PERSONALIZED_MODELS:
        exp_meta = {
            "dataset_cache_npz": DATASET_CACHE_NPZ,
            "loso_models_npz": LOSO_MODELS_NPZ,
            "encoder_dict_npz": ENCODER_DICT_NPZ,
            "PERS_FRACTION": PERS_FRACTION,
            "ORDER": ORDER,
            "SPLIT_SEED": SPLIT_SEED,
            "EVAL_EVERY": EVAL_EVERY,
            "ALPHA_WRONG": ALPHA_WRONG,
            "ALPHA_CORRECT": ALPHA_CORRECT,
            "MARGIN_TAU": MARGIN_TAU,
            "PATIENCE_EVALS": PATIENCE_EVALS,
            "MIN_DELTA": MIN_DELTA,
            "REPORT_BEST": REPORT_BEST,
            "UPD_WRONG": UPD_WRONG,
            "UPD_LOW_MARGIN_CORRECT": UPD_LOW_MARGIN_CORRECT,
            "D": D, "Q": Q, "seed": seed, "use_time": use_time, "win_size": win_size, "clamp_z": clamp_z,
            "n_segments": n_segments, "segment_stats": segment_stats, "use_delta": use_delta,
        }
        print(f'Writing personalized models/logs to: {PERSONALIZED_MODELS_NPZ}')
        save_personalized_models_npz(PERSONALIZED_MODELS_NPZ, feat_cols_data, personalized_models, exp_meta)
        
