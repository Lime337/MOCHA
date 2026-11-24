import os
import numpy as np
import pandas as pd


import HDC_exp4 as base

OUT_DIR = "stability_outputs"
os.makedirs(OUT_DIR, exist_ok=True)


CYCLE_COUNTS = [10, 100, 1000, 10000]


ALPHA = int(base.ALPHA_CORRECT)


POOL_SIZE = 64


def cosine_similarity(a, b):
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den == 0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.dot(a, b) / den)


def prototype_metrics(P_ref, P_cur, labels):
    max_abs_diff = 0
    changed_dims = 0
    min_cos = 1.0
    for lab in labels:
        d = P_cur[lab].astype(np.int64) - P_ref[lab].astype(np.int64)
        max_abs_diff = max(max_abs_diff, int(np.max(np.abs(d))))
        changed_dims += int(np.count_nonzero(d))
        min_cos = min(min_cos, cosine_similarity(P_ref[lab], P_cur[lab]))
    return max_abs_diff, changed_dims, min_cos


def build_encoder_and_data():
    data_by_subject, feat_cols_data = base.load_cached_dataset_npz(base.DATASET_CACHE_NPZ)
    feat_cols_model, loso_models, meta = base.load_loso_models_npz(base.LOSO_MODELS_NPZ)
    feat_cols_enc, meta_enc, ID_feat, VAL, ID_seg = base.load_encoder_dict_npz(base.ENCODER_DICT_NPZ)

    if feat_cols_data != feat_cols_model or feat_cols_data != feat_cols_enc:
        raise ValueError("Feature columns mismatch between dataset / LOSO / encoder dictionary.")

    D = int(meta.get("D", 4096))
    Q = int(meta.get("Q", 21))
    seed = int(meta.get("seed", 0))
    use_time = bool(meta.get("use_time", True))
    win_size = int(meta.get("win_size", 512))
    clamp_z = float(meta.get("clamp_z", 1.0))

    n_segments = int(meta_enc.get("n_segments", 4))
    segment_stats = tuple(meta_enc.get("segment_stats", ("mean", "std")))
    use_delta = bool(meta_enc.get("use_delta", True))

    encoder = base.HDCEncoderPooled(
        D=D, Q=Q, seed=seed, use_time=use_time, win_size=win_size,
        clamp_z=clamp_z, n_segments=n_segments,
        segment_stats=segment_stats, use_delta=use_delta,
    )
    encoder.set_encoding_dict(feat_cols_data, ID_feat=ID_feat, VAL=VAL, ID_seg=ID_seg)
    return data_by_subject, loso_models, encoder


def run_subject(subj, data_by_subject, loso_models, encoder):
    X = data_by_subject[subj]["X"]
    y = data_by_subject[subj]["y"]
    fold = loso_models[subj]
    mu, sigma = fold["mu"], fold["sigma"]
    labels = [int(x) for x in fold["labels"]]
    P_base = {lab: fold["P"][lab].copy() for lab in labels}


    idx = np.arange(len(y), dtype=np.int32)
    rng = np.random.default_rng(base.SPLIT_SEED + int(subj))
    rng.shuffle(idx)
    n_pers = int(round(base.PERS_FRACTION * len(idx)))
    pers_idx = idx[:n_pers]
    test_idx = idx[n_pers:]

    X_pers, y_pers = X[pers_idx], y[pers_idx]
    X_test, y_test = X[test_idx], y[test_idx]


    H_test = np.empty((len(y_test), encoder.D), dtype=np.int8)
    for i in range(len(y_test)):
        H_test[i] = encoder.encode_window(X_test[i], mu, sigma)

    baseline_acc = base.eval_on_set(H_test, y_test, labels, P_base)


    vals, counts = np.unique(y_pers, return_counts=True)
    target_class = int(vals[np.argmax(counts)])
    class_idx = np.where(y_pers == target_class)[0]
    if len(class_idx) == 0:
        raise RuntimeError(f"No personalization windows for target class {target_class}")

    class_idx = class_idx[: min(POOL_SIZE, len(class_idx))]
    H_pool = np.empty((len(class_idx), encoder.D), dtype=np.int32)
    for i, j in enumerate(class_idx):
        H_pool[i] = encoder.encode_window(X_pers[j], mu, sigma).astype(np.int32)

    rows = []
    base_target_max = int(np.max(np.abs(P_base[target_class].astype(np.int64))))
    base_target_norm = float(np.linalg.norm(P_base[target_class].astype(np.float64)))

    for n_updates in CYCLE_COUNTS:
        P = {lab: P_base[lab].copy() for lab in labels}


        for k in range(n_updates):
            hv = H_pool[k % len(H_pool)]
            P[target_class] += ALPHA * hv

        acc_after_add = base.eval_on_set(H_test, y_test, labels, P)
        peak_abs = int(np.max(np.abs(P[target_class].astype(np.int64))))
        norm_after_add = float(np.linalg.norm(P[target_class].astype(np.float64)))


        for k in range(n_updates - 1, -1, -1):
            hv = H_pool[k % len(H_pool)]
            P[target_class] -= ALPHA * hv

        restored_acc = base.eval_on_set(H_test, y_test, labels, P)
        max_diff, changed_dims, min_cos = prototype_metrics(P_base, P, labels)

        rows.append({
            "subject": int(subj),
            "target_class": target_class,
            "updates_added_then_removed": int(n_updates),
            "alpha": ALPHA,
            "baseline_acc": baseline_acc,
            "acc_after_imbalanced_adds": acc_after_add,
            "restored_acc": restored_acc,
            "restored_acc_delta": restored_acc - baseline_acc,
            "max_abs_element_diff_after_restore": max_diff,
            "changed_dimensions_after_restore": changed_dims,
            "min_prototype_cosine_after_restore": min_cos,
            "target_max_abs_before": base_target_max,
            "target_max_abs_after_adds": peak_abs,
            "target_norm_before": base_target_norm,
            "target_norm_after_adds": norm_after_add,
            "exact_restore": int(max_diff == 0 and changed_dims == 0),
        })

        print(f'[SUBJ {subj}] N={n_updates:5d} | acc {baseline_acc:.4f}->{acc_after_add:.4f}->{restored_acc:.4f} | maxdiff={max_diff} changed={changed_dims} cos={min_cos:.12f} | max|P| {base_target_max}->{peak_abs}')

    return rows


def main():
    data_by_subject, loso_models, encoder = build_encoder_and_data()
    subjects = sorted(set(data_by_subject.keys()) & set(loso_models.keys()), key=int)

    rows = []
    for subj in subjects:
        rows.extend(run_subject(subj, data_by_subject, loso_models, encoder))

    df = pd.DataFrame(rows)
    detail_path = os.path.join(OUT_DIR, "reversible_stability_per_subject.csv")
    df.to_csv(detail_path, index=False)

    summary = (
        df.groupby("updates_added_then_removed", as_index=False)
        .agg(
            subjects=("subject", "count"),
            exact_restore_rate=("exact_restore", "mean"),
            max_element_diff=("max_abs_element_diff_after_restore", "max"),
            max_changed_dimensions=("changed_dimensions_after_restore", "max"),
            min_prototype_cosine=("min_prototype_cosine_after_restore", "min"),
            mean_baseline_acc=("baseline_acc", "mean"),
            mean_acc_after_imbalanced_adds=("acc_after_imbalanced_adds", "mean"),
            mean_restored_acc=("restored_acc", "mean"),
            max_abs_acc_restore_delta=("restored_acc_delta", lambda s: float(np.max(np.abs(s)))),
            max_peak_abs_accumulator=("target_max_abs_after_adds", "max"),
        )
    )
    summary_path = os.path.join(OUT_DIR, "reversible_stability_summary.csv")
    summary.to_csv(summary_path, index=False)

    print('REVERSIBLE STABILITY SUMMARY:')
    print(summary.to_string(index=False))
    print(f'{detail_path}')
    print(f'{summary_path}')


if __name__ == "__main__":
    main()
