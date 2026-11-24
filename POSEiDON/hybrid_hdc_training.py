

import os
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from hybrid_hdc_encoder import encode_hybrid_from_row, load_grid_dict
from angle_hdc_dict import hdc_angle_dict, save_hdc_angle_dict_npz
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


ROOT = r"SobolHybrid"

CSV_PATH = os.path.join(ROOT, "4extracted_angles.csv")
GRID_DICT_JSON = os.path.join(ROOT, "hdc_pose_dict_4096.json")

VAL_FRACTION = 0.2
RANDOM_SEED = 123

MODEL_DIR = os.path.join(ROOT, "4hybrid_angle_grid_model")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def build_class_hypervectors(
    X_train: np.ndarray,
    y_train: List[str],
    labels: List[str],
) -> Dict[str, np.ndarray]:
    
    D = X_train.shape[1]
    class_hvs: Dict[str, np.ndarray] = {}

    for label in labels:
        idxs = [i for i, y in enumerate(y_train) if y == label]
        if not idxs:
            raise ValueError(f"No training samples for label '{label}'")

        acc = np.zeros(D, dtype=np.int32)
        for i in idxs:
            acc += X_train[i].astype(np.int32)

        hv = np.where(acc >= 0, 1, -1).astype(np.int8)
        class_hvs[label] = hv

    return class_hvs


def evaluate_classifier(
    X_val: np.ndarray,
    y_val: List[str],
    class_hvs: Dict[str, np.ndarray],
) -> Tuple[float, dict]:
    
    D = X_val.shape[1]


    for lbl, hv in class_hvs.items():
        if hv.shape[0] != D:
            raise ValueError(
                f"Dimension mismatch for label '{lbl}': "
                f"class_hv has {hv.shape[0]} dims but X_val has {D}"
            )

    y_pred_list = []
    confusion = {}


    for x, y_true in zip(X_val, y_val):
        sims = {label: cosine_similarity(x, hv) for label, hv in class_hvs.items()}
        y_pred = max(sims.items(), key=lambda kv: kv[1])[0]

        y_pred_list.append(y_pred)
        confusion[(y_true, y_pred)] = confusion.get((y_true, y_pred), 0) + 1


    labels_sorted = sorted(class_hvs.keys())

    accuracy = accuracy_score(y_val, y_pred_list)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_val,
        y_pred_list,
        labels=labels_sorted,
        zero_division=0
    )


    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()


    metrics = {
        "labels": labels_sorted,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "confusion": confusion
    }

    return accuracy, metrics


class HybridHdcTrainer:
    def __init__(
        self,
        csv_path: str = CSV_PATH,
        grid_json_path: str = GRID_DICT_JSON,
        val_fraction: float = VAL_FRACTION,
        random_seed: int = RANDOM_SEED,
    ):
        self.csv_path = csv_path
        self.grid_json_path = grid_json_path
        self.val_fraction = val_fraction
        self.random_seed = random_seed

        self.df: pd.DataFrame = pd.DataFrame()
        self.labels: List[str] = []


        self.grid_dict = load_grid_dict(self.grid_json_path)

        D = hdc_angle_dict["dimension"]
        self.X_train: np.ndarray = np.empty((0, D), dtype=np.int8)
        self.y_train: List[str] = []
        self.X_val: np.ndarray = np.empty((0, D), dtype=np.int8)
        self.y_val: List[str] = []

        self.class_hvs: Dict[str, np.ndarray] = {}


    def load_data(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        if "label" not in self.df.columns:
            raise ValueError("CSV must contain a 'label' column.")

        print(f'Loaded {len(self.df)} rows from {self.csv_path}')
        print('Labels present:', self.df['label'].value_counts().to_dict())

    def encode_all_poses(self) -> Tuple[np.ndarray, List[str]]:
        D = hdc_angle_dict["dimension"]
        N = len(self.df)
        X = np.zeros((N, D), dtype=np.int8)
        y: List[str] = []

        kept = 0
        for _, row in self.df.iterrows():
            hv = encode_hybrid_from_row(row, grid_dict=self.grid_dict)
            if not np.any(hv != 0):
                continue
            X[kept] = hv
            y.append(str(row["label"]))
            kept += 1

        X = X[:kept]
        print(f'Encoded {len(X)} valid hybrid poses (out of {len(self.df)})')
        return X, y


    def balance_and_split(self, X: np.ndarray, y: List[str]):
        rng = np.random.default_rng(self.random_seed)

        label_to_indices: Dict[str, List[int]] = {}
        for i, label in enumerate(y):
            label_to_indices.setdefault(label, []).append(i)

        class_sizes = {lbl: len(idxs) for lbl, idxs in label_to_indices.items()}
        print('Class sizes before balancing:', class_sizes)

        min_size = min(class_sizes.values())
        print(f'Balancing to {min_size} samples per class')

        X_train_list = []
        y_train_list: List[str] = []
        X_val_list = []
        y_val_list: List[str] = []

        for label, indices in label_to_indices.items():
            indices = np.array(indices)
            rng.shuffle(indices)
            indices = indices[:min_size]

            n_val = max(1, int(self.val_fraction * min_size))
            val_idx = indices[:n_val]
            train_idx = indices[n_val:]

            X_train_list.append(X[train_idx])
            y_train_list.extend([label] * len(train_idx))

            X_val_list.append(X[val_idx])
            y_val_list.extend([label] * len(val_idx))

        self.X_train = np.vstack(X_train_list)
        self.y_train = y_train_list
        self.X_val = np.vstack(X_val_list)
        self.y_val = y_val_list

        self._shuffle_split_sets(rng)
        self.labels = sorted(label_to_indices.keys())

        print(f'Balanced train size: {len(self.X_train)}')
        print(f'Balanced val size:   {len(self.X_val)}')
        print(f'Using labels: {self.labels}')

    def _shuffle_split_sets(self, rng: np.random.Generator):
        perm_train = rng.permutation(len(self.X_train))
        self.X_train = self.X_train[perm_train]
        self.y_train = [self.y_train[i] for i in perm_train]

        perm_val = rng.permutation(len(self.X_val))
        self.X_val = self.X_val[perm_val]
        self.y_val = [self.y_val[i] for i in perm_val]


    def train(self):
        if self.X_train.size == 0:
            raise RuntimeError("Training set is empty. Run balance_and_split() first.")
        self.class_hvs = build_class_hypervectors(self.X_train, self.y_train, self.labels)
        print('Trained class hypervectors for labels:', self.labels)

    def evaluate(self):
        if not self.class_hvs:
            raise RuntimeError("Class hypervectors not trained. Run train() first.")
        if self.X_val.size == 0:
            raise RuntimeError("Validation set is empty.")


        acc, metrics = evaluate_classifier(self.X_val, self.y_val, self.class_hvs)

        print(f'Validation accuracy: {acc * 100:.2f}%')
        print('Confusion counts (true, pred):')
        for (y_true, y_pred), count in metrics["confusion"].items():
            print(f'  {y_true:>10} -> {y_pred:<10}: {count}')

        return acc, metrics


def save_model_json(trainer: HybridHdcTrainer):
    os.makedirs(MODEL_DIR, exist_ok=True)


    class_hvs_json = {
        label: hv.astype(int).tolist()
        for label, hv in trainer.class_hvs.items()
    }
    class_hvs_path = os.path.join(MODEL_DIR, "class_hvs_hybrid.json")
    with open(class_hvs_path, "w") as f:
        json.dump(class_hvs_json, f)
    print(f'Hybrid class hypervectors JSON saved to {class_hvs_path}')


    angle_dict_json = {
        "dimension": int(hdc_angle_dict["dimension"]),
        "angle_bin_size_deg": float(hdc_angle_dict["angle_bin_size_deg"]),
        "symbols": {},
        "angle_bins": {},
    }
    for name, hv in hdc_angle_dict["symbols"].items():
        angle_dict_json["symbols"][name] = hv.astype(int).tolist()
    for b, hv in hdc_angle_dict["angle_bins"].items():
        angle_dict_json["angle_bins"][str(b)] = hv.astype(int).tolist()

    angle_dict_json_path = os.path.join(MODEL_DIR, "angle_dict.json")
    with open(angle_dict_json_path, "w") as f:
        json.dump(angle_dict_json, f)
    print(f'Angle dictionary JSON saved to {angle_dict_json_path}')


    npz_path = os.path.join(MODEL_DIR, "angle_dict.npz")
    save_hdc_angle_dict_npz(hdc_angle_dict, npz_path)
    print(f'Angle dictionary NPZ saved to {npz_path}')


    grid_basename = os.path.basename(trainer.grid_json_path)
    grid_copy_path = os.path.join(MODEL_DIR, grid_basename)
    if os.path.abspath(trainer.grid_json_path) != os.path.abspath(grid_copy_path):
        with open(trainer.grid_json_path, "r") as f_in, open(grid_copy_path, "w") as f_out:
            f_out.write(f_in.read())
        print(f'Copied grid dictionary JSON to {grid_copy_path}')
    else:
        print('Grid dictionary JSON already in model directory; not copied.')


if __name__ == "__main__":
    trainer = HybridHdcTrainer(
        csv_path=CSV_PATH,
        grid_json_path=GRID_DICT_JSON,
        val_fraction=VAL_FRACTION,
        random_seed=RANDOM_SEED,
    )

    
    trainer.load_data()

    
    X, y = trainer.encode_all_poses()

    
    trainer.balance_and_split(X, y)

    
    trainer.train()

    
    acc, metrics = trainer.evaluate()

    print('HDC model metrics:')
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall:    {metrics['macro_recall']:.4f}")
    print(f"Macro F1 Score:  {metrics['macro_f1']:.4f}")

    print('Per-Class Metrics:')
    print(f"{'Class':<15}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Support':>10}")

    for lbl, p, r, f, s in zip(
        metrics["labels"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["support"],
    ):
        print(f'{lbl:<15}{p:>10.2f}{r:>10.2f}{f:>10.2f}{int(s):>10}')


    total_support = int(sum(metrics["support"]))
    
    print(f"{'Total':<15}{metrics['macro_precision']:>10.3f}{metrics['macro_recall']:>10.3f}{metrics['macro_f1']:>10.3f}{total_support:>10}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
