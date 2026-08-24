import os
import numpy as np
import torch
from hashlib import md5
from typing import List, Dict, Tuple
from androguard.core.bytecodes.apk import APK
from androguard.core.bytecodes.dvm import DalvikVMFormat
import random

# === CONFIG ===
D = 2048
MAX_OPS_PER_APP = 10000
NUM_TEST_FILES_PER_CLASS = 50
EVAL_TAIL_SIZE = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BENIGN_FOLDER = "/tmleclab/androzoo_dataset/benign_apks"
MALWARE_FOLDER = "/tmleclab/androzoo_dataset/malware_apks"

# === Load class vectors built with GPU training ===
benign_class_vector = torch.from_numpy(
    np.load("benign_class_vector.npy").astype(np.int8)
).to(torch.int8).to(DEVICE)

malware_class_vector = torch.from_numpy(
    np.load("malware_class_vector.npy").astype(np.int8)
).to(torch.int8).to(DEVICE)

# === Caches ===
_OPVEC_CACHE: Dict[Tuple[str, int], torch.Tensor] = {}
_METHODVEC_CACHE: Dict[Tuple[str, int], torch.Tensor] = {}

# === Deterministic full-D bipolar embeddings ===
def _md5_seed(s: str) -> int:
    return int(md5(s.encode("utf-8")).hexdigest()[:8], 16)

def _bipolar_from_seed(seed: int, D: int) -> torch.Tensor:
    rng = np.random.RandomState(seed)
    v = rng.randint(0, 2, size=D, dtype=np.uint8).astype(np.int8)
    v[v == 0] = -1
    return torch.from_numpy(v).to(torch.int8).to(DEVICE)

def opcode_vec(op: str, D: int = D) -> torch.Tensor:
    key = (op, D)
    if key in _OPVEC_CACHE:
        return _OPVEC_CACHE[key]
    seed = _md5_seed("OPC::" + op)
    v = _bipolar_from_seed(seed, D)
    _OPVEC_CACHE[key] = v
    return v

def method_name_to_vec(method_name: str, D: int = D) -> torch.Tensor:
    key = (method_name, D)
    if key in _METHODVEC_CACHE:
        return _METHODVEC_CACHE[key]
    seed = _md5_seed("MTH::" + method_name)
    v = _bipolar_from_seed(seed, D)
    _METHODVEC_CACHE[key] = v
    return v

def bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a * b

def sign_bundle(acc: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(acc, dtype=torch.int8, device=DEVICE)
    out[acc > 0] = 1
    out[acc < 0] = -1
    if torch.any(acc == 0):
        tie = _bipolar_from_seed(0xC0FFEE, acc.numel())
        out[acc == 0] = tie[acc == 0]
    return out

# === Feature extraction for one APK ===
def extract_test_vector(apk_path: str, D: int = D) -> torch.Tensor:
    try:
        a = APK(apk_path)
        dex_list = a.get_all_dex()
        if not dex_list:
            dex = a.get_dex()
            if not dex:
                print(f"[SKIP] No dex found in {apk_path}")
                return torch.zeros(D, dtype=torch.int8, device=DEVICE)
            dex_list = [dex]

        acc = torch.zeros(D, dtype=torch.int32, device=DEVICE)
        total_ops = 0

        for dex_bytes in dex_list:
            if total_ops >= MAX_OPS_PER_APP:
                break
            dvm = DalvikVMFormat(dex_bytes)
            for method in dvm.get_methods():
                if total_ops >= MAX_OPS_PER_APP:
                    break
                code = method.get_code()
                if code is None:
                    continue
                try:
                    bc = code.get_bc()
                    method_name = method.get_class_name() + "->" + method.get_name()
                    mvec = method_name_to_vec(method_name, D)

                    remaining = MAX_OPS_PER_APP - total_ops
                    if remaining <= 0:
                        break

                    used_here = 0
                    for ins in bc.get_instructions():
                        if used_here >= remaining:
                            break
                        op = ins.get_name()
                        acc += bind(opcode_vec(op, D), mvec).to(torch.int32)
                        used_here += 1
                        total_ops += 1
                        if total_ops >= MAX_OPS_PER_APP:
                            break
                except Exception:
                    continue

        return sign_bundle(acc)
    except Exception as e:
        print(f"[ERROR] Failed to parse {apk_path}: {e}")
        return torch.zeros(D, dtype=torch.int8, device=DEVICE)

# === Similarity & classification ===
def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    na = torch.norm(a)
    nb = torch.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(torch.dot(a, b) / (na * nb))

def classify_apk(apk_path: str) -> str:
    test_vec = extract_test_vector(apk_path)
    sim_b = cosine_similarity(test_vec, benign_class_vector)
    sim_m = cosine_similarity(test_vec, malware_class_vector)
    return "benign" if sim_b > sim_m else "malware"

# === Tail sampling for quick eval ===
def get_tail_sample(folder: str, num_files: int) -> List[str]:
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".apk"))
    tail_files = files[-EVAL_TAIL_SIZE:] if EVAL_TAIL_SIZE > 0 else files
    if not tail_files:
        return []
    k = min(num_files, len(tail_files))
    return random.sample(tail_files, k)

# === Metrics ===
def compute_metrics(y_true, y_pred, labels=("benign", "malware")):
    from sklearn.metrics import confusion_matrix, classification_report

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, target_names=labels, digits=4)
    return cm, report

if __name__ == "__main__":
    random.seed(41)

    benign_test = get_tail_sample(BENIGN_FOLDER, NUM_TEST_FILES_PER_CLASS)
    malware_test = get_tail_sample(MALWARE_FOLDER, NUM_TEST_FILES_PER_CLASS)

    y_true, y_pred = [], []

    print("\n=== BENIGN TESTS ===")
    for f in benign_test:
        path = os.path.join(BENIGN_FOLDER, f)
        pred = classify_apk(path)
        print(f"{f}: predicted = {pred}\n")
        y_true.append("benign")
        y_pred.append(pred)

    print("\n=== MALWARE TESTS ===")
    for f in malware_test:
        path = os.path.join(MALWARE_FOLDER, f)
        pred = classify_apk(path)
        print(f"{f}: predicted = {pred}\n")
        y_true.append("malware")
        y_pred.append(pred)

    if y_true:
        cm, report = compute_metrics(y_true, y_pred)
        print("\n✅ Confusion Matrix:")
        print(cm)
        print("\n Classification Report (Precision/Recall/F1):")
        print(report)
    else:
        print("\n⚠️ No test files found to evaluate.")
