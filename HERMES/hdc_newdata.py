import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


D = 2048
SAMPLES_PER_FAMILY = 21
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BENIGN_FILE = r"C:/Users/ejmey/OneDrive/Desktop/OOPS/Dataset/CCCS-CIC-Benign-CSVs/Ben0.csv"
MALWARE_DIR = r"C:/Users/ejmey/OneDrive/Desktop/OOPS/Dataset/CCCS-CIC-Malicious-CSVs/"

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

METHOD_NAME = "CSV_METHOD"


from hashlib import md5

def seed_hash(s):
    return int(md5(s.encode()).hexdigest()[:8], 16)

def bipolar(seed, D):
    rng = np.random.RandomState(seed)
    v = rng.randint(0, 2, size=D).astype(np.int8)
    v[v == 0] = -1
    return torch.tensor(v, dtype=torch.int8, device=DEVICE)

_opcode_cache = {}
_method_cache = {}

def opcode_vec(idx):
    name = f"OPCODE_{idx}"
    if name not in _opcode_cache:
        _opcode_cache[name] = bipolar(seed_hash("OPC::" + name), D)
    return _opcode_cache[name]

def method_vec():
    if METHOD_NAME not in _method_cache:
        _method_cache[METHOD_NAME] = bipolar(seed_hash("MTH::" + METHOD_NAME), D)
    return _method_cache[METHOD_NAME]

def bind(a, b):
    return a * b

def sign(a):
    out = torch.empty_like(a, dtype=torch.int8)
    out[a > 0] = 1
    out[a < 0] = -1
    zeros = (a == 0)
    if torch.any(zeros):
        out[zeros] = -1
    return out


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
        hv = bind(opcode_vec(idx), mvec)
        acc += weight * hv.float()

    return sign(acc)


def load_benign(n=300):
    return pd.read_csv(BENIGN_FILE, header=None, nrows=n)

def load_malware_families():
    dfs = []
    for fname in MALWARE_FILES:
        path = os.path.join(MALWARE_DIR, fname)
        df = pd.read_csv(path, header=None, nrows=SAMPLES_PER_FAMILY)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


if __name__ == "__main__":
    print('Using device:', DEVICE)

    df_ben = load_benign(n=300)
    df_mal = load_malware_families()

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )


    acc_ben = torch.zeros(D, dtype=torch.float32, device=DEVICE)
    acc_mal = torch.zeros(D, dtype=torch.float32, device=DEVICE)

    for hv, lab in zip(X_train, y_train):
        if lab == 0:
            acc_ben += hv.float()
        else:
            acc_mal += hv.float()

    C_ben = sign(acc_ben)
    C_mal = sign(acc_mal)

    preds = []
    for hv in X_test:
        sb = torch.dot(hv.float(), C_ben.float()).item()
        sm = torch.dot(hv.float(), C_mal.float()).item()
        preds.append(0 if sb > sm else 1)

    print('CONFUSION MATRIX:')
    print(confusion_matrix(y_test, preds))

    print('CLASSIFICATION REPORT:')
    print(classification_report(y_test, preds, target_names=['Benign', 'Malware'], digits=4))

    print('ACCURACY:')
    print(accuracy_score(y_test, preds))
