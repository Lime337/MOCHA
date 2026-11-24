import os
import time
import threading
import numpy as np
import torch
from hashlib import md5
from typing import Dict, List
from androguard.core.bytecodes.apk import APK
from androguard.core.bytecodes.dvm import DalvikVMFormat


try:
    import pynvml
    pynvml.nvmlInit()
    NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception as e:
    print(f'[WARN] NVML not available: {e}')
    pynvml = None
    NVML_HANDLE = None


D = 2048
MAX_OPS_PER_APP = 10000
MAX_APKS_PER_CLASS = 300
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 1337):
    np.random.seed(seed)
    torch.manual_seed(seed)


_OPVEC_CACHE: Dict[tuple, torch.Tensor] = {}
_METHODVEC_CACHE: Dict[tuple, torch.Tensor] = {}


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


def extract_opcodes_by_method_fast(apk_path: str, max_total_ops: int = MAX_OPS_PER_APP) -> Dict[str, List[str]]:
    try:
        a = APK(apk_path)
        dex_list = a.get_all_dex() or [a.get_dex()]
        if not dex_list:
            return {}
        method_to_opcodes: Dict[str, List[str]] = {}
        total_ops = 0
        for dex_bytes in dex_list:
            dvm = DalvikVMFormat(dex_bytes)
            for method in dvm.get_methods():
                code = method.get_code()
                if code is None:
                    continue
                try:
                    bc = code.get_bc()
                    op_list = [ins.get_name() for ins in bc.get_instructions()]
                    if not op_list:
                        continue
                    method_name = method.get_class_name() + "->" + method.get_name()
                    remaining = max_total_ops - total_ops
                    if remaining <= 0:
                        return method_to_opcodes
                    clipped = op_list[:remaining]
                    method_to_opcodes[method_name] = clipped
                    total_ops += len(clipped)
                    if total_ops >= max_total_ops:
                        return method_to_opcodes
                except Exception:
                    continue
        return method_to_opcodes
    except Exception as e:
        print(f'[ERROR] Failed to parse {apk_path}: {e}')
        return {}


def preload_opmaps_paired(benign_folder: str, malware_folder: str, max_apks: int):
    
    benign_files = sorted(f for f in os.listdir(benign_folder) if f.lower().endswith(".apk"))
    malware_files = sorted(f for f in os.listdir(malware_folder) if f.lower().endswith(".apk"))

    num_files = min(len(benign_files), len(malware_files), max_apks)
    benign_maps, malware_maps = [], []

    for i in range(num_files):
        bpath = os.path.join(benign_folder, benign_files[i])
        mpath = os.path.join(malware_folder, malware_files[i])

        print(f'Preloading [{i + 1}/{num_files}] benign {benign_files[i]} and malware {malware_files[i]}')

        b_map = extract_opcodes_by_method_fast(bpath)
        m_map = extract_opcodes_by_method_fast(mpath)

        if b_map:
            benign_maps.append(b_map)
        if m_map:
            malware_maps.append(m_map)

    return benign_maps, malware_maps


def encode_app_from_method_opcodes(method_op_map: Dict[str, List[str]], D: int = D) -> torch.Tensor:
    acc = torch.zeros(D, dtype=torch.int32, device=DEVICE)
    ops_used = 0
    for method_name, op_list in method_op_map.items():
        if ops_used >= MAX_OPS_PER_APP:
            break
        mvec = method_name_to_vec(method_name, D)
        remaining = MAX_OPS_PER_APP - ops_used
        for op in op_list[:remaining]:
            acc += bind(opcode_vec(op, D), mvec).to(torch.int32)
            ops_used += 1
            if ops_used >= MAX_OPS_PER_APP:
                break
    return sign_bundle(acc)

def build_class_vectors_from_preloaded(benign_maps, malware_maps, D: int = D):
    benign_acc = torch.zeros(D, dtype=torch.int32, device=DEVICE)
    malware_acc = torch.zeros(D, dtype=torch.int32, device=DEVICE)

    num_files = min(len(benign_maps), len(malware_maps))
    for i in range(num_files):
        benign_acc += encode_app_from_method_opcodes(benign_maps[i], D).to(torch.int32)
        malware_acc += encode_app_from_method_opcodes(malware_maps[i], D).to(torch.int32)

    return sign_bundle(benign_acc), sign_bundle(malware_acc)


def sample_gpu_power(interval: float, readings: list, stop_flag: list):
    if not (pynvml and NVML_HANDLE):
        return
    while not stop_flag[0]:
        try:
            p = pynvml.nvmlDeviceGetPowerUsage(NVML_HANDLE) / 1000.0
            readings.append(p)
        except:
            pass
        time.sleep(interval)


if __name__ == "__main__":
    benign_folder = "/tmleclab/androzoo_dataset/benign_apks"
    malware_folder = "/tmleclab/androzoo_dataset/malware_apks"
    set_seed(42)


    benign_maps, malware_maps = preload_opmaps_paired(
        benign_folder, malware_folder, MAX_APKS_PER_CLASS
    )


    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    power_readings = []
    stop_flag = [False]
    if pynvml and NVML_HANDLE:
        t = threading.Thread(target=sample_gpu_power, args=(0.2, power_readings, stop_flag))
        t.start()

    start_time = time.time()
    benign_vec, malware_vec = build_class_vectors_from_preloaded(benign_maps, malware_maps)
    end_time = time.time()

    stop_flag[0] = True
    if pynvml and NVML_HANDLE:
        t.join()

    elapsed = end_time - start_time

    if DEVICE.type == "cuda":
        mem_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
        max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        mem_alloc = max_mem = 0

    avg_power = sum(power_readings) / len(power_readings) if power_readings else -1


    print('Class vectors built.')


    print(f'Bundling+Binding time: {elapsed:.2f} sec')
    print(f'GPU memory allocated: {mem_alloc:.2f} MB (max {max_mem:.2f} MB)')
    if avg_power >= 0:
        print(f'Avg GPU power usage: {avg_power:.2f} W (sampled every 0.2s)')

    np.save("benign_class_vector.npy", benign_vec.cpu().numpy().astype(np.int8))
    np.save("malware_class_vector.npy", malware_vec.cpu().numpy().astype(np.int8))
    print('Saved class vectors to benign_class_vector.npy and malware_class_vector.npy')
