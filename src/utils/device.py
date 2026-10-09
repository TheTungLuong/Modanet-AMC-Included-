"""Chọn thiết bị + tự cấu hình theo máy (CUDA mọi đời GPU / Apple MPS / CPU).

Chạy trực tiếp để kiểm máy:  python -m src.utils.device
"""
import multiprocessing
import os
import re
import shutil
import sys
import warnings

import torch

from configs import train_config as TC


# ------------------------------------------------------------------ CUDA
def _arch_supports(cap, arch_list):
    """cubin sm_XY chạy được trên GPU X.Z nếu Z >= Y (cùng major); PTX compute_XY
    JIT được trên mọi GPU >= X.Y."""
    major, minor = cap
    for a in arch_list:
        m = re.fullmatch(r"(sm|compute)_(\d+?)(\d)", a)
        if not m:
            continue
        kind, ma, mi = m.group(1), int(m.group(2)), int(m.group(3))
        if kind == "sm" and ma == major and mi <= minor:
            return True
        if kind == "compute" and (ma, mi) <= (major, minor):
            return True
    return False


def _kernel_test(device):
    """Chạy kernel thật (conv + pool + backward): bắt lỗi 'no kernel image is available'."""
    try:
        x = torch.randn(2, 5, 64, 2, device=device)
        conv = torch.nn.Conv2d(5, 8, (7, 1), padding=(3, 0)).to(device)
        torch.nn.functional.max_pool2d(conv(x), (3, 1), (2, 1), (1, 0)).sum().backward()
        synchronize(device)
        return "OK"
    except RuntimeError as e:
        return f"FAIL: {e}"


def gpu_report():
    rep = {"torch": torch.__version__, "cuda_build": torch.version.cuda}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rep["cuda_available"] = torch.cuda.is_available()
        rep["init_warning"] = str(w[0].message) if w else ""
    rep["mps_available"] = _mps_available()
    if not rep["cuda_available"]:
        return rep
    rep["n_gpus"] = torch.cuda.device_count()
    rep["name"] = torch.cuda.get_device_name(0)
    rep["capability"] = torch.cuda.get_device_capability(0)
    rep["arch_list"] = torch.cuda.get_arch_list()
    rep["arch_compatible"] = _arch_supports(rep["capability"], rep["arch_list"])
    rep["vram_gb"] = torch.cuda.get_device_properties(0).total_memory / 2**30
    rep["kernel_test"] = _kernel_test(torch.device("cuda"))
    return rep


def advice(rep):
    msg = []
    if not rep["cuda_available"]:
        wmsg = rep.get("init_warning", "")
        if "804" in wmsg or "forward compatibility" in wmsg or "mismatch" in wmsg.lower():
            msg.append("Driver NVIDIA vừa cập nhật nhưng kernel module cũ còn chạy "
                       "(nvidia-smi: 'Driver/library version mismatch') -> KHỞI ĐỘNG LẠI MÁY.")
        elif torch.version.cuda is None:
            msg.append("Bản torch này là bản CPU-only. Có GPU NVIDIA thì cài bản CUDA: "
                       "https://pytorch.org/get-started/locally/")
    elif rep.get("kernel_test") != "OK":
        cap = "sm_%d%d" % rep["capability"]
        msg.append(f"GPU {rep['name']} ({cap}) không có trong bản torch {rep['torch']} "
                   f"(arch list {rep['arch_list']}). Cài bản torch hỗ trợ {cap}: GPU cũ (Pascal "
                   f"sm_6x) -> torch==2.6.0; GPU mới (Blackwell sm_12x, RTX 50xx) -> torch>=2.7 bản cu128.")
    return msg


# ------------------------------------------------------------------ MPS / chung
def _mps_available():
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def synchronize(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def select_device(pref="auto", verbose=True):
    """pref: auto | cuda | cuda:N | mps | cpu. auto: CUDA dùng được -> MPS -> CPU."""
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "mps" or (pref == "auto" and not torch.cuda.is_available() and _mps_available()):
        if not _mps_available():
            sys.exit("--device mps nhưng máy không có Apple MPS.")
        if verbose:
            print("Thiết bị: Apple MPS")
        return torch.device("mps")
    rep = gpu_report()
    ok = rep["cuda_available"] and rep.get("kernel_test") == "OK"
    if verbose:
        if ok:
            print(f"GPU: {rep['name']} cap={rep['capability']} VRAM={rep['vram_gb']:.1f} GB "
                  f"({rep['n_gpus']} GPU) | torch {rep['torch']} cu{rep['cuda_build']}")
        for m in advice(rep):
            print("⚠️ ", m)
    if ok:
        return torch.device(pref if pref.startswith("cuda:") else "cuda")
    if pref.startswith("cuda"):
        sys.exit("--device cuda nhưng GPU không dùng được (xem cảnh báo ở trên).")
    if verbose:
        print("⚠️  Không có GPU dùng được -> CPU (train đầy đủ sẽ rất chậm).")
    return torch.device("cpu")


# ------------------------------------------------------------------ tự cấu hình
def auto_micro_batch(device, effective=TC.BATCH_SIZE):
    """Micro-batch theo VRAM — MODANET_INSTRUCTION.md 9.3 (4 GB -> 64, 2 GB -> 32), mở rộng:
    >= 6 GB -> 128. Batch hiệu dụng luôn = effective (accum = effective // micro)."""
    device = torch.device(device)
    if device.type != "cuda":
        return effective
    gb = torch.cuda.get_device_properties(device).total_memory / 2**30
    for min_gb, micro in TC.AUTO_BATCH_BY_VRAM_GB:
        if gb >= min_gb:
            return min(micro, effective)
    return min(TC.AUTO_BATCH_BY_VRAM_GB[-1][1], effective)


def available_cpus():
    try:
        return len(os.sched_getaffinity(0))      # tôn trọng giới hạn cgroup/cpuset (docker)
    except AttributeError:                        # Windows / macOS
        return os.cpu_count() or 1


def shm_free_bytes():
    """Dung lượng trống của /dev/shm (Linux/docker). None nếu không áp dụng."""
    try:
        return shutil.disk_usage("/dev/shm").free
    except (FileNotFoundError, OSError):
        return None


def auto_num_workers(batch_bytes, prefetch=TC.PREFETCH_FACTOR, verbose=True):
    """min(NUM_WORKERS, số CPU - 1), rồi hạ xuống nếu /dev/shm quá nhỏ (docker mặc định 64 MB
    -> DataLoader chết với 'Bus error')."""
    n = max(0, min(TC.NUM_WORKERS, available_cpus() - 1))
    free = shm_free_bytes()
    if free is not None and n > 0:
        while n > 0 and n * prefetch * batch_bytes * 2 > free:
            n -= 1
        if verbose and n < min(TC.NUM_WORKERS, available_cpus() - 1):
            print(f"⚠️  /dev/shm chỉ còn {free / 2**20:.0f} MB -> num_workers = {n}. "
                  f"(docker: chạy với --shm-size=8g để nhanh hơn)")
    return n


def workers_share_memory():
    """True nếu worker DataLoader được tạo bằng fork -> dùng chung mảng trong RAM (copy-on-write).
    Chưa đặt start method thì lấy mặc định của hệ: Linux fork (Python < 3.14);
    macOS, Windows, Linux Python >= 3.14 là spawn/forkserver -> mỗi worker chép lại dữ liệu."""
    method = (torch.multiprocessing.get_start_method(allow_none=True)
              or multiprocessing.get_all_start_methods()[0])
    return method == "fork"


def resolve_num_workers(requested, batch_bytes, in_memory=False, verbose=True):
    """'auto' | int -> số worker. --in-memory chỉ dùng worker khi fork (không chép 9.2 GB sang
    từng worker); khi đó worker chuẩn bị batch song song với GPU, tránh CPU thành nút cổ chai."""
    if in_memory and requested != 0 and not workers_share_memory():
        if verbose and requested != "auto":
            print("⚠️  --in-memory + worker không dùng fork sẽ chép dữ liệu sang từng worker -> num_workers = 0")
        return 0
    return auto_num_workers(batch_bytes, verbose=verbose) if requested == "auto" else requested


def mem_available_bytes():
    """RAM còn trống (Linux /proc/meminfo). None nếu không đọc được."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


if __name__ == "__main__":
    r = gpu_report()
    for k, v in r.items():
        print(f"{k:16s}: {v}")
    for m in advice(r):
        print("⚠️ ", m)
    dev = select_device("auto", verbose=False)
    print(f"{'-> device':16s}: {dev}")
    print(f"{'-> micro-batch':16s}: {auto_micro_batch(dev)} × accum "
          f"{TC.BATCH_SIZE // auto_micro_batch(dev)} = {TC.BATCH_SIZE}")
    print(f"{'-> CPU / shm':16s}: {available_cpus()} CPU, /dev/shm trống "
          f"{(shm_free_bytes() or 0) / 2**30:.1f} GB, RAM trống {(mem_available_bytes() or 0) / 2**30:.1f} GB")
    ok = dev.type != "cpu" or not (r["cuda_available"] and r.get("kernel_test") != "OK")
    sys.exit(0 if ok else 1)
