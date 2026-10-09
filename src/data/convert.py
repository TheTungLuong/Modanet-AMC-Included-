""".mat (450,120 file, ~17 GB) -> memmap .npy đã permute sẵn (chạy 1 lần).

Đầu ra trong out_dir:
    x_<dtype>.npy   (N_TOTAL, 5, 1024, 2)   hàng r = file có chỉ số r+1
    labels.npy      (N_TOTAL, 3) int16      [mod_idx, doa_idx, snr_dB]
    progress.json   số hàng đã ghi (để chạy tiếp sau khi ngắt)
    meta.json       dtype, shape, thứ tự MODULATIONS, thống kê — chỉ ghi khi XONG
"""
import json
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.io as sio

from configs.data_config import (
    N_TOTAL, N_REP, MAT_VARIABLE, RAW_SHAPE, SAMPLE_SHAPE, PERMUTE_RAW_TO_TORCH,
    MODULATIONS, CONVERT_BATCH, CONVERT_DTYPES, AMPLITUDE_LIMIT, SIGNAL_POWER_RANGE, COL_SNR,
    expected_frame_power, idx_to_angle,
)
from src.data.labels import (
    decode_index, index_from_filename, labels_for_all_rows, parse_folder_labels,
)


# --------------------------------------------------------------------- scan
def scan_dataset(root):
    """Duyệt đệ quy root, trả dict {index: path} của mọi file dataXXXXXXXX.mat."""
    found, dup = {}, []
    for dirpath, _, files in os.walk(root):
        for f in files:
            i = index_from_filename(f)
            if i is None:
                continue
            p = os.path.join(dirpath, f)
            if i in found:
                dup.append((i, found[i], p))
            found[i] = p
    return found, dup


def verify_index(found, dup=()):
    """Kiểm hợp đồng dữ liệu ở cấp tên file/thư mục. Trả (ok, report_dict)."""
    idx = np.fromiter(found.keys(), dtype=np.int64, count=len(found))
    rep = {"n_files": int(idx.size), "expected": N_TOTAL,
           "min_index": int(idx.min()) if idx.size else None,
           "max_index": int(idx.max()) if idx.size else None,
           "duplicates": len(dup)}
    missing = np.setdiff1d(np.arange(1, N_TOTAL + 1), idx)
    extra = idx[(idx < 1) | (idx > N_TOTAL)]
    rep["missing"] = int(missing.size)
    rep["missing_examples"] = missing[:10].tolist()
    rep["out_of_range"] = int(extra.size)

    snr, mod, doa, _ = decode_index(idx)
    combos = Counter(zip(snr.tolist(), mod.tolist(), doa.tolist()))
    rep["combos"] = len(combos)
    rep["combos_not_10"] = sum(1 for v in combos.values() if v != N_REP)

    # đối chiếu chéo với tên thư mục (nếu có cấu trúc SNR/MOD/DOA)
    n_folder, bad = 0, []
    for i, s, m, d in zip(idx.tolist(), snr.tolist(), mod.tolist(), doa.tolist()):
        fl = parse_folder_labels(found[i])
        if fl is None:
            continue
        n_folder += 1
        if fl != (s, MODULATIONS[m], idx_to_angle(d)):
            bad.append((i, found[i], (s, MODULATIONS[m], idx_to_angle(d))))
    rep["folder_labels_checked"] = n_folder
    rep["folder_label_mismatch"] = len(bad)
    rep["folder_mismatch_examples"] = [(i, p, str(e)) for i, p, e in bad[:5]]

    ok = (rep["n_files"] == N_TOTAL and rep["missing"] == 0 and rep["duplicates"] == 0
          and rep["out_of_range"] == 0 and rep["combos_not_10"] == 0
          and rep["folder_label_mismatch"] == 0)
    return ok, rep


# ------------------------------------------------------------------- loading
def load_mat_raw(path):
    """Đọc 1 file .mat -> mảng thô (1024, 5, 2) float32 (kiểm hợp đồng)."""
    d = sio.loadmat(path)
    keys = [k for k in d if not k.startswith("__")]
    if keys != [MAT_VARIABLE]:
        raise ValueError(f"{path}: biến {keys}, kỳ vọng ['{MAT_VARIABLE}']")
    a = d[MAT_VARIABLE]
    if a.shape != RAW_SHAPE:
        raise ValueError(f"{path}: shape {a.shape}, kỳ vọng {RAW_SHAPE}")
    return np.asarray(a, dtype=np.float32)


def load_mat_sample(path):
    """Đọc 1 file .mat -> (5, 1024, 2) float32, đúng layout PyTorch (M, N, IQ)."""
    return np.ascontiguousarray(np.transpose(load_mat_raw(path), PERMUTE_RAW_TO_TORCH))


def _load_many(paths):
    return np.stack([load_mat_sample(p) for p in paths])


# ------------------------------------------------------------------- convert
def _read_json(p, default=None):
    try:
        with open(p) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write_json(p, obj):
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, p)


def convert(root, out_dir, dtype="float16", batch=CONVERT_BATCH, workers=4,
            found=None, progress=True):
    """Ghi memmap. Có thể ngắt (Ctrl+C) và chạy lại: tiếp tục từ progress.json."""
    if dtype not in CONVERT_DTYPES:
        raise ValueError(f"dtype {dtype} không thuộc {CONVERT_DTYPES}")
    os.makedirs(out_dir, exist_ok=True)
    x_path = os.path.join(out_dir, f"x_{dtype}.npy")
    prog_path = os.path.join(out_dir, "progress.json")
    shape = (N_TOTAL,) + SAMPLE_SHAPE

    if found is None:
        found, _ = scan_dataset(root)
    if len(found) != N_TOTAL:
        raise RuntimeError(f"chỉ có {len(found)} file / {N_TOTAL} — chạy --verify-index trước")
    paths = [found[i] for i in range(1, N_TOTAL + 1)]      # hàng r = chỉ số r+1

    prog = _read_json(prog_path, {})
    start = prog.get("done", 0) if prog.get("dtype") == dtype and os.path.exists(x_path) else 0
    if start == 0:
        x = np.lib.format.open_memmap(x_path, mode="w+", dtype=dtype, shape=shape)
    else:
        x = np.load(x_path, mmap_mode="r+")
        assert x.shape == shape and str(x.dtype) == dtype, "memmap cũ không khớp — xóa và chạy lại"
        print(f"  tiếp tục từ hàng {start:,}/{N_TOTAL:,}")

    labels = labels_for_all_rows()
    np.save(os.path.join(out_dir, "labels.npy"), labels)

    stats = prog.get("stats", {"amax": 0.0, "nonfinite": 0, "fp16_max_rel_err": 0.0,
                               "fp16_max_abs_err": 0.0, "power_min": 1e9, "power_max": 0.0,
                               "sigpow_min": 1e9, "sigpow_max": 0.0})
    t0 = time.time()
    bar = None
    if progress:
        from tqdm import tqdm
        bar = tqdm(total=N_TOTAL, initial=start, unit="file", smoothing=0.05)
    starts = list(range(start, N_TOTAL, batch))
    chunks = [paths[s:s + batch] for s in starts]
    sub = max(1, batch // max(1, workers))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for s, chunk in zip(starts, chunks):
            parts = list(ex.map(_load_many, [chunk[j:j + sub] for j in range(0, len(chunk), sub)]))
            arr = np.concatenate(parts)                    # (b, 5, 1024, 2) float32
            stats["nonfinite"] += int((~np.isfinite(arr)).sum())
            stats["amax"] = max(stats["amax"], float(np.abs(arr).max()))
            pw = (arr ** 2).sum(axis=-1).mean(axis=(1, 2))
            stats["power_min"] = min(stats["power_min"], float(pw.min()))
            stats["power_max"] = max(stats["power_max"], float(pw.max()))
            ratio = pw / expected_frame_power(labels[s:s + len(arr), COL_SNR].astype(np.float64))
            stats["sigpow_min"] = min(stats.get("sigpow_min", 1e9), float(ratio.min()))
            stats["sigpow_max"] = max(stats.get("sigpow_max", 0.0), float(ratio.max()))
            stored = arr.astype(dtype)
            if dtype == "float16":
                back = stored.astype(np.float32)
                err = np.abs(back - arr)
                stats["fp16_max_abs_err"] = max(stats["fp16_max_abs_err"], float(err.max()))
                big = np.abs(arr) > 1e-2           # sai số tương đối chỉ có nghĩa khi |x| không quá nhỏ
                if big.any():
                    stats["fp16_max_rel_err"] = max(stats["fp16_max_rel_err"],
                                                    float((err[big] / np.abs(arr[big])).max()))
            x[s:s + len(arr)] = stored
            x.flush()
            _write_json(prog_path, {"done": s + len(arr), "dtype": dtype, "stats": stats})
            if bar:
                bar.update(len(arr))
    if bar:
        bar.close()
    del x

    elapsed = time.time() - t0
    meta = {"dtype": dtype, "shape": list(shape), "layout": "(row, M, N, IQ)",
            "row_to_index": "row r <-> dataXXXXXXXX.mat with index r+1",
            "modulations": MODULATIONS, "label_columns": ["mod_idx", "doa_idx", "snr_db"],
            "source_root": os.path.abspath(root), "stats": stats,
            "size_bytes": os.path.getsize(x_path), "convert_seconds_last_run": elapsed,
            "complete": True}
    if stats["amax"] > AMPLITUDE_LIMIT:
        print(f"  ⚠️ biên độ lớn nhất {stats['amax']:.3f} > {AMPLITUDE_LIMIT}")
    lo, hi = SIGNAL_POWER_RANGE
    if "sigpow_min" in stats and not lo <= stats["sigpow_min"] <= stats["sigpow_max"] <= hi:
        print(f"  ⚠️ công suất/(1+10^(-SNR/10)) = {stats['sigpow_min']:.3f}..{stats['sigpow_max']:.3f} "
              f"ngoài {SIGNAL_POWER_RANGE}")
    _write_json(os.path.join(out_dir, "meta.json"), meta)
    return meta
