"""Chia train/val/test, lưu index ra file MỘT LẦN để mọi lần chạy giống nhau.

random (mặc định, như bài báo): hoán vị ngẫu nhiên từng mẫu, 20% test; val = 10% phần train.
group: đơn vị chia là tổ hợp (mod, doa, snr) — 10 realization luôn nằm cùng một phía
       (cả test lẫn val đều là tổ hợp chưa gặp). Khắt khe hơn -> số liệu sẽ thấp hơn bài báo.
"""
import os

import numpy as np

from configs.data_config import (
    TEST_FRACTION, VAL_FRACTION_OF_TRAIN, SPLIT_SEED, SPLIT_MODES,
    COL_MOD, COL_DOA, COL_SNR, N_MOD, N_DOA, SNR_MIN,
)


def group_ids(labels):
    """id tổ hợp (snr, mod, doa) cho mỗi hàng."""
    snr_i = labels[:, COL_SNR].astype(np.int64) - SNR_MIN
    return (snr_i * N_MOD + labels[:, COL_MOD]) * N_DOA + labels[:, COL_DOA]


def _split_units(n_units, rng, test_frac, val_frac):
    perm = rng.permutation(n_units)
    n_test = int(round(n_units * test_frac))
    test, trainval = perm[:n_test], perm[n_test:]
    n_val = int(round(trainval.size * val_frac))
    return trainval[n_val:], trainval[:n_val], test


def make_splits(labels, mode="random", seed=SPLIT_SEED,
                test_frac=TEST_FRACTION, val_frac=VAL_FRACTION_OF_TRAIN):
    if mode not in SPLIT_MODES:
        raise ValueError(f"split mode {mode!r} không thuộc {SPLIT_MODES}")
    rng = np.random.default_rng(seed)
    if mode == "random":
        tr, va, te = _split_units(len(labels), rng, test_frac, val_frac)
    else:
        g = group_ids(labels)
        uniq, inv = np.unique(g, return_inverse=True)
        gtr, gva, gte = _split_units(uniq.size, rng, test_frac, val_frac)
        side = np.empty(uniq.size, dtype=np.int8)
        side[gtr], side[gva], side[gte] = 0, 1, 2
        s = side[inv]
        rows = np.arange(len(labels))
        tr, va, te = rows[s == 0], rows[s == 1], rows[s == 2]
    return {k: np.sort(v).astype(np.int64) for k, v in
            (("train", tr), ("val", va), ("test", te))}


def split_path(out_dir, mode, seed):
    return os.path.join(out_dir, "splits", f"{mode}_seed{seed}.npz")


def load_or_make_splits(labels, out_dir, mode="random", seed=SPLIT_SEED):
    """Nạp file split nếu có, nếu chưa thì tạo và lưu (không bao giờ ghi đè)."""
    p = split_path(out_dir, mode, seed)
    if os.path.exists(p):
        z = np.load(p)
        return {k: z[k] for k in ("train", "val", "test")}
    sp = make_splits(labels, mode, seed)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.savez(p, **sp)
    return sp


def check_splits(sp, labels, mode):
    """Các bất biến của hợp đồng dữ liệu (dùng trong test và lúc train)."""
    tr, va, te = sp["train"], sp["val"], sp["test"]
    assert np.intersect1d(tr, te).size == 0, "train ∩ test ≠ ∅"
    assert np.intersect1d(tr, va).size == 0, "train ∩ val ≠ ∅"
    assert np.intersect1d(va, te).size == 0, "val ∩ test ≠ ∅"
    assert tr.size + va.size + te.size == len(labels)
    if mode == "group":
        g = group_ids(labels)
        assert np.intersect1d(g[tr], g[te]).size == 0, "tổ hợp nằm cả train lẫn test"
        assert np.intersect1d(g[tr], g[va]).size == 0, "tổ hợp nằm cả train lẫn val"
    return True
