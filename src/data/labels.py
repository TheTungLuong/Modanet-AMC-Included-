"""Giải mã nhãn từ tên file dataXXXXXXXX.mat (chỉ số đếm từ 1).

    index = ((snr_i * N_MOD + mod_i) * N_DOA + doa_i) * N_REP + rep
            snr_i = SNR + 10 (0..30), mod_i = chỉ số trong MODULATIONS (0..11),
            doa_i = góc + 60 (0..120), rep = 1..10

Đã kiểm trên dataset đầy đủ: SNR và DOA giải mã khớp tên thư mục 450,120/450,120;
mod_i khớp tên thư mục khi dùng configs.data_config.MODULATIONS (thứ tự theo chỉ số file).
"""
import os
import re

import numpy as np

from configs.data_config import (
    N_MOD, N_DOA, N_REP, N_TOTAL, SNR_MIN, MODULATIONS, FILENAME_REGEX,
    COL_MOD, COL_DOA, COL_SNR, LABEL_COLUMNS, idx_to_angle,
)

_FILE_RE = re.compile(FILENAME_REGEX)


def decode_index(index1based):
    """tên file -> (snr_dB, mod_idx, doa_idx, rep). Hoạt động cả với mảng numpy."""
    z = index1based - 1
    rep = z % N_REP
    c = z // N_REP
    doa_i = c % N_DOA
    c = c // N_DOA
    mod_i = c % N_MOD
    snr_i = c // N_MOD
    return snr_i + SNR_MIN, mod_i, doa_i, rep + 1


def encode_index(snr_dB, mod_i, doa_i, rep):
    """(snr_dB, mod_idx, doa_idx, rep) -> chỉ số trong tên file (1-based)."""
    return ((snr_dB - SNR_MIN) * N_MOD + mod_i) * N_DOA * N_REP + doa_i * N_REP + rep


def index_from_filename(name):
    """'data00199051.mat' -> 199051; không khớp mẫu -> None."""
    m = _FILE_RE.match(os.path.basename(name))
    return int(m.group(1)) if m else None


def labels_for_all_rows():
    """Nhãn (N_TOTAL, 3) int16 [mod_idx, doa_idx, snr_dB] cho hàng r = index-1.
    Chỉ phụ thuộc công thức -> dùng được khi chưa có dữ liệu (test, chia tập)."""
    snr, mod, doa, _ = decode_index(np.arange(1, N_TOTAL + 1, dtype=np.int64))
    out = np.empty((N_TOTAL, len(LABEL_COLUMNS)), dtype=np.int16)
    out[:, COL_MOD], out[:, COL_DOA], out[:, COL_SNR] = mod, doa, snr
    return out


def parse_folder_labels(path):
    """Đọc nhãn từ cây thư mục .../SNR+03dB/4PAM/+01/dataX.mat -> (snr, mod_name, angle).
    Trả None nếu đường dẫn không theo cấu trúc đó (vd. thư mục phẳng)."""
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) < 4:
        return None
    snr_s, mod_s, doa_s = parts[-4], parts[-3], parts[-2]
    m = re.fullmatch(r"SNR([+-]\d+)dB", snr_s)
    if not m or not re.fullmatch(r"[+-]\d+", doa_s) or mod_s not in MODULATIONS:
        return None
    return int(m.group(1)), mod_s, int(doa_s)


def describe(mod_idx, doa_idx):
    return MODULATIONS[int(mod_idx)], idx_to_angle(int(doa_idx))
