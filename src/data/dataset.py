"""Dataset đọc memmap (lazy) + sampler xáo trộn theo khối cho HDD.

Dùng theo kiểu "batched fetch": DataLoader(ds, sampler=<batch sampler>, batch_size=None)
-> ds[array_of_rows] trả nguyên 1 batch bằng MỘT lần fancy-index memmap (đã sắp chỉ số
để đọc gần tuần tự), thay vì 128 lần __getitem__ lẻ.
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from configs.data_config import COL_MOD, COL_DOA, COL_SNR, MODULATIONS


def open_processed(processed_dir, dtype=None):
    """Mở memmap đã convert. Trả (x_memmap, labels, meta). Kiểm đã convert XONG."""
    meta_p = os.path.join(processed_dir, "meta.json")
    if not os.path.exists(meta_p):
        raise FileNotFoundError(
            f"Chưa có dữ liệu đã convert trong {processed_dir}. "
            f"Chạy: python scripts/convert_dataset.py --out {processed_dir}")
    with open(meta_p) as f:
        meta = json.load(f)
    if not meta.get("complete"):
        raise RuntimeError("convert chưa xong — chạy lại convert_dataset.py để tiếp tục")
    if meta["modulations"] != MODULATIONS:
        raise RuntimeError("thứ tự MODULATIONS trong meta.json khác configs/data_config.py")
    dtype = dtype or meta["dtype"]
    x = np.load(os.path.join(processed_dir, f"x_{dtype}.npy"), mmap_mode="r")
    labels = np.load(os.path.join(processed_dir, "labels.npy"))
    return x, labels, meta


def load_into_memory(x_memmap):
    """Đọc toàn bộ memmap vào RAM (đọc tuần tự). Dùng khi RAM đủ lớn (server)."""
    return np.array(x_memmap)


class ModaNetDataset(Dataset):
    """rows: chỉ số hàng trong memmap thuộc tập này.
    ds[i] (int, vị trí trong tập) -> (x (5,1024,2) f32, mod, doa, snr)
    ds[array vị trí] -> batch tensor.

    x: np.memmap (mặc định) hoặc ndarray trong RAM (--in-memory).
    An toàn khi DataLoader dùng 'spawn'/'forkserver' (Windows, macOS, Python >= 3.14): pickle
    chỉ mang đường dẫn + vị trí memmap, worker tự mở lại — KHÔNG chép 9 GB dữ liệu sang mỗi worker.
    (ndarray trong RAM thì vẫn bị chép -> train.py dùng num_workers=0 với --in-memory.)"""

    def __init__(self, x, labels, rows):
        self.x = x
        self.rows = np.asarray(rows, dtype=np.int64)
        self.y_mod = torch.from_numpy(labels[self.rows, COL_MOD].astype(np.int64))
        self.y_doa = torch.from_numpy(labels[self.rows, COL_DOA].astype(np.int64))
        self.snr = torch.from_numpy(labels[self.rows, COL_SNR].astype(np.int64))

    def __getstate__(self):
        st = self.__dict__.copy()
        x = st["x"]
        if isinstance(x, np.memmap) and x.filename:
            st["x"] = None
            st["_memmap"] = (os.fspath(x.filename), x.dtype.str, x.shape, x.offset)
        return st

    def __setstate__(self, st):
        mm = st.pop("_memmap", None)
        self.__dict__.update(st)
        if mm is not None:
            fn, dt, shape, offset = mm
            self.x = np.memmap(fn, dtype=np.dtype(dt), mode="r", shape=tuple(shape), offset=offset)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, pos):
        if np.isscalar(pos):
            xb = torch.from_numpy(np.asarray(self.x[self.rows[pos]], dtype=np.float32))
            return xb, self.y_mod[pos], self.y_doa[pos], self.snr[pos]
        pos = np.sort(np.asarray(pos, dtype=np.int64))       # tăng dần -> đọc đĩa tuần tự
        xb = torch.from_numpy(np.asarray(self.x[self.rows[pos]], dtype=np.float32))
        t = torch.from_numpy(pos)
        return xb, self.y_mod[t], self.y_doa[t], self.snr[t]


class ChunkShuffleBatchSampler(Sampler):
    """Sinh batch vị trí. chunk_size=1 -> hoán vị ngẫu nhiên thuần.

    Vì rows đã sắp tăng, các vị trí liền nhau cũng liền nhau trên đĩa. Mỗi epoch:
    xáo thứ tự các khối chunk_size vị trí, gom group_chunks khối thành nhóm, xáo trộn
    mẫu trong nhóm rồi cắt batch. set_epoch(e) để mỗi epoch một thứ tự khác (tái lập được)."""

    def __init__(self, n, batch_size, chunk_size=1, group_chunks=1, shuffle=True,
                 drop_last=False, seed=0):
        self.n, self.bs = n, batch_size
        self.chunk, self.group = max(1, chunk_size), max(1, group_chunks)
        self.shuffle, self.drop_last, self.seed = shuffle, drop_last, seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _order(self):
        if not self.shuffle:
            return np.arange(self.n)
        rng = np.random.default_rng((self.seed, self.epoch))
        if self.chunk == 1:
            return rng.permutation(self.n)
        starts = rng.permutation(np.arange(0, self.n, self.chunk))
        out = []
        for g in range(0, len(starts), self.group):
            pos = np.concatenate([np.arange(s, min(s + self.chunk, self.n))
                                  for s in starts[g:g + self.group]])
            out.append(rng.permutation(pos))
        return np.concatenate(out)

    def __iter__(self):
        order = self._order()
        end = (self.n // self.bs) * self.bs if self.drop_last else self.n
        for i in range(0, end, self.bs):
            yield order[i:i + self.bs]

    def __len__(self):
        return self.n // self.bs if self.drop_last else -(-self.n // self.bs)


def make_loader(ds, batch_size, shuffle, chunk_size=1, group_chunks=1, seed=0,
                num_workers=4, pin_memory=True, prefetch_factor=4, drop_last=False):
    sampler = ChunkShuffleBatchSampler(len(ds), batch_size, chunk_size, group_chunks,
                                       shuffle, drop_last, seed)
    kw = {}
    if num_workers > 0:
        kw = dict(persistent_workers=True, prefetch_factor=prefetch_factor)
    loader = torch.utils.data.DataLoader(ds, sampler=sampler, batch_size=None,
                                         num_workers=num_workers, pin_memory=pin_memory, **kw)
    return loader, sampler
