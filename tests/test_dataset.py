"""7.7 — hợp đồng dữ liệu (MODANET_INSTRUCTION.md 6.6), 3 tầng:
  Tầng 1: không cần dữ liệu (luôn chạy)
  Tầng 2: 22 file mẫu thật trong tests/fixtures/
  Tầng 3: dataset đầy đủ / memmap đã convert (tự skip nếu chưa có)
"""
import glob
import os

import numpy as np
import pytest
import scipy.io as sio
import torch

from configs.data_config import (
    DATA_ROOT, FIXTURES_DIR, MODULATIONS, PAPER_MODULATION_ORDER, N_MOD, N_DOA, N_SNR, N_REP,
    N_TOTAL, AMPLITUDE_LIMIT, AMPLITUDE_LIMIT_SNR_GE0, FRAME_POWER_RANGE, SIGNAL_POWER_RANGE,
    PROCESSED_DIR, COL_MOD, COL_DOA, COL_SNR, idx_to_angle, angle_to_idx, expected_frame_power,
)
from src.data.convert import load_mat_raw, load_mat_sample, scan_dataset, verify_index
from src.data.dataset import ChunkShuffleBatchSampler, ModaNetDataset
from src.data.labels import (
    decode_index, encode_index, index_from_filename, labels_for_all_rows, parse_folder_labels,
)
from src.data.physics import c42, estimate_doa_deg, frame_power
from src.data.splits import check_splits, group_ids, make_splits

# =========================================================== TẦNG 1


def test_roundtrip_all_indices_vectorized():
    i = np.arange(1, N_TOTAL + 1)
    s, m, d, r = decode_index(i)
    assert np.array_equal(encode_index(s, m, d, r), i)
    assert s.min() == -10 and s.max() == 20
    assert m.min() == 0 and m.max() == N_MOD - 1
    assert d.min() == 0 and d.max() == N_DOA - 1
    assert r.min() == 1 and r.max() == N_REP


def test_roundtrip_all_combos_scalar():
    n = 0
    for snr in range(-10, 21):
        for mod in range(N_MOD):
            for doa in range(N_DOA):
                for rep in range(1, N_REP + 1):
                    assert decode_index(encode_index(snr, mod, doa, rep)) == (snr, mod, doa, rep)
                    n += 1
    assert n == N_TOTAL == 12 * 121 * 31 * 10


def test_bounds():
    assert decode_index(1) == (-10, 0, 0, 1)
    assert decode_index(N_TOTAL) == (20, 11, 120, 10)
    assert encode_index(3, MODULATIONS.index("4PAM"), angle_to_idx(1), 1) == 199_051


def test_angle_mapping():
    for d in range(N_DOA):
        assert angle_to_idx(idx_to_angle(d)) == d
    assert idx_to_angle(0) == -60 and idx_to_angle(60) == 0 and idx_to_angle(120) == 60


def test_filename_parse():
    assert index_from_filename("data00199051.mat") == 199_051
    assert index_from_filename("/a/b/data00000001.mat") == 1
    assert index_from_filename("myData_info.mat") is None
    assert parse_folder_labels("/x/SNR+03dB/4PAM/+01/data00199051.mat") == (3, "4PAM", 1)
    assert parse_folder_labels("/x/SNR-10dB/PSK/-60/data00002421.mat") == (-10, "PSK", -60)
    assert parse_folder_labels("/flat/data00000001.mat") is None


def test_modulation_order_file_contract():
    """Thứ tự theo chỉ số file (đã kiểm trên dataset đầy đủ), KHÁC thứ tự bài báo liệt kê."""
    assert MODULATIONS[:8] == ["16QAM", "64QAM", "PSK", "QPSK", "8PSK", "QFSK", "8FSK", "16APSK"]
    assert MODULATIONS[8:] == PAPER_MODULATION_ORDER[8:] == ["4PAM", "LFM", "DSB-SC", "SSB-SC"]
    # bằng chứng từ tên thư mục: SNR-10dB/PSK/-60 chứa data00002421..2430
    s, m, d, _ = decode_index(2421)
    assert (s, MODULATIONS[m], idx_to_angle(d)) == (-10, "PSK", -60)


def test_labels_for_all_rows():
    lab = labels_for_all_rows()
    assert lab.shape == (N_TOTAL, 3) and lab.dtype == np.int16
    assert lab[:, COL_MOD].min() == 0 and lab[:, COL_MOD].max() == 11
    assert lab[:, COL_DOA].min() == 0 and lab[:, COL_DOA].max() == 120
    assert lab[:, COL_SNR].min() == -10 and lab[:, COL_SNR].max() == 20
    # hàng r <-> chỉ số r+1
    assert tuple(lab[199_050]) == (8, 61, 3)


def test_fake_permute_and_dtype():
    raw = np.random.randn(10, 1024, 5, 2).astype(np.float32)
    x = np.transpose(raw, (0, 2, 1, 3))           # (1,0,2) trên từng mẫu
    t = torch.from_numpy(np.ascontiguousarray(x))
    assert t.shape == (10, 5, 1024, 2) and t.dtype == torch.float32
    assert torch.equal(t[3, 2, :, 0], torch.from_numpy(raw[3, :, 2, 0]))


@pytest.fixture(scope="module")
def all_labels():
    return labels_for_all_rows()


@pytest.mark.parametrize("mode", ["random", "group"])
def test_splits_contract(all_labels, mode):
    sp = make_splits(all_labels, mode, seed=42)
    check_splits(sp, all_labels, mode)
    n = len(all_labels)
    assert abs(sp["test"].size / n - 0.2) < 0.005
    assert abs((sp["train"].size + sp["val"].size) / n - 0.8) < 0.005
    assert abs(sp["val"].size / (sp["train"].size + sp["val"].size) - 0.1) < 0.005
    tr = all_labels[sp["train"]]
    assert np.unique(tr[:, COL_MOD]).size == 12
    assert np.unique(tr[:, COL_DOA]).size == 121
    assert np.unique(tr[:, COL_SNR]).size == 31
    # tái lập
    sp2 = make_splits(all_labels, mode, seed=42)
    assert all(np.array_equal(sp[k], sp2[k]) for k in sp)


def test_group_mode_keeps_realizations_together(all_labels):
    sp = make_splits(all_labels, "group", seed=42)
    g = group_ids(all_labels)
    for k in ("train", "val", "test"):
        _, counts = np.unique(g[sp[k]], return_counts=True)
        assert (counts == N_REP).all()                     # cả 10 realization cùng phía
    assert np.intersect1d(g[sp["train"]], g[sp["test"]]).size == 0


def test_random_mode_mixes_realizations(all_labels):
    """Chế độ random (như bài): hầu hết tổ hợp có realization ở cả train và test."""
    sp = make_splits(all_labels, "random", seed=42)
    g = group_ids(all_labels)
    shared = np.intersect1d(g[sp["train"]], g[sp["test"]]).size
    assert shared / (N_MOD * N_DOA * N_SNR) > 0.8


@pytest.mark.parametrize("chunk,group", [(1, 1), (64, 64), (7, 3)])
def test_sampler_covers_each_index_once(chunk, group):
    s = ChunkShuffleBatchSampler(10_000, 128, chunk, group, shuffle=True, seed=3)
    s.set_epoch(0)
    e0 = np.concatenate(list(s))
    assert np.array_equal(np.sort(e0), np.arange(10_000))
    assert len(list(s)) == len(s)
    s.set_epoch(1)
    assert not np.array_equal(e0, np.concatenate(list(s)))


def test_chunk_sampler_batches_are_diverse():
    """Mỗi batch 128 lấy từ nhiều vị trí khác nhau trên đĩa (gần iid)."""
    s = ChunkShuffleBatchSampler(360_000, 128, 64, 64, shuffle=True, seed=0)
    n_chunks = [np.unique(b // 64).size for b in list(s)[:200]]
    assert np.mean(n_chunks) > 50


def test_dataset_on_fake_memmap(tmp_path):
    x = np.lib.format.open_memmap(tmp_path / "x.npy", mode="w+", dtype=np.float16,
                                  shape=(50, 5, 1024, 2))
    x[:] = np.random.randn(50, 5, 1024, 2).astype(np.float16)
    lab = labels_for_all_rows()[:50]
    rows = np.array([3, 7, 20, 41])
    ds = ModaNetDataset(np.load(tmp_path / "x.npy", mmap_mode="r"), lab, rows)
    xb, ym, yd, snr = ds[1]
    assert xb.shape == (5, 1024, 2) and xb.dtype == torch.float32
    assert torch.equal(xb, torch.from_numpy(x[7].astype(np.float32)))
    xb, ym, yd, snr = ds[np.array([2, 0])]
    assert xb.shape == (2, 5, 1024, 2)
    assert torch.equal(xb[0], torch.from_numpy(x[3].astype(np.float32)))   # đã sắp tăng
    assert ym.tolist() == lab[[3, 20], COL_MOD].tolist()


def test_dataset_pickle_does_not_copy_memmap(tmp_path):
    """Windows/macOS/Python>=3.14 dùng spawn/forkserver: pickle dataset KHÔNG được mang theo dữ liệu."""
    import pickle
    x = np.lib.format.open_memmap(tmp_path / "x.npy", mode="w+", dtype=np.float16,
                                  shape=(2000, 5, 1024, 2))
    x[:] = 1
    x.flush()
    mm = np.load(tmp_path / "x.npy", mmap_mode="r")
    ds = ModaNetDataset(mm, labels_for_all_rows()[:2000], np.arange(2000))
    blob = pickle.dumps(ds)
    assert len(blob) < mm.nbytes / 50            # 41 MB dữ liệu -> pickle chỉ vài chục KB
    ds2 = pickle.loads(blob)
    assert isinstance(ds2.x, np.memmap)
    assert torch.equal(ds[np.arange(5)][0], ds2[np.arange(5)][0])


@pytest.mark.parametrize("ctx", ["spawn", "forkserver"])
def test_dataloader_with_spawn_workers(tmp_path, ctx):
    import multiprocessing as mp
    if ctx not in mp.get_all_start_methods():
        pytest.skip(f"{ctx} không có trên nền tảng này")
    x = np.lib.format.open_memmap(tmp_path / "x.npy", mode="w+", dtype=np.float16,
                                  shape=(300, 5, 1024, 2))
    x[:] = np.arange(300, dtype=np.float16)[:, None, None, None]
    x.flush()
    ds = ModaNetDataset(np.load(tmp_path / "x.npy", mmap_mode="r"), labels_for_all_rows()[:300],
                        np.arange(300))
    sampler = ChunkShuffleBatchSampler(300, 64, 8, 4, shuffle=True, seed=0)
    ld = torch.utils.data.DataLoader(ds, sampler=sampler, batch_size=None, num_workers=2,
                                     multiprocessing_context=ctx)
    seen = torch.cat([b[0][:, 0, 0, 0] for b in ld])
    assert sorted(seen.long().tolist()) == list(range(300))


def test_dataloader_in_memory_fork_workers():
    """--in-memory trên Linux: worker fork đọc mảng RAM dùng chung, batch giống hệt 0 worker."""
    import multiprocessing as mp
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("fork không có trên nền tảng này")
    x = np.arange(300, dtype=np.float16)[:, None, None, None] * np.ones((1, 5, 1024, 2), np.float16)
    ds = ModaNetDataset(x, labels_for_all_rows()[:300], np.arange(300))
    sampler = ChunkShuffleBatchSampler(300, 64, 8, 4, shuffle=True, seed=0)
    ref = [b[0] for b in torch.utils.data.DataLoader(ds, sampler=sampler, batch_size=None)]
    got = [b[0] for b in torch.utils.data.DataLoader(ds, sampler=sampler, batch_size=None,
                                                     num_workers=2, multiprocessing_context="fork")]
    assert len(got) == len(ref) and all(torch.equal(a, b) for a, b in zip(got, ref))


# =========================================================== TẦNG 2 (fixtures)

FIX_4PAM = sorted(glob.glob(str(FIXTURES_DIR / "SNR+03dB" / "4PAM" / "+01" / "*.mat")))
FIX_MODS = sorted(glob.glob(str(FIXTURES_DIR / "SNR+20dB" / "*" / "+00" / "*.mat")))
need_fixtures = pytest.mark.skipif(len(FIX_4PAM) != 10 or len(FIX_MODS) != 12,
                                   reason="thiếu file mẫu trong tests/fixtures")


@need_fixtures
def test_fixture_format():
    for p in FIX_4PAM + FIX_MODS:
        d = sio.loadmat(p)
        assert [k for k in d if not k.startswith("__")] == ["data"]
        assert d["data"].shape == (1024, 5, 2) and d["data"].dtype == np.float32
        s = load_mat_sample(p)
        assert s.shape == (5, 1024, 2) and s.dtype == np.float32
        assert np.isfinite(s).all() and np.abs(s).max() < AMPLITUDE_LIMIT_SNR_GE0   # fixtures SNR >= 3 dB


@need_fixtures
def test_fixture_labels_from_filename():
    reps = []
    for p in FIX_4PAM:
        s, m, d, r = decode_index(index_from_filename(p))
        assert (s, MODULATIONS[m], d, idx_to_angle(d)) == (3, "4PAM", 61, 1)
        reps.append(r)
    assert sorted(reps) == list(range(1, 11))


@need_fixtures
def test_fixture_filename_matches_folder():
    for p in FIX_4PAM + FIX_MODS:
        s, m, d, _ = decode_index(index_from_filename(p))
        assert parse_folder_labels(p) == (s, MODULATIONS[m], idx_to_angle(d)), p


@need_fixtures
def test_fixture_frame_power():
    for p in FIX_4PAM:
        assert FRAME_POWER_RANGE[0] <= frame_power(load_mat_sample(p)) <= FRAME_POWER_RANGE[1]
    lo, hi = SIGNAL_POWER_RANGE
    for p in FIX_4PAM + FIX_MODS:
        snr = decode_index(index_from_filename(p))[0]
        assert lo <= frame_power(load_mat_sample(p)) / expected_frame_power(snr) <= hi, p


@need_fixtures
def test_physics_doa_estimate():
    """Ước lượng góc từ pha qua 5 anten ∈ [-1°, +3°] với nhãn +1°. Bắt lỗi permute sai trục."""
    est = np.array([estimate_doa_deg(load_mat_sample(p)) for p in FIX_4PAM])
    print(f"\n  DOA ước lượng: {np.round(est, 2)}  TB {est.mean():+.2f}° (nhãn +1°)")
    assert ((est >= -1) & (est <= 3)).all()
    assert abs(est.mean() - 1) < 1.0


@need_fixtures
def test_wrong_permute_is_detectable():
    """Nếu dùng nhầm (2,0,1) theo mô tả bài báo -> 'anten' chỉ còn 2 kênh, shape sai ngay."""
    raw = load_mat_raw(FIX_4PAM[0])
    wrong = np.transpose(raw, (2, 0, 1))
    assert wrong.shape == (2, 1024, 5) != (5, 1024, 2)


@need_fixtures
def test_physics_modulation_names():
    """C42 (bất biến pha): nhóm đường bao hằng ≈ -1. Xác nhận tên thư mục = tên lớp,
    và bác bỏ việc dùng thứ tự bài báo làm chỉ số lớp."""
    const_env = {"PSK", "QPSK", "8PSK", "QFSK", "8FSK", "SSB-SC"}
    got = {}
    for p in FIX_MODS:
        m = decode_index(index_from_filename(p))[1]
        got[MODULATIONS[m]] = c42(load_mat_sample(p))
    print("\n  C42 @20dB: " + ", ".join(f"{k} {v:+.2f}" for k, v in got.items()))
    assert {k for k, v in got.items() if v < -0.9} == const_env
    assert -0.80 < got["16QAM"] < -0.60 and -0.75 < got["64QAM"] < -0.50
    # theo thứ tự bài báo, chỉ số 0 sẽ là "PSK" (đường bao hằng) — thực tế lớp 0 không phải
    assert PAPER_MODULATION_ORDER[0] in const_env and got[MODULATIONS[0]] > -0.9


@need_fixtures
def test_fp16_roundtrip_error():
    x = np.stack([load_mat_sample(p) for p in FIX_4PAM + FIX_MODS])
    back = x.astype(np.float16).astype(np.float32)
    err = np.abs(back - x)
    big = np.abs(x) > 1e-2
    rel = (err[big] / np.abs(x[big])).max()
    print(f"\n  fp32->fp16->fp32: abs max {err.max():.2e}, rel max {rel:.2e} (22 file thật)")
    assert err.max() < 2e-3 and rel < 1e-3


# =========================================================== TẦNG 3 (dataset đầy đủ)

has_full = DATA_ROOT.exists() and any(DATA_ROOT.glob("SNR*"))
has_proc = (PROCESSED_DIR / "meta.json").exists()


@pytest.mark.fulldata
@pytest.mark.skipif(not has_full, reason=f"không có dataset đầy đủ tại {DATA_ROOT}")
def test_full_dataset_index():
    found, dup = scan_dataset(str(DATA_ROOT))
    ok, rep = verify_index(found, dup)
    print(f"\n  {rep}")
    assert rep["n_files"] == N_TOTAL and rep["missing"] == 0 and rep["duplicates"] == 0
    assert rep["min_index"] == 1 and rep["max_index"] == N_TOTAL
    assert rep["combos_not_10"] == 0
    assert rep["folder_label_mismatch"] == 0 and rep["folder_labels_checked"] == N_TOTAL
    assert ok


@pytest.mark.fulldata
@pytest.mark.skipif(not (has_full and has_proc), reason="cần cả dataset .mat gốc và memmap đã convert")
def test_processed_memmap_matches_mat():
    import json
    meta = json.loads((PROCESSED_DIR / "meta.json").read_text())
    assert meta["complete"] and meta["modulations"] == MODULATIONS
    x = np.load(PROCESSED_DIR / f"x_{meta['dtype']}.npy", mmap_mode="r")
    lab = np.load(PROCESSED_DIR / "labels.npy")
    assert x.shape == (N_TOTAL, 5, 1024, 2)
    assert np.array_equal(lab, labels_for_all_rows())
    found, _ = scan_dataset(str(DATA_ROOT))
    rng = np.random.default_rng(0)
    errs = []
    for r in np.concatenate([[0, N_TOTAL - 1, 199_050], rng.integers(0, N_TOTAL, 30)]):
        ref = load_mat_sample(found[int(r) + 1])
        assert np.array_equal(x[r], ref.astype(meta["dtype"])), r
        errs.append(np.abs(x[r].astype(np.float32) - ref).max())
    s = meta["stats"]
    print(f"\n  memmap {meta['dtype']}: sai số abs max trên 33 mẫu {max(errs):.2e}; "
          f"toàn bộ: abs {s.get('fp16_max_abs_err', 0):.2e}, rel {s.get('fp16_max_rel_err', 0):.2e}; "
          f"|x|max {s['amax']:.3f}; công suất {s['power_min']:.3f}..{s['power_max']:.3f}; "
          f"không hữu hạn {s['nonfinite']}")
    assert s["nonfinite"] == 0 and s["amax"] < AMPLITUDE_LIMIT
    if meta["dtype"] == "float16":
        assert s["fp16_max_rel_err"] < 1e-3


@pytest.mark.fulldata
@pytest.mark.skipif(not has_proc, reason="chưa convert memmap")
def test_processed_power_follows_snr():
    """Công suất frame ≈ 1 + 10^(-SNR/10): tín hiệu công suất đơn vị + nhiễu (NORMALIZE=False)."""
    import json
    meta = json.loads((PROCESSED_DIR / "meta.json").read_text())
    x = np.load(PROCESSED_DIR / f"x_{meta['dtype']}.npy", mmap_mode="r")
    lab = np.load(PROCESSED_DIR / "labels.npy")
    rows = np.sort(np.random.default_rng(0).choice(N_TOTAL, 5000, replace=False))
    X = np.asarray(x[rows], dtype=np.float32)
    snr = lab[rows, COL_SNR].astype(np.float64)
    ratio = (X ** 2).sum(-1).mean((1, 2)) / expected_frame_power(snr)
    amax = np.abs(X).max((1, 2, 3))
    print(f"\n  công suất/(1+10^(-SNR/10)) trên 5000 mẫu: min {ratio.min():.3f} median "
          f"{np.median(ratio):.3f} max {ratio.max():.3f}; |x| max @SNR>=0: {amax[snr >= 0].max():.2f}")
    assert SIGNAL_POWER_RANGE[0] <= ratio.min() and ratio.max() <= SIGNAL_POWER_RANGE[1]
    assert 0.9 < np.median(ratio) < 1.1
    assert amax[snr >= 0].max() < AMPLITUDE_LIMIT_SNR_GE0
