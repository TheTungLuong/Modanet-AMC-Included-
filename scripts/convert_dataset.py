"""Chuyển 450,120 file .mat -> memmap (chạy 1 lần, ngắt được, chạy tiếp được).

    python scripts/convert_dataset.py --verify-index          # chỉ kiểm, không ghi gì
    python scripts/convert_dataset.py --inspect               # thống kê + kiểm vật lý trên mẫu
    python scripts/convert_dataset.py --dtype float16         # kiểm + convert (9.2 GB)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from configs.data_config import (  # noqa: E402
    DATA_ROOT, PROCESSED_DIR, CONVERT_DTYPES, DEFAULT_CONVERT_DTYPE, CONVERT_BATCH,
    MODULATIONS, N_REP, AMPLITUDE_LIMIT, FRAME_POWER_RANGE, idx_to_angle, expected_frame_power,
)
from src.data.convert import convert, load_mat_sample, scan_dataset, verify_index  # noqa: E402
from src.data.labels import decode_index, encode_index  # noqa: E402
from src.data.physics import c42, estimate_doa_deg, frame_power  # noqa: E402


def print_report(rep):
    print(f"  số file           : {rep['n_files']:,} / {rep['expected']:,}")
    print(f"  chỉ số min / max  : {rep['min_index']} / {rep['max_index']}")
    print(f"  thiếu / trùng     : {rep['missing']:,} / {rep['duplicates']:,}"
          f"  {rep['missing_examples'] if rep['missing'] else ''}")
    print(f"  ngoài phạm vi     : {rep['out_of_range']:,}")
    print(f"  tổ hợp (snr,mod,doa): {rep['combos']:,}  | không đủ {N_REP} file: {rep['combos_not_10']:,}")
    print(f"  đối chiếu tên thư mục: {rep['folder_labels_checked']:,} file, "
          f"lệch {rep['folder_label_mismatch']:,}")
    for e in rep["folder_mismatch_examples"]:
        print(f"    lệch: {e}")


def inspect(found):
    print("\n== INSPECT (mẫu) ==")
    # 1) 10 realization của SNR=3, 4PAM, +1° (tổ hợp mẫu trong MODANET_INSTRUCTION.md)
    idxs = [encode_index(3, MODULATIONS.index("4PAM"), 61, r) for r in range(1, N_REP + 1)]
    X = np.stack([load_mat_sample(found[i]) for i in idxs])
    print(f"  shape mẫu (M,N,IQ): {X.shape[1:]}  dtype {X.dtype}")
    print(f"  min {X.min():+.3f} max {X.max():+.3f} std {X.std():.3f} "
          f"(giới hạn ±{AMPLITUDE_LIMIT})")
    pw = [frame_power(x) for x in X]
    print(f"  công suất frame {min(pw):.3f}..{max(pw):.3f}  (fixture 4PAM@3dB kỳ vọng {FRAME_POWER_RANGE}; "
          f"mô hình 1+10^(-SNR/10) = {expected_frame_power(3):.3f})")
    print("  công suất frame theo SNR (median 20 mẫu, kỳ vọng ≈ 1 + 10^(-SNR/10)):")
    for snr in (-10, -5, 0, 10, 20):
        v = [frame_power(load_mat_sample(found[encode_index(snr, m, 60, 1)])) for m in range(len(MODULATIONS))]
        v += [frame_power(load_mat_sample(found[encode_index(snr, m, 20, 2)])) for m in range(len(MODULATIONS))]
        print(f"    {snr:+3d} dB  {np.median(v):7.3f}   kỳ vọng {expected_frame_power(snr):7.3f}")
    est = [estimate_doa_deg(x) for x in X]
    print(f"  DOA ước lượng từ pha: {np.round(est, 2)}  TB {np.mean(est):+.2f}° (nhãn +1°)")
    C = np.corrcoef(X.reshape(len(X), -1))
    off = C[~np.eye(len(X), dtype=bool)]
    print(f"  tương quan giữa realization: TB {off.mean():+.4f}  max {off.max():+.4f}")
    # 2) C42 từng lớp điều chế ở SNR 20 dB (xác nhận bảng MODULATIONS)
    print("  C42 ở SNR 20 dB, DOA 0° (TB 10 realization) — đường bao hằng ≈ -1:")
    for m, name in enumerate(MODULATIONS):
        v = [c42(load_mat_sample(found[encode_index(20, m, 60, r)])) for r in range(1, N_REP + 1)]
        print(f"    {m:2d} {name:7s} C42 = {np.mean(v):+.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--root", default=str(DATA_ROOT), help="thư mục dataset gốc (.mat)")
    ap.add_argument("--out", default=str(PROCESSED_DIR), help="thư mục ghi memmap")
    ap.add_argument("--dtype", choices=CONVERT_DTYPES, default=DEFAULT_CONVERT_DTYPE)
    ap.add_argument("--batch", type=int, default=CONVERT_BATCH)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--verify-index", action="store_true", help="chỉ kiểm tên file rồi dừng")
    ap.add_argument("--inspect", action="store_true", help="thống kê + kiểm vật lý rồi dừng")
    args = ap.parse_args()

    t = time.time()
    print(f"Quét {args.root} ...")
    found, dup = scan_dataset(args.root)
    ok, rep = verify_index(found, dup)
    print(f"== VERIFY INDEX ({time.time() - t:.1f}s) ==")
    print_report(rep)
    print(f"  => {'ĐẠT' if ok else 'KHÔNG ĐẠT — DỪNG, không convert'}")
    if args.verify_index or not ok:
        sys.exit(0 if ok else 1)
    if args.inspect:
        inspect(found)
        return

    print(f"\n== CONVERT -> {args.out} ({args.dtype}) ==")
    t = time.time()
    meta = convert(args.root, args.out, args.dtype, args.batch, args.workers, found=found)
    s = meta["stats"]
    print(f"  xong trong {time.time() - t:.0f}s | dung lượng {meta['size_bytes'] / 1e9:.2f} GB")
    print(f"  |x| max {s['amax']:.3f} | không hữu hạn {s['nonfinite']} | "
          f"công suất frame {s['power_min']:.3f}..{s['power_max']:.3f} | "
          f"công suất/(1+10^(-SNR/10)) {s.get('sigpow_min', float('nan')):.3f}..{s.get('sigpow_max', float('nan')):.3f}")
    if args.dtype == "float16":
        print(f"  round-trip fp32->fp16->fp32: sai số tuyệt đối max {s['fp16_max_abs_err']:.2e}, "
              f"tương đối max (|x|>1e-2) {s['fp16_max_rel_err']:.2e}")
    with open(os.path.join(args.out, "meta.json")) as f:
        json.load(f)   # kiểm file hợp lệ
    snr, mod, doa, _ = decode_index(1)
    print(f"  hàng 0 = data00000001.mat -> SNR {snr} dB, {MODULATIONS[mod]}, {idx_to_angle(doa)}°")


if __name__ == "__main__":
    main()
