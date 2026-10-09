"""Suy luận MoDANet — softmax CHỈ áp ở đây.

    python scripts/inference.py --ckpt outputs/run01/best.pt --input sample.npy
    python scripts/inference.py --ckpt outputs/run01/best.pt --input path/to/data00199051.mat
    python scripts/inference.py --ckpt outputs/run01/best.pt --input thu_muc_chua_mat/
    python scripts/inference.py --ckpt outputs/run01/best.pt --from-split test --num 16
    python scripts/inference.py --ckpt outputs/run01/best.pt --benchmark 1000   # ms/frame

.npy nhận (1024,5,2) thô, (5,1024,2), hoặc batch (B,1024,5,2) / (B,5,1024,2).
"""
import argparse
import glob
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from configs.data_config import (  # noqa: E402
    PROCESSED_DIR, RAW_SHAPE, SAMPLE_SHAPE, PERMUTE_RAW_TO_TORCH, SPLIT_SEED, idx_to_angle,
)
from src.data.convert import load_mat_sample  # noqa: E402
from src.data.labels import decode_index, index_from_filename  # noqa: E402
from src.utils.checkpoint import build_model_from_checkpoint, load_checkpoint  # noqa: E402
from src.utils.device import select_device, synchronize  # noqa: E402


def to_batch(a):
    """Đưa mảng về (B, 5, 1024, 2) float32."""
    a = np.asarray(a, dtype=np.float32)
    if a.shape == RAW_SHAPE:
        a = a[None]
    elif a.shape == SAMPLE_SHAPE:
        return a[None]
    if a.ndim == 4 and a.shape[1:] == RAW_SHAPE:
        return np.ascontiguousarray(np.transpose(a, (0,) + tuple(p + 1 for p in PERMUTE_RAW_TO_TORCH)))
    if a.ndim == 4 and a.shape[1:] == SAMPLE_SHAPE:
        return a
    raise ValueError(f"shape {a.shape} không hỗ trợ (cần {RAW_SHAPE} hoặc {SAMPLE_SHAPE}, có/không trục batch)")


def load_inputs(path):
    """Trả (x (B,5,1024,2), tên, nhãn thật nếu biết [(mod, doa)] hoặc None)."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "**", "data*.mat"), recursive=True))
    elif path.endswith(".mat"):
        files = [path]
    else:
        x = to_batch(np.load(path))
        return x, [f"{os.path.basename(path)}[{i}]" for i in range(len(x))], None
    if not files:
        sys.exit(f"Không tìm thấy file .mat trong {path}")
    x = np.stack([load_mat_sample(f) for f in files])
    truth = []
    for f in files:
        i = index_from_filename(f)
        truth.append(None if i is None else decode_index(i)[1:3])
    return x, [os.path.basename(f) for f in files], truth


@torch.no_grad()
def predict(model, x, device, batch=256):
    pa, pd = [], []
    for i in range(0, len(x), batch):
        xb = torch.from_numpy(x[i:i + batch]).to(device)
        la, ld = model(xb)
        pa.append(torch.softmax(la, 1).cpu())
        pd.append(torch.softmax(ld, 1).cpu())
    return torch.cat(pa).numpy(), torch.cat(pd).numpy()


@torch.no_grad()
def benchmark(model, device, n_frames, batch):
    x = torch.randn((batch,) + SAMPLE_SHAPE, device=device)
    for _ in range(5):
        model(x)
    synchronize(device)
    reps = max(1, n_frames // batch)
    t = time.perf_counter()
    for _ in range(reps):
        model(x)
    synchronize(device)
    return (time.perf_counter() - t) / (reps * batch) * 1000


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", help=".npy / .mat / thư mục chứa .mat")
    ap.add_argument("--from-split", choices=["train", "val", "test"], help="lấy mẫu từ memmap")
    ap.add_argument("--num", type=int, default=16)
    ap.add_argument("--data-dir", default=str(PROCESSED_DIR))
    ap.add_argument("--device", default="auto", help="auto | cpu | mps | cuda | cuda:N")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--benchmark", type=int, default=0, help="đo ms/frame trên N frame ngẫu nhiên")
    ap.add_argument("--bench-batch", type=int, default=1, help="batch khi đo (1 = độ trễ 1 frame)")
    args = ap.parse_args()

    device = select_device(args.device, verbose=False)
    ckpt = load_checkpoint(args.ckpt)
    model = build_model_from_checkpoint(ckpt).to(device).eval()
    mods = ckpt["config_snapshot"]["modulations"]
    print(f"Checkpoint {args.ckpt}: epoch {ckpt['epoch']}, {ckpt['config_snapshot']['model']}, "
          f"device={device.type}")

    if args.benchmark:
        ms = benchmark(model, device, args.benchmark, args.bench_batch)
        print(f"Thời gian suy luận: {ms:.3f} ms/frame (batch {args.bench_batch}, {device.type}) "
              f"— bài báo Fig.6b ~1.9 ms/frame")
        if not (args.input or args.from_split):
            return

    if args.from_split:
        from src.data.dataset import open_processed
        from src.data.splits import load_or_make_splits
        x_mm, labels, _ = open_processed(args.data_dir)
        mode = ckpt["config_snapshot"]["args"].get("split_mode", "random")
        rows = load_or_make_splits(labels, args.data_dir, mode, SPLIT_SEED)[args.from_split]
        rows = np.sort(np.random.default_rng(0).choice(rows, min(args.num, rows.size), replace=False))
        x = np.asarray(x_mm[rows], dtype=np.float32)
        names = [f"data{r + 1:08d}.mat" for r in rows]
        truth = [tuple(labels[r, :2]) for r in rows]
    elif args.input:
        x, names, truth = load_inputs(args.input)
    else:
        sys.exit("Cần --input, --from-split hoặc --benchmark")

    t = time.perf_counter()
    pa, pd = predict(model, x, device, args.batch)
    dt = (time.perf_counter() - t) / len(x) * 1000
    n_ok_m = n_ok_d = n_known = 0
    for i, name in enumerate(names):
        m, d = int(pa[i].argmax()), int(pd[i].argmax())
        top = np.argsort(-pa[i])[:args.topk]
        alt = ", ".join(f"{mods[j]} {pa[i, j] * 100:.1f}%" for j in top[1:])
        line = (f"{name:22s} mod: {mods[m]:7s} ({pa[i, m] * 100:5.1f}%)  "
                f"DOA: {idx_to_angle(d):+4d}° ({pd[i, d] * 100:5.1f}%)")
        if truth and truth[i] is not None:
            tm, td = int(truth[i][0]), int(truth[i][1])
            n_known += 1
            n_ok_m += m == tm
            n_ok_d += d == td
            line += f"  | thật: {mods[tm]:7s} {idx_to_angle(td):+4d}°"
        print(line + (f"   [khác: {alt}]" if args.topk > 1 else ""))
    print(f"{len(x)} frame, {dt:.3f} ms/frame (gồm chuyển dữ liệu)")
    if n_known:
        print(f"Đúng: mod {n_ok_m}/{n_known}, DOA {n_ok_d}/{n_known}")


if __name__ == "__main__":
    main()
