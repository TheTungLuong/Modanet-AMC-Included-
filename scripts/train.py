"""Huấn luyện MoDANet.

    python scripts/train.py --preset paper --batch-size 64 --accum-steps 2 --out outputs/run01
    python scripts/train.py --resume outputs/run01/last.pt        # chống crash / mất điện
    python scripts/train.py --smoke --subset 2000 --epochs 2       # kiểm pipeline, ~1 phút
    python scripts/train.py --split-mode group --out outputs/run02 # phép đo khắt khe hơn
    python scripts/train.py --in-memory                            # server nhiều RAM: nạp 9.2 GB vào RAM

Chạy được trên mọi máy: --device auto (CUDA -> Apple MPS -> CPU), --batch-size auto (micro-batch theo
VRAM, batch hiệu dụng luôn 128), --num-workers auto (theo số CPU và /dev/shm). Resume được trên máy
khác (vd. GTX 1050 -> server): micro-batch tính lại theo GPU mới, batch hiệu dụng giữ nguyên.
"""
import argparse
import csv
import math
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from configs import train_config as TC  # noqa: E402
from configs.data_config import (  # noqa: E402
    PROCESSED_DIR, PROJECT_ROOT, SPLIT_MODES, SPLIT_SEED, VAL_FRACTION_OF_TRAIN, MODULATIONS,
    SAMPLE_SHAPE,
)
from configs.model_config import (  # noqa: E402
    PRESETS, DEFAULT_PRESET, FILTER_SIZE_CHOICES, VARIANT_SHARED_BLOCKS,
    LOSS_WEIGHT_AMC, LOSS_WEIGHT_DOA,
)
from src.data.dataset import (  # noqa: E402
    ModaNetDataset, load_into_memory, make_loader, open_processed,
)
from src.data.splits import check_splits, load_or_make_splits, split_path  # noqa: E402
from src.models.modanet import MoDANet, MultiTaskLoss, count_params  # noqa: E402
from src.utils.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from src.utils.device import (  # noqa: E402
    auto_micro_batch, mem_available_bytes, resolve_num_workers, select_device,
)
from src.utils.metrics import rmse_deg  # noqa: E402
from src.utils.seed import get_rng_state, seed_everything, set_rng_state  # noqa: E402

CSV_FIELDS = ["epoch", "lr", "train_loss", "train_loss_amc", "train_loss_doa",
              "val_loss", "val_loss_amc", "val_loss_doa", "val_acc_amc", "val_acc_doa",
              "val_rmse_deg", "gap", "epoch_time"]
# tham số được phép đổi khi --resume (còn lại lấy từ checkpoint để tái lập đúng lần chạy)
RUNTIME_ARGS = ("max_hours", "num_workers", "device", "data_dir", "epochs", "patience",
                "no_early_stop", "log_every", "batch_size", "accum_steps", "in_memory")
DEVICE_RE = r"auto|cpu|mps|cuda(:\d+)?"


def _batch_arg(v):
    if v == "auto":
        return v
    try:
        n = int(v)
    except ValueError:
        raise argparse.ArgumentTypeError("--batch-size phải là số nguyên hoặc 'auto'")
    if n < 1:
        raise argparse.ArgumentTypeError("--batch-size phải >= 1")
    return n


def _workers_arg(v):
    return v if v == "auto" else int(v)


def _device_arg(v):
    import re
    if not re.fullmatch(DEVICE_RE, v):
        raise argparse.ArgumentTypeError("--device: auto | cpu | mps | cuda | cuda:N")
    return v


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--preset", choices=list(PRESETS), default=DEFAULT_PRESET)
    ap.add_argument("--filters", type=int, default=None, help="ghi đè K của preset")
    ap.add_argument("--filter-size", type=int, choices=FILTER_SIZE_CHOICES, default=None)
    ap.add_argument("--variant", choices=list(VARIANT_SHARED_BLOCKS), default=None)
    ap.add_argument("--batch-size", type=_batch_arg, default=None,
                    help="micro-batch (số) hoặc 'auto' (mặc định: theo VRAM, instruction 9.3)")
    ap.add_argument("--accum-steps", type=int, default=None,
                    help=f"gradient accumulation; mặc định: 'auto' -> {TC.BATCH_SIZE} // micro-batch, "
                         f"số -> 1. Batch hiệu dụng = batch-size × accum-steps")
    ap.add_argument("--lr", type=float, default=None,
                    help=f"mặc định {TC.LR} × (batch hiệu dụng / {TC.BATCH_SIZE}) — quy tắc tuyến tính")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", default=None,
                    help="thư mục run (mặc định <project>/outputs/run01, smoke: <project>/outputs/smoke)")
    ap.add_argument("--resume", default=None, help="đường dẫn last.pt để chạy tiếp")
    ap.add_argument("--smoke", action="store_true", help="chạy nhanh trên tập con để kiểm pipeline")
    ap.add_argument("--subset", type=int, default=None, help="chỉ dùng N mẫu train (val thu nhỏ theo)")
    ap.add_argument("--split-mode", choices=SPLIT_MODES, default="random")
    ap.add_argument("--seed", type=int, default=TC.SEED)
    ap.add_argument("--patience", type=int, default=TC.PATIENCE)
    ap.add_argument("--no-early-stop", action="store_true")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="dừng an toàn trước khi vượt ngân sách thời gian (vẫn lưu last.pt)")
    ap.add_argument("--data-dir", default=str(PROCESSED_DIR))
    ap.add_argument("--num-workers", type=_workers_arg, default="auto",
                    help=f"số worker DataLoader hoặc 'auto' (= min({TC.NUM_WORKERS}, số CPU - 1), hạ nếu /dev/shm nhỏ)")
    ap.add_argument("--device", type=_device_arg, default="auto", help="auto | cpu | mps | cuda | cuda:N")
    ap.add_argument("--in-memory", action="store_true",
                    help="nạp toàn bộ dữ liệu vào RAM (cần ~9.2 GB trống). Nhanh nhất trên server. "
                         "Linux vẫn dùng worker (fork, chung RAM); Windows/macOS: 0 worker")
    ap.add_argument("--shuffle", choices=TC.SHUFFLE_MODES, default=TC.SHUFFLE_MODE,
                    help="chunk: xáo theo khối (HDD); random: hoán vị thuần (SSD/RAM)")
    ap.add_argument("--deterministic", action="store_true", help="cudnn tất định (chậm hơn)")
    ap.add_argument("--log-every", type=int, default=200, help="in tiến độ mỗi N iteration")
    args = ap.parse_args(argv)
    if args.smoke:
        args.subset = args.subset or TC.SMOKE_SUBSET
        args.epochs = args.epochs or TC.SMOKE_EPOCHS
        args.out = args.out or str(PROJECT_ROOT / "outputs" / "smoke")
    args.epochs = args.epochs or TC.EPOCHS
    args.out = args.out or str(PROJECT_ROOT / "outputs" / "run01")
    return args


def resolve_batch(args, device, saved_effective=None):
    """-> (micro, accum). Resume: giữ batch hiệu dụng của checkpoint, micro tính theo máy hiện tại."""
    bs = args.batch_size if args.batch_size is not None else "auto"
    if saved_effective:
        micro = auto_micro_batch(device, saved_effective) if bs == "auto" else bs
        if saved_effective % micro:
            sys.exit(f"--batch-size {micro} không chia hết batch hiệu dụng {saved_effective} của checkpoint")
        return micro, saved_effective // micro
    if bs == "auto":
        micro = auto_micro_batch(device)
        return micro, args.accum_steps or max(1, TC.BATCH_SIZE // micro)
    return bs, args.accum_steps or 1


def fmt_time(s):
    s = int(s)
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def model_hparams(args):
    p = dict(PRESETS[args.preset])
    return dict(K=args.filters or p["K"], k=args.filter_size or p["k"],
                variant=args.variant or p["variant"])


def acc_dtype(device):
    """Kiểu cộng dồn loss trên device. float64 cho kết quả giống hệt cộng float Python;
    MPS không có float64."""
    return torch.float32 if device.type == "mps" else torch.float64


@torch.no_grad()
def evaluate(model, loader, crit, device):
    # cộng dồn trên device, chỉ đọc về CPU một lần ở cuối -> không đồng bộ GPU mỗi batch
    model.eval()
    n = 0
    s_loss = torch.zeros(3, dtype=acc_dtype(device), device=device)   # loss, amc, doa (× số mẫu)
    hits = torch.zeros(2, dtype=torch.long, device=device)            # số mẫu đúng AMC, DOA
    pd, td = [], []
    for x, ym, yd, _ in loader:
        x, ym, yd = x.to(device, non_blocking=True), ym.to(device), yd.to(device)
        la, ld = model(x)
        loss, l_amc, l_doa = crit(la, ld, ym, yd)
        b = x.shape[0]
        n += b
        s_loss += torch.stack((loss, l_amc, l_doa)).to(s_loss.dtype) * b
        pdoa = ld.argmax(1)
        hits += torch.stack(((la.argmax(1) == ym).sum(), (pdoa == yd).sum()))
        pd.append(pdoa)
        td.append(yd)
    l_sum, l_amc, l_doa = (s_loss / n).tolist()
    c_amc, c_doa = hits.tolist()
    return dict(val_loss=l_sum, val_loss_amc=l_amc, val_loss_doa=l_doa,
                val_acc_amc=c_amc / n, val_acc_doa=c_doa / n,
                val_rmse_deg=rmse_deg(torch.cat(pd).cpu().numpy(), torch.cat(td).cpu().numpy()))


def train(args):
    # ------------------------------------------------------------ resume
    ckpt = None
    if args.resume:
        ckpt = load_checkpoint(args.resume)
        saved = ckpt["config_snapshot"]["args"]
        cli = vars(args)
        for k, v in saved.items():
            if k not in RUNTIME_ARGS and k != "resume":
                cli[k] = v
        args.out = os.path.dirname(os.path.abspath(args.resume))
        print(f"Resume từ {args.resume} (đã xong {ckpt['epoch']} epoch)")
    os.makedirs(args.out, exist_ok=True)

    seed_everything(args.seed, args.deterministic)
    device = select_device(args.device)
    saved_eff = ckpt["config_snapshot"].get("effective_batch") if ckpt else None
    args.batch_size, args.accum_steps = resolve_batch(args, device, saved_eff)

    # ------------------------------------------------------------ data
    x_mm, labels, meta = open_processed(args.data_dir)
    if args.in_memory:
        need, avail = x_mm.nbytes, mem_available_bytes()
        if avail is not None and avail < need * 1.1:
            sys.exit(f"--in-memory cần ~{need * 1.1 / 2**30:.1f} GB RAM trống (dữ liệu {need / 2**30:.1f} GB "
                     f"+ 10%), máy chỉ còn {avail / 2**30:.1f} GB -> bỏ --in-memory")
        t = time.time()
        x_mm = load_into_memory(x_mm)
        print(f"Đã nạp {need / 2**30:.1f} GB vào RAM trong {time.time() - t:.0f}s")
    batch_bytes = args.batch_size * int(np.prod(SAMPLE_SHAPE)) * 4
    args.num_workers = resolve_num_workers(args.num_workers, batch_bytes, args.in_memory)
    sp = load_or_make_splits(labels, args.data_dir, args.split_mode, SPLIT_SEED)
    check_splits(sp, labels, args.split_mode)
    tr_rows, va_rows = sp["train"], sp["val"]
    if args.subset:
        rng = np.random.default_rng(args.seed)
        n_val = max(args.batch_size, int(round(args.subset * VAL_FRACTION_OF_TRAIN
                                               / (1 - VAL_FRACTION_OF_TRAIN))))
        tr_rows = np.sort(rng.choice(tr_rows, min(args.subset, tr_rows.size), replace=False))
        va_rows = np.sort(rng.choice(va_rows, min(n_val, va_rows.size), replace=False))
    tr_ds = ModaNetDataset(x_mm, labels, tr_rows)
    va_ds = ModaNetDataset(x_mm, labels, va_rows)
    chunk, group = (TC.CHUNK_SIZE, TC.SHUFFLE_GROUP_CHUNKS) if args.shuffle == "chunk" else (1, 1)
    pin = TC.PIN_MEMORY and device.type == "cuda"
    tr_loader, tr_sampler = make_loader(tr_ds, args.batch_size, True, chunk, group, args.seed,
                                        args.num_workers, pin, TC.PREFETCH_FACTOR)
    va_loader, _ = make_loader(va_ds, args.batch_size, False, num_workers=args.num_workers,
                               pin_memory=pin, prefetch_factor=TC.PREFETCH_FACTOR)

    # ------------------------------------------------------------ model
    hp = model_hparams(args)
    model = MoDANet(**hp).to(device)
    crit = MultiTaskLoss(LOSS_WEIGHT_AMC, LOSS_WEIGHT_DOA)
    eff_bs = args.batch_size * args.accum_steps
    lr = args.lr if args.lr is not None else TC.LR * eff_bs / TC.BATCH_SIZE
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=TC.MOMENTUM)
    sched = torch.optim.lr_scheduler.StepLR(opt, TC.LR_DROP_EPOCH, TC.LR_DROP_FACTOR)

    start_epoch, best_val, bad_epochs, history = 0, math.inf, 0, []
    if ckpt:
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["optimizer_state"])
        sched.load_state_dict(ckpt["scheduler_state"])
        set_rng_state(ckpt["rng_state"])
        start_epoch, best_val = ckpt["epoch"], ckpt["best_val_loss"]
        bad_epochs, history = ckpt["epochs_no_improve"], ckpt.get("history", [])

    snapshot = {"model": model.hparams, "preset": args.preset, "args": vars(args).copy(),
                "modulations": MODULATIONS, "data_dtype": meta["dtype"],
                "split_file": split_path(args.data_dir, args.split_mode, SPLIT_SEED),
                "effective_batch": eff_bs, "lr0": lr}

    print(f"Model: preset={args.preset} {model.hparams} | {count_params(model):,} tham số")
    print(f"Dữ liệu: train {len(tr_ds):,} | val {len(va_ds):,} | split={args.split_mode} "
          f"| shuffle={args.shuffle} | {meta['dtype']} | {'RAM' if args.in_memory else 'memmap'} "
          f"| workers {args.num_workers}")
    print(f"Tối ưu: SGD lr={lr:g} momentum={TC.MOMENTUM} | batch {args.batch_size}×{args.accum_steps}"
          f"={eff_bs} | lr×{TC.LR_DROP_FACTOR} tại epoch {TC.LR_DROP_EPOCH} | epochs {args.epochs}"
          f" | early stop {'tắt' if args.no_early_stop else f'patience {args.patience}'}")
    if args.lr is None and eff_bs != TC.BATCH_SIZE:
        print(f"  (lr hạ tuyến tính theo batch hiệu dụng {eff_bs} ≠ {TC.BATCH_SIZE})")

    # ------------------------------------------------------------ log
    csv_path = os.path.join(args.out, "log.csv")
    new_csv = not os.path.exists(csv_path) or start_epoch == 0
    csv_f = open(csv_path, "w" if new_csv else "a", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS)
    if new_csv:
        writer.writeheader()
        csv_f.flush()

    header = (f"{'epoch':>5}  {'lr':>7}  {'train_loss':>10}  {'val_loss':>8}  {'gap':>7}  "
              f"{'val_acc_amc':>11}  {'val_acc_doa':>11}  {'rmse°':>6}  {'time':>7}  {'ETA':>7}")
    print(header)
    t_run = time.time()
    n_iter = len(tr_loader)
    epoch_times = []

    def ckpt_state(epoch):
        return dict(epoch=epoch, model_state=model.state_dict(), optimizer_state=opt.state_dict(),
                    scheduler_state=sched.state_dict(), rng_state=get_rng_state(),
                    config_snapshot=snapshot, best_val_loss=best_val,
                    epochs_no_improve=bad_epochs, history=history)

    stop_reason = "đủ số epoch"
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        tr_sampler.set_epoch(epoch)
        # loss cộng dồn trên device; .item() chỉ khi in log -> CPU chuẩn bị batch sau trong lúc GPU chạy
        n = 0
        s = torch.zeros(3, dtype=acc_dtype(device), device=device)   # loss, amc, doa (× số mẫu)
        opt.zero_grad(set_to_none=True)
        for it, (x, ym, yd, _) in enumerate(tr_loader):
            x = x.to(device, non_blocking=True)
            ym, yd = ym.to(device, non_blocking=True), yd.to(device, non_blocking=True)
            la, ld = model(x)
            loss, l_amc, l_doa = crit(la, ld, ym, yd)
            (loss / args.accum_steps).backward()
            if (it + 1) % args.accum_steps == 0 or it + 1 == n_iter:
                opt.step()
                opt.zero_grad(set_to_none=True)
            b = x.shape[0]
            n += b
            s += torch.stack((loss.detach(), l_amc.detach(), l_doa.detach())).to(s.dtype) * b
            if args.log_every and (it + 1) % args.log_every == 0:
                el = time.time() - t0
                print(f"   [{epoch + 1}] it {it + 1}/{n_iter}  loss {s[0].item() / n:.4f}  "
                      f"{el / (it + 1) * 1000:.0f} ms/it  còn ~{fmt_time(el / (it + 1) * (n_iter - it - 1))}",
                      flush=True)
        s_loss, s_amc, s_doa = (s / n).tolist()
        cur_lr = opt.param_groups[0]["lr"]
        sched.step()
        ev = evaluate(model, va_loader, crit, device)
        dt = time.time() - t0
        epoch_times.append(dt)
        row = dict(epoch=epoch + 1, lr=cur_lr, train_loss=s_loss, train_loss_amc=s_amc,
                   train_loss_doa=s_doa, gap=ev["val_loss"] - s_loss, epoch_time=dt, **ev)
        history.append(row)
        writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
        csv_f.flush()
        os.fsync(csv_f.fileno())

        improved = ev["val_loss"] < best_val
        if improved:
            best_val, bad_epochs = ev["val_loss"], 0
        else:
            bad_epochs += 1
        eta = np.mean(epoch_times) * (args.epochs - epoch - 1)
        print(f"{epoch + 1:>5}  {cur_lr:7.4f}  {row['train_loss']:10.4f}  {ev['val_loss']:8.4f}  "
              f"{row['gap']:+7.3f}  {ev['val_acc_amc'] * 100:10.1f}%  {ev['val_acc_doa'] * 100:10.1f}%  "
              f"{ev['val_rmse_deg']:6.2f}  {fmt_time(dt):>7}  {fmt_time(eta):>7}"
              f"{'  *best' if improved else ''}", flush=True)

        save_checkpoint(os.path.join(args.out, "last.pt"), **ckpt_state(epoch + 1))
        if improved:
            save_checkpoint(os.path.join(args.out, "best.pt"), **ckpt_state(epoch + 1))

        # 6B.3: val_loss tăng liên tục N epoch trong khi train_loss giảm -> overfitting thật
        W = TC.OVERFIT_WARN_EPOCHS
        if len(history) > W:
            h = history[-(W + 1):]
            if all(h[i + 1]["val_loss"] > h[i]["val_loss"] for i in range(W)) and \
               all(h[i + 1]["train_loss"] < h[i]["train_loss"] for i in range(W)):
                print(f"⚠️  val_loss tăng {W} epoch liên tục trong khi train_loss giảm -> "
                      f"dấu hiệu overfitting (xem MODANET_INSTRUCTION.md 6B.3)")

        if not args.no_early_stop and bad_epochs >= args.patience:
            stop_reason = f"early stopping (val_loss không cải thiện {args.patience} epoch)"
            break
        if args.max_hours is not None:
            elapsed = time.time() - t_run
            if elapsed + np.mean(epoch_times) > args.max_hours * 3600:
                stop_reason = f"hết ngân sách --max-hours {args.max_hours} (chạy tiếp bằng --resume)"
                break
    csv_f.close()
    print(f"Dừng: {stop_reason}. best val_loss = {best_val:.4f} | tổng {fmt_time(time.time() - t_run)}")
    print(f"Checkpoint: {os.path.join(args.out, 'last.pt')} , {os.path.join(args.out, 'best.pt')}")
    if epoch_times:
        print(f"Thời gian/epoch đo được: {fmt_time(np.mean(epoch_times))} "
              f"({len(tr_ds):,} mẫu train, device={device.type})")


def _sigterm_to_keyboard_interrupt(signum, frame):
    # `kill <PID>` (SIGTERM, mặc định) cư xử giống hệt Ctrl+C: dừng an toàn ở boundary
    # iteration/epoch gần nhất, in gợi ý --resume, thay vì bị kernel giết ngay lập tức.
    raise KeyboardInterrupt


def main():
    signal.signal(signal.SIGTERM, _sigterm_to_keyboard_interrupt)
    args = parse_args()
    try:
        train(args)
    except torch.cuda.OutOfMemoryError:
        micro = args.batch_size if isinstance(args.batch_size, int) else TC.BATCH_SIZE
        acc = args.accum_steps if isinstance(args.accum_steps, int) else 1
        print(f"\n❌ Hết VRAM với micro-batch {micro}. Giảm micro-batch, tăng accumulation để giữ "
              f"batch hiệu dụng {micro * acc}, ví dụ:\n"
              f"   --batch-size {max(1, micro // 2)} --accum-steps {acc * 2}\n"
              f"Checkpoint của epoch đã xong gần nhất vẫn ở {args.out}/last.pt (dùng --resume).")
        sys.exit(2)
    except KeyboardInterrupt:
        print(f"\nĐã ngắt. Chạy tiếp: python scripts/train.py --resume {args.out}/last.pt")
        sys.exit(130)


if __name__ == "__main__":
    main()
