"""Đánh giá theo SNR + tái tạo Fig. 3(b) (Type A/B/C: AMC accuracy vs SNR, DOA RMSE vs SNR).

    python scripts/evaluate.py --typeA outputs/typeA/best.pt --typeB outputs/typeB/best.pt \\
        --typeC outputs/typeC/best.pt --out outputs/eval

Chạy 1 lượt forward trên tập test (giữ nguyên split lúc train, checkpoint tự khai
split_mode) cho từng checkpoint, gộp theo SNR (31 mức, -10..20 dB, bước 1). Ghi
outputs/eval/per_snr.csv (variant, snr_db, n, amc_acc, doa_rmse_deg) + 2 hình:
fig3_amc_vs_snr.png, fig3_doa_rmse_vs_snr.png (mỗi hình 1 trục, không vẽ 2 trục y
chồng lên nhau như bản gốc — xem MODANET_INSTRUCTION.md).
"""
import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from configs.data_config import PROCESSED_DIR, SNR_LEVELS, SPLIT_SEED, idx_to_angle  # noqa: E402
from src.data.dataset import ModaNetDataset, make_loader, open_processed  # noqa: E402
from src.data.splits import load_or_make_splits  # noqa: E402
from src.utils.checkpoint import build_model_from_checkpoint, load_checkpoint  # noqa: E402
from src.utils.device import select_device  # noqa: E402

# palette validé (dataviz skill, references/palette.md) — slot 1/2/3
COLOR = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}


@torch.no_grad()
def predict_test_set(ckpt_path, data_dir, device, batch_size):
    ckpt = load_checkpoint(ckpt_path)
    model = build_model_from_checkpoint(ckpt).to(device).eval()
    x_mm, labels, _ = open_processed(data_dir)
    mode = ckpt["config_snapshot"]["args"].get("split_mode", "random")
    rows = load_or_make_splits(labels, data_dir, mode, SPLIT_SEED)["test"]
    ds = ModaNetDataset(x_mm, labels, rows)
    loader, _ = make_loader(ds, batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    pm, tm, pd_, td, snr = [], [], [], [], []
    for x, ym, yd, s in loader:
        la, ld = model(x.to(device, non_blocking=True))
        pm.append(la.argmax(1).cpu())
        tm.append(ym)
        pd_.append(ld.argmax(1).cpu())
        td.append(yd)
        snr.append(s)
    return (torch.cat(pm).numpy(), torch.cat(tm).numpy(), torch.cat(pd_).numpy(),
            torch.cat(td).numpy(), torch.cat(snr).numpy())


def per_snr_metrics(pred_mod, true_mod, pred_doa, true_doa, snr):
    rows = []
    for s in SNR_LEVELS:
        m = snr == s
        n = int(m.sum())
        if n == 0:
            rows.append((s, 0, float("nan"), float("nan")))
            continue
        acc = float((pred_mod[m] == true_mod[m]).mean())
        err = idx_to_angle(pred_doa[m]) - idx_to_angle(true_doa[m])
        rmse = float(np.sqrt(np.mean(err.astype(np.float64) ** 2)))
        rows.append((s, n, acc, rmse))
    return rows


def write_csv(path, all_rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "snr_db", "n", "amc_acc", "doa_rmse_deg"])
        for variant, rows in all_rows.items():
            for s, n, acc, rmse in rows:
                w.writerow([variant, s, n, f"{acc:.6g}", f"{rmse:.6g}"])


def plot_metric(all_rows, col_idx, ylabel, title, out_path, scale=1.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
    for variant, rows in all_rows.items():
        xs = [r[0] for r in rows]
        ys = [r[col_idx] * scale for r in rows]
        ax.plot(xs, ys, marker="o", markersize=4, linewidth=2,
                color=COLOR[variant], label=f"MoDANet-Type{variant}")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linewidth=0.5, alpha=0.4)
    ax.legend(frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def read_csv(path):
    all_rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            all_rows.setdefault(r["variant"], []).append(
                (int(r["snr_db"]), int(r["n"]), float(r["amc_acc"]), float(r["doa_rmse_deg"])))
    return all_rows


# Kiểu vẽ giống Fig. 3(b) của bài (MATLAB): marker/màu từng Type, trục trái xanh, trục phải cam
FIG3B_STYLE = {"A": ("<", "#EDB120"), "B": ("o", "#E00000"), "C": ("D", "#0072BD")}
FIG3B_LEFT, FIG3B_RIGHT = "#0072BD", "#D95319"


def plot_fig3b(all_rows, out_path):
    """Tái tạo Fig. 3(b): AMC accuracy (trục trái, %) + DOA RMSE (trục phải, log) trên cùng 1 hình."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix"})
    fig, ax = plt.subplots(figsize=(5.6, 4.4), dpi=200)
    ax2 = ax.twinx()
    ax2.set_yscale("log")
    for variant, rows in all_rows.items():
        marker, color = FIG3B_STYLE[variant]
        xs = [r[0] for r in rows]
        kw = dict(marker=marker, color=color, markersize=5, linewidth=1.3)
        ax.plot(xs, [r[2] * 100 for r in rows], label=f"MoDANet-Type{variant}", **kw)
        ax2.plot(xs, [r[3] for r in rows], **kw)

    ax.set_xlim(-10, 20)
    ax.set_xticks([-10, 0, 10, 20])
    ax.set_ylim(40, 100)
    ax.set_xlabel("SNR (dB)", fontsize=13)
    ax.set_ylabel("AMC Accuracy (%)", color=FIG3B_LEFT, fontsize=13)
    ax.tick_params(axis="y", colors=FIG3B_LEFT, labelsize=11)
    ax.tick_params(axis="x", labelsize=11)
    ax2.set_ylabel("RMSE of DOA estimation (°)", color=FIG3B_RIGHT, fontsize=13)
    ax2.tick_params(axis="y", which="both", colors=FIG3B_RIGHT, labelsize=11)
    ax2.spines["right"].set_color(FIG3B_RIGHT)
    ax2.spines["left"].set_color(FIG3B_LEFT)
    ax.legend(loc="center right", fontsize=10, framealpha=1, edgecolor="black", fancybox=False)

    # Mũi tên chỉ trục như bài: đường tăng -> trục trái, đường giảm -> trục phải
    ax.annotate("", xy=(-9.5, 63), xytext=(-5, 63),
                arrowprops=dict(arrowstyle="-|>", color=FIG3B_LEFT, lw=0.8))
    ax.annotate("", xy=(-1, 80), xytext=(-6, 80),
                arrowprops=dict(arrowstyle="-|>", color=FIG3B_RIGHT, lw=0.8))

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(os.path.splitext(out_path)[0] + ".pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--typeA", default="outputs/typeA/best.pt")
    ap.add_argument("--typeB", default="outputs/typeB/best.pt")
    ap.add_argument("--typeC", default="outputs/typeC/best.pt")
    ap.add_argument("--data-dir", default=str(PROCESSED_DIR))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--from-csv", metavar="PER_SNR_CSV",
                    help="chỉ vẽ lại Fig. 3(b) từ per_snr.csv đã có, không chạy suy luận")
    args = ap.parse_args()

    if args.from_csv:
        out_path = os.path.join(args.out, "fig3b.png")
        plot_fig3b(read_csv(args.from_csv), out_path)
        print(f"Đã ghi {out_path} (+ .pdf)")
        return

    device = select_device(args.device)
    ckpts = {"A": args.typeA, "B": args.typeB, "C": args.typeC}
    all_rows, overall = {}, {}
    for variant, path in ckpts.items():
        if not os.path.exists(path):
            print(f"⚠️  bỏ qua Type{variant}: không thấy {path}")
            continue
        print(f"Đang suy luận Type{variant} ({path}) trên tập test...")
        pm, tm, pd_, td, snr = predict_test_set(path, args.data_dir, device, args.batch_size)
        all_rows[variant] = per_snr_metrics(pm, tm, pd_, td, snr)
        err = idx_to_angle(pd_) - idx_to_angle(td)
        overall[variant] = (float((pm == tm).mean()),
                            float(np.sqrt(np.mean(err.astype(np.float64) ** 2))))

    if not all_rows:
        sys.exit("Không có checkpoint nào để đánh giá.")

    csv_path = os.path.join(args.out, "per_snr.csv")
    write_csv(csv_path, all_rows)
    print(f"\nĐã ghi {csv_path}")

    plot_metric(all_rows, 2, "AMC Accuracy (%)",
                "MoDANet Type A/B/C — AMC accuracy vs SNR (Fig. 3b)",
                os.path.join(args.out, "fig3_amc_vs_snr.png"), scale=100.0)
    plot_metric(all_rows, 3, "RMSE of DOA estimation (°)",
                "MoDANet Type A/B/C — DOA RMSE vs SNR (Fig. 3b)",
                os.path.join(args.out, "fig3_doa_rmse_vs_snr.png"))
    print(f"Đã ghi {args.out}/fig3_amc_vs_snr.png , {args.out}/fig3_doa_rmse_vs_snr.png")
    plot_fig3b(all_rows, os.path.join(args.out, "fig3b.png"))
    print(f"Đã ghi {args.out}/fig3b.png (+ .pdf) — bản 2 trục giống Fig. 3(b) của bài")

    print("\nTrung bình toàn tập test (mọi SNR):")
    for variant, (acc, rmse) in overall.items():
        print(f"  Type{variant}: AMC acc {acc * 100:5.2f}%  |  DOA RMSE {rmse:.3f}°")


if __name__ == "__main__":
    main()
