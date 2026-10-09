"""In bảng shape từng chặng + số tham số, đối chiếu MODANET_INSTRUCTION.md PHẦN 2.4 / 2.5.
Thoát với mã 1 nếu có bất kỳ chỗ nào lệch.

    python scripts/summary.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from configs.model_config import (  # noqa: E402
    EXPECTED_PARAM_COUNT, EXPECTED_PARAM_COUNT_BY_N, EXPECTED_PARAM_COUNT_PRESET,
    EXPECTED_SHAPES_N1024, M_ANTENNAS, N_SAMPLES, IQ_CHANNELS, PRESETS,
)
from src.models.modanet import MoDANet, count_params  # noqa: E402


def stage_shapes(model, N=N_SAMPLES, B=2):
    rec = {}
    hooks = [model.input_block.register_forward_hook(lambda m, i, o: rec.__setitem__("after_input", o.shape)),
             model.shared.register_forward_hook(lambda m, i, o: rec.__setitem__("after_shared", o.shape)),
             model.amc_head.fc.register_forward_hook(lambda m, i, o: rec.__setitem__("flatten", i[0].shape))]
    for j, blk in enumerate(model.amc_br):
        hooks.append(blk.register_forward_hook(
            lambda m, i, o, j=j: rec.__setitem__(f"after_branch_{j + 1}", o.shape)))
    model.eval()
    with torch.no_grad():
        a, d = model(torch.randn(B, M_ANTENNAS, N, IQ_CHANNELS))
    for h in hooks:
        h.remove()
    rec["amc"], rec["doa"] = a.shape, d.shape
    return rec


def main():
    ok = True

    def check(name, got, exp):
        nonlocal ok
        good = got == exp
        ok &= good
        print(f"  {name:28s} {str(got):>22s}   kỳ vọng {str(exp):>22s}   {'✓' if good else '✗ LỆCH'}")

    B = 2
    print(f"== Bảng shape (Type A, N={N_SAMPLES}, B={B}) — PHẦN 2.4 ==")
    m = MoDANet()
    rec = stage_shapes(m, B=B)
    check("input", (B, M_ANTENNAS, N_SAMPLES, IQ_CHANNELS), (B, 5, 1024, 2))
    for key in ("after_input", "after_shared", "after_branch_1", "after_branch_2"):
        check(key, tuple(rec[key]), (B,) + EXPECTED_SHAPES_N1024[key])
    check("flatten", tuple(rec["flatten"]), (B, EXPECTED_SHAPES_N1024["flatten_dim"]))
    check("AMC logits", tuple(rec["amc"]), (B, 12))
    check("DOA logits", tuple(rec["doa"]), (B, 121))

    print("\n== Số tham số theo (variant, filter size), K=64, N=1024 — PHẦN 2.5 ==")
    for (v, k), e in EXPECTED_PARAM_COUNT.items():
        check(f"Type {v}, k={k}×1", count_params(MoDANet(k=k, variant=v)), e)

    print("\n== Theo N (Type A, k=7) — PHẦN 2.4 ==")
    for n, e in EXPECTED_PARAM_COUNT_BY_N.items():
        mn = MoDANet(N=n)
        check(f"N={n} (L cuối {mn.final_len}, flat {mn.flat_dim})", count_params(mn), e)

    print("\n== Preset (6B.2) ==")
    for p, e in EXPECTED_PARAM_COUNT_PRESET.items():
        check(f"{p} {PRESETS[p]}", count_params(MoDANet.from_preset(p)), e)
    print("\n" + ("TẤT CẢ KHỚP" if ok else "CÓ CHỖ LỆCH — không được 'sửa cho chạy', báo cáo lại"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
