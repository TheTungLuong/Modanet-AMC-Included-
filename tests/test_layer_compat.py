"""7.5 — khớp lớp kế tiếp trên toàn mạng (dùng forward hook).

Chạy `pytest tests/test_layer_compat.py -v -s` để in bảng shape đối chiếu PHẦN 2.4.
"""
import math

import pytest
import torch
import torch.nn as nn

from src.models.blocks import ConvUnit, OutputBlock, ResBlock
from src.models.modanet import MoDANet


def trace(model, x):
    """Ghi (tên, shape vào, shape ra) của MỌI module theo thứ tự thực thi."""
    rec = {}
    order = []
    hooks = []
    for name, m in model.named_modules():
        if name == "":
            continue

        def hook(mod, inp, out, name=name):
            rec[name] = (tuple(inp[0].shape), tuple(out.shape))
            order.append(name)
        hooks.append(m.register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return rec, order


def children_chain(model, rec):
    """Các cặp (trước, sau) cần khớp: con liền kề trong module tuần tự."""
    pairs = []
    for name, m in model.named_modules():
        seq = None
        if isinstance(m, (nn.Sequential, ConvUnit)):
            seq = [n for n, _ in m.named_children()]
        elif isinstance(m, OutputBlock):
            seq = ["conv", "norm", "act"]          # act -> flatten -> fc kiểm riêng
        if not seq:
            continue
        full = [f"{name}.{c}" if name else c for c in seq]
        full = [f for f in full if f in rec]
        pairs += list(zip(full[:-1], full[1:]))
    return pairs


@pytest.fixture(params=["A", "B", "C"])
def traced(request):
    model = MoDANet(variant=request.param)
    rec, order = trace(model, torch.randn(2, 5, 1024, 2))
    return request.param, model, rec, order


def test_sequential_neighbours_match(traced):
    _, model, rec, _ = traced
    pairs = children_chain(model, rec)
    assert pairs
    for a, b in pairs:
        assert rec[a][1] == rec[b][0], f"{a} ra {rec[a][1]} != {b} vào {rec[b][0]}"


def test_channel_continuity(traced):
    """out_channels của conv trước == in_channels của conv kế tiếp trên cùng đường đi."""
    _, model, _, _ = traced
    for br, head in ((model.amc_br, model.amc_head), (model.doa_br, model.doa_head)):
        blocks = [model.input_block] + list(model.shared) + list(br)
        c = model.input_block.conv.out_channels
        for blk in blocks[1:]:
            assert isinstance(blk, ResBlock)
            assert blk.backbone[0].conv.in_channels == c
            assert blk.skip[0].conv.in_channels == c
            for g1, g2 in zip(blk.backbone[:-1], blk.backbone[1:]):
                assert g1.conv.out_channels == g2.conv.in_channels
            c = blk.backbone[-1].conv.out_channels
            assert blk.skip[-1].conv.out_channels == c
        assert head.conv.in_channels == c


def test_batchnorm_matches_conv(traced):
    _, model, _, _ = traced
    units = [m for m in model.modules() if isinstance(m, (ConvUnit, OutputBlock))]
    assert units
    for u in units:
        assert u.norm.num_features == u.conv.out_channels


def test_linear_matches_feature_map(traced):
    _, model, rec, _ = traced
    for h in ("amc_head", "doa_head"):
        act_out = rec[f"{h}.act"][1]
        fc_in = rec[f"{h}.fc"][0]
        assert fc_in == (act_out[0], math.prod(act_out[1:]))
        assert fc_in[1] == getattr(model, h).fc.in_features


def test_branch_point_matches(traced):
    variant, model, rec, _ = traced
    shared_out = rec["shared"][1] if len(model.shared) else rec["input_block"][1]
    for br, head in (("amc_br", "amc_head"), ("doa_br", "doa_head")):
        first = f"{br}.0" if len(getattr(model, br)) else f"{head}.conv"
        assert rec[first][0] == shared_out, f"Type {variant}: {first}"


def test_merge_point_matches(traced):
    _, model, rec, _ = traced
    n = 0
    for name, m in model.named_modules():
        if isinstance(m, ResBlock):
            assert rec[f"{name}.backbone"][1] == rec[f"{name}.skip"][1], name
            assert rec[name][1] == rec[f"{name}.backbone"][1]
            n += 1
    assert n == len(model.shared) + len(model.amc_br) + len(model.doa_br)


def test_print_shape_table(traced):
    variant, _, rec, order = traced
    if variant != "A":
        return
    top = [n for n in order if n.count(".") <= 1 and not n.split(".")[-1] in ("act",)]
    print("\n  Bảng shape Type A (đối chiếu MODANET_INSTRUCTION.md PHẦN 2.4)")
    print(f"  {'module':22s} {'vào':>18s} {'ra':>18s}")
    for n in top:
        print(f"  {n:22s} {str(rec[n][0]):>18s} {str(rec[n][1]):>18s}")
    assert rec["input_block"][1] == (2, 64, 512, 2)
    assert rec["shared"][1] == (2, 64, 128, 2)
    assert rec["amc_br.0"][1] == (2, 64, 32, 2)
    assert rec["amc_br.1"][1] == (2, 64, 8, 2)
