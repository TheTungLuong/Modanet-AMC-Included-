"""7.3 — Res-block: backbone 2 nhóm + skip 1 nhóm -> add."""
import math

import pytest
import torch
import torch.nn as nn

from configs.model_config import EXPECTED_RESBLOCK_PARAMS_K7, FILTER_SIZE, K_FILTERS
from src.models.blocks import ConvUnit, ResBlock


@pytest.fixture
def rb():
    return ResBlock(K_FILTERS, K_FILTERS, FILTER_SIZE)


def test_backbone_has_two_groups(rb):
    assert len(rb.backbone) == 2
    assert all(isinstance(m, ConvUnit) for m in rb.backbone)


def test_skip_has_one_group(rb):
    assert len(rb.skip) == 1 and isinstance(rb.skip[0], ConvUnit)


def test_backbone_conv(rb):
    for g in rb.backbone:
        assert g.conv.kernel_size == (FILTER_SIZE, 1)
        assert g.conv.out_channels == 64
        assert tuple(g.pool.kernel_size) == (3, 1) and tuple(g.pool.stride) == (2, 1)


def test_skip_conv_is_1x1(rb):
    """Bắt lỗi dùng nhầm FILTER_SIZE cho skip."""
    assert rb.skip[0].conv.kernel_size == (1, 1)
    assert rb.skip[0].conv.padding == (0, 0)


def test_skip_pool_5x1_s4(rb):
    p = rb.skip[0].pool
    assert tuple(p.kernel_size) == (5, 1)
    assert tuple(p.stride) == (4, 1)
    assert tuple(p.padding) == (2, 0)


@pytest.mark.parametrize("k", [3, 5, 7, 9])
@pytest.mark.parametrize("L", [512, 256, 128, 64, 32, 31, 17, 8, 7, 5, 3])
def test_branch_shapes_match(k, L):
    """Điều kiện tồn tại của phép add, kể cả L lẻ và rất ngắn."""
    blk = ResBlock(64, 64, k).eval()
    x = torch.randn(2, 64, L, 2)
    with torch.no_grad():
        b, s = blk.backbone(x), blk.skip(x)
    assert b.shape == s.shape
    assert b.shape[2] == math.ceil(L / 4)


def test_merge_is_add(rb):
    rb.eval()
    x = torch.randn(2, 64, 512, 2)
    with torch.no_grad():
        assert torch.allclose(rb(x), rb.backbone(x) + rb.skip(x))


def test_output_shape(rb):
    assert rb(torch.randn(2, 64, 512, 2)).shape == (2, 64, 128, 2)


def test_gradient_through_skip_and_backbone(rb):
    rb(torch.randn(4, 64, 512, 2)).sum().backward()
    g_skip = rb.skip[0].conv.weight.grad
    g_bb = rb.backbone[0].conv.weight.grad
    assert g_skip is not None and g_skip.abs().sum() > 0
    assert g_bb is not None and g_bb.abs().sum() > 0


def test_param_count_k7(rb):
    n = sum(p.numel() for p in rb.parameters())
    assert n == 2 * (64 * 64 * 7 + 64) + (64 * 64 * 1 + 64) + 3 * (2 * 64)
    assert n == EXPECTED_RESBLOCK_PARAMS_K7 == 62_016


@pytest.mark.parametrize("L", [512, 100, 33])
def test_chainable(L):
    a, b = ResBlock(64, 64, 7), ResBlock(64, 64, 7)
    assert b(a(torch.randn(2, 64, L, 2))).shape == (2, 64, math.ceil(math.ceil(L / 4) / 4), 2)


def test_mismatched_branches_raise():
    """Nếu ai đó sửa pool làm lệch 2 nhánh, forward phải báo lỗi rõ ràng (không broadcast âm thầm)."""
    blk = ResBlock(64, 64, 7)
    blk.skip[0].pool = nn.MaxPool2d((5, 1), stride=(2, 1), padding=(2, 0))
    with pytest.raises(RuntimeError, match="backbone"):
        blk(torch.randn(2, 64, 64, 2))
