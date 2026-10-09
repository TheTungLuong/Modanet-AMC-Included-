"""7.2 — Input block: conv -> pool -> norm -> relu."""
import math

import pytest
import torch
import torch.nn as nn

from configs.model_config import FILTER_SIZE, K_FILTERS, M_ANTENNAS
from src.models.blocks import InputBlock


@pytest.fixture
def blk():
    return InputBlock(M_ANTENNAS, K_FILTERS, FILTER_SIZE)


def test_conv_config(blk):
    c = blk.conv
    assert isinstance(c, nn.Conv2d)
    assert c.in_channels == 5 and c.out_channels == 64
    assert c.kernel_size == (FILTER_SIZE, 1)
    assert c.stride == (1, 1)
    assert c.padding == ((FILTER_SIZE - 1) // 2, 0)
    assert c.bias is not None


def test_pool_config(blk):
    p = blk.pool
    assert isinstance(p, nn.MaxPool2d)
    assert tuple(p.kernel_size) == (3, 1)
    assert tuple(p.stride) == (2, 1)
    assert tuple(p.padding) == (1, 0)


def test_norm_config(blk):
    assert isinstance(blk.norm, nn.BatchNorm2d)
    assert blk.norm.num_features == 64


def test_layer_order_conv_pool_norm_relu(blk):
    """Bắt lỗi đảo sang conv-norm-relu-pool."""
    assert [type(m) for m in blk.children()] == [nn.Conv2d, nn.MaxPool2d, nn.BatchNorm2d, nn.ReLU]


def test_forward_order_is_pool_before_norm(blk, x1024):
    """Không chỉ thứ tự khai báo: forward thật phải gọi pool TRƯỚC norm."""
    calls = []
    hs = [m.register_forward_hook(lambda m, i, o, n=n: calls.append(n))
          for n, m in blk.named_children()]
    blk(x1024)
    for h in hs:
        h.remove()
    assert calls == ["conv", "pool", "norm", "act"]


def test_output_shape(blk, x1024):
    out = blk(x1024)
    assert out.shape == (2, 64, 512, 2)
    assert out.shape[-1] == 2                 # trục I/Q bất biến
    assert out.min() >= 0                     # ReLU thật sự
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("N", [128, 256, 512, 1024, 1000, 999])
def test_many_N(blk, N):
    out = blk(torch.randn(2, 5, N, 2))
    assert out.shape == (2, 64, math.ceil(N / 2), 2)
