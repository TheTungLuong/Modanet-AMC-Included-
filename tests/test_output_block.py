"""7.4 — Output block: conv(1×1) -> norm -> relu -> fc, trả logits."""
import pytest
import torch
import torch.nn as nn

from src.models.blocks import OutputBlock
from src.models.modanet import MoDANet


@pytest.fixture(params=[("amc", 12), ("doa", 121)], ids=["amc", "doa"])
def head(request):
    m = MoDANet()
    return getattr(m, f"{request.param[0]}_head"), request.param[1], m


def test_conv_1x1(head):
    h, _, _ = head
    assert h.conv.kernel_size == (1, 1)
    assert h.conv.in_channels == h.conv.out_channels == 64


@pytest.mark.parametrize("k", [3, 5, 7, 9])
def test_conv_1x1_for_every_filter_size(k):
    """MODANET_INSTRUCTION.md 1.3: conv output block LUÔN 1×1, không đổi theo FILTER_SIZE."""
    m = MoDANet(k=k)
    assert m.amc_head.conv.kernel_size == m.doa_head.conv.kernel_size == (1, 1)


def test_no_pool(head):
    h, _, _ = head
    assert not any(isinstance(m, (nn.MaxPool2d, nn.AvgPool2d)) for m in h.modules())


def test_order(head):
    h, _, _ = head
    assert [type(m) for m in h.children()] == [nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.Linear]


def test_fc_dims(head):
    h, n_cls, model = head
    assert model.final_len == 8
    assert h.fc.in_features == 64 * model.final_len * 2 == 1024
    assert h.fc.out_features == n_cls


def test_returns_logits_not_softmax(head):
    h, n_cls, _ = head
    assert not any(isinstance(m, (nn.Softmax, nn.LogSoftmax)) for m in h.modules())
    h.eval()
    with torch.no_grad():
        out = h(torch.randn(16, 64, 8, 2))
    assert out.shape == (16, n_cls)
    assert out.min() < 0                                  # xác suất không bao giờ âm
    assert not torch.allclose(out.exp().sum(1), torch.ones(16))


def test_standalone_construct():
    h = OutputBlock(64, 1024, 12)
    assert h(torch.randn(2, 64, 8, 2)).shape == (2, 12)
