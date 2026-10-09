"""Các khối của MoDANet [FIG 2]. KHÔNG chứa số magic — mọi hằng số import từ configs.

Layout tensor: (B, C, N, IQ). Mọi kernel dạng (k, 1): chỉ trượt dọc trục thời gian N,
trục I/Q giữ nguyên suốt mạng.
"""
import torch
import torch.nn as nn

from configs.model_config import (
    CONV_STRIDE, FLATTEN_START_DIM, KERNEL_WIDTH, PAD_W, POOL_STRIDE_W,
    BACKBONE_GROUPS, SKIP_GROUPS, SKIP_CONV_SIZE, OUTPUT_CONV_SIZE,
    POOL_BACKBONE_SIZE, POOL_BACKBONE_STRIDE, POOL_BACKBONE_PAD,
    POOL_SKIP_SIZE, POOL_SKIP_STRIDE, POOL_SKIP_PAD,
    same_pad,
)


def make_conv(cin, cout, k):
    """conv k×1, stride 1, 'same' padding, có bias [GAP: mặc định MATLAB/PyTorch]."""
    return nn.Conv2d(cin, cout, kernel_size=(k, KERNEL_WIDTH),
                     stride=(CONV_STRIDE, CONV_STRIDE),
                     padding=(same_pad(k), PAD_W), bias=True)


def _in_channels(cin, K, n_groups):
    """Số kênh vào của n nhóm nối tiếp: nhóm đầu nhận cin, các nhóm sau nhận K."""
    return ([cin] + [K] * n_groups)[:n_groups]


class ConvUnit(nn.Module):
    """Nhóm 4 lớp: conv -> pool -> norm -> relu   [TEXT] Fig.2b
    LƯU Ý: pool ĐỨNG TRƯỚC norm. Không đổi thứ tự."""

    def __init__(self, cin, cout, k, pool_k, pool_s, pool_p):
        super().__init__()
        self.conv = make_conv(cin, cout, k)
        self.pool = nn.MaxPool2d(kernel_size=(pool_k, KERNEL_WIDTH),
                                 stride=(pool_s, POOL_STRIDE_W),
                                 padding=(pool_p, PAD_W))
        self.norm = nn.BatchNorm2d(cout)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.norm(self.pool(self.conv(x))))


class InputBlock(ConvUnit):
    """[TEXT] input -> conv(k×1) -> pool(3×1, s2) -> norm -> relu."""

    def __init__(self, cin, K, k):
        super().__init__(cin, K, k, POOL_BACKBONE_SIZE, POOL_BACKBONE_STRIDE,
                         POOL_BACKBONE_PAD)


class ResBlock(nn.Module):
    """[FIG 2b] backbone (2 nhóm, conv k×1) + skip (1 nhóm, conv 1×1, pool 5×1 s4) -> add."""

    def __init__(self, cin, K, k):
        super().__init__()
        self.backbone = nn.Sequential(*[
            ConvUnit(c, K, k, POOL_BACKBONE_SIZE, POOL_BACKBONE_STRIDE, POOL_BACKBONE_PAD)
            for c in _in_channels(cin, K, BACKBONE_GROUPS)])
        self.skip = nn.Sequential(*[
            ConvUnit(c, K, SKIP_CONV_SIZE, POOL_SKIP_SIZE, POOL_SKIP_STRIDE, POOL_SKIP_PAD)
            for c in _in_channels(cin, K, SKIP_GROUPS)])

    def forward(self, x):
        b = self.backbone(x)
        s = self.skip(x)
        # bất biến sống còn: ceil(ceil(L/2)/2) == ceil(L/4)
        if b.shape != s.shape:
            raise RuntimeError(f"backbone {tuple(b.shape)} != skip {tuple(s.shape)}")
        return b + s


class OutputBlock(nn.Module):
    """[TEXT] conv -> norm -> relu -> fc (-> softmax CHỈ ở inference). Không có pool.
    forward() trả về LOGITS (nn.CrossEntropyLoss đã gồm log-softmax)."""

    def __init__(self, K, flat_dim, n_classes, conv_k=OUTPUT_CONV_SIZE):
        super().__init__()
        self.conv = make_conv(K, K, conv_k)
        self.norm = nn.BatchNorm2d(K)
        self.act = nn.ReLU()
        self.fc = nn.Linear(flat_dim, n_classes)

    def forward(self, x):
        h = self.act(self.norm(self.conv(x)))
        return self.fc(torch.flatten(h, start_dim=FLATTEN_START_DIM))
