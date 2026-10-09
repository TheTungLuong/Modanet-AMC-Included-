"""MoDANet — mạng đa nhiệm Y-shape cho AMC + DOA [FIG 2a, FIG 3a].

Type A (mặc định):
    input -> InputBlock -> RB1 (chung) -+-> RB2a -> RB3a -> OutputBlock -> 12 logits (AMC)
                                        +-> RB2b -> RB3b -> OutputBlock -> 121 logits (DOA)
"""
import torch.nn as nn

from configs.model_config import (
    M_ANTENNAS, N_SAMPLES, IQ_CHANNELS, K_FILTERS, FILTER_SIZE, FILTER_SIZE_CHOICES,
    OUTPUT_CONV_SIZE, VARIANT, VARIANT_SHARED_BLOCKS, TOTAL_RESBLOCK_DEPTH,
    NUM_MOD_CLASSES, NUM_DOA_CLASSES, LOSS_WEIGHT_AMC, LOSS_WEIGHT_DOA,
    POOL_BACKBONE_SIZE, POOL_BACKBONE_STRIDE, POOL_BACKBONE_PAD,
    POOL_SKIP_SIZE, POOL_SKIP_STRIDE, POOL_SKIP_PAD,
    PRESETS, pool_out_len,
)
from src.models.blocks import InputBlock, OutputBlock, ResBlock


def resblock_out_len(L):
    """Chiều dài sau 1 res-block (tính theo nhánh skip; test đảm bảo bằng backbone)."""
    return pool_out_len(L, POOL_SKIP_SIZE, POOL_SKIP_STRIDE, POOL_SKIP_PAD)


def final_length(N):
    """Chiều dài thời gian cuối cùng trước output block."""
    L = pool_out_len(N, POOL_BACKBONE_SIZE, POOL_BACKBONE_STRIDE, POOL_BACKBONE_PAD)
    for _ in range(TOTAL_RESBLOCK_DEPTH):
        L = resblock_out_len(L)
    return L


class MoDANet(nn.Module):
    def __init__(self, M=M_ANTENNAS, K=K_FILTERS, k=FILTER_SIZE, N=N_SAMPLES,
                 n_amc=NUM_MOD_CLASSES, n_doa=NUM_DOA_CLASSES, variant=VARIANT,
                 output_conv_k=OUTPUT_CONV_SIZE):
        super().__init__()
        if k not in FILTER_SIZE_CHOICES:
            raise ValueError(f"filter size {k} không thuộc {FILTER_SIZE_CHOICES}")
        if variant not in VARIANT_SHARED_BLOCKS:
            raise ValueError(f"variant {variant!r} không thuộc {tuple(VARIANT_SHARED_BLOCKS)}")
        self.hparams = dict(M=M, K=K, k=k, N=N, n_amc=n_amc, n_doa=n_doa,
                            variant=variant, output_conv_k=output_conv_k)
        self.n_shared = VARIANT_SHARED_BLOCKS[variant]
        self.n_branch = TOTAL_RESBLOCK_DEPTH - self.n_shared   # tổng độ sâu luôn = 3

        self.input_block = InputBlock(M, K, k)
        self.shared = nn.Sequential(*[ResBlock(K, K, k) for _ in range(self.n_shared)])
        self.amc_br = nn.Sequential(*[ResBlock(K, K, k) for _ in range(self.n_branch)])
        self.doa_br = nn.Sequential(*[ResBlock(K, K, k) for _ in range(self.n_branch)])

        self.final_len = final_length(N)
        self.flat_dim = K * self.final_len * IQ_CHANNELS
        self.amc_head = OutputBlock(K, self.flat_dim, n_amc, output_conv_k)
        self.doa_head = OutputBlock(K, self.flat_dim, n_doa, output_conv_k)

    @classmethod
    def from_preset(cls, name, **overrides):
        p = PRESETS[name]
        kw = dict(K=p["K"], k=p["k"], variant=p["variant"])
        kw.update(overrides)
        return cls(**kw)

    def forward(self, x):                    # x: (B, M, N, IQ)
        h = self.shared(self.input_block(x))
        return self.amc_head(self.amc_br(h)), self.doa_head(self.doa_br(h))   # logits


class MultiTaskLoss(nn.Module):
    """[TEXT] ct.(5): L = L_AMC + L_DOA, mỗi thành phần là categorical cross-entropy (6)."""

    def __init__(self, w_amc=LOSS_WEIGHT_AMC, w_doa=LOSS_WEIGHT_DOA):
        super().__init__()
        self.w_amc, self.w_doa = w_amc, w_doa
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits_amc, logits_doa, y_amc, y_doa):
        l_amc = self.ce(logits_amc, y_amc)
        l_doa = self.ce(logits_doa, y_doa)
        return self.w_amc * l_amc + self.w_doa * l_doa, l_amc, l_doa


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = ["MoDANet", "MultiTaskLoss", "count_params", "final_length"]
