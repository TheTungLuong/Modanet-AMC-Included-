"""Hằng số kiến trúc MoDANet. Nguồn: Doan et al., IEEE Comm. Letters 26(2), 2022.
Nhãn xuất xứ: [TABLE-I] [TEXT] [FIG] [DERIVED] [GAP] — xem MODANET_INSTRUCTION.md

Đây là NƠI DUY NHẤT chứa hằng số kiến trúc. src/models/*.py không được có số magic
(tests/test_config.py quét AST để bắt).
"""

# ---- Kích thước tín hiệu ----
M_ANTENNAS   = 5      # [TEXT] ULA 5 phần tử
N_SAMPLES    = 1024   # [TEXT] số mẫu envelope phức
IQ_CHANNELS  = 2      # [TEXT] I và Q

# ---- Bộ lọc ----
K_FILTERS    = 64     # [TEXT] "K = 64 filters in all conv layers"
FILTER_SIZE  = 7      # [TABLE-I] ghi 3, nhưng [TEXT] chốt 7×1 là trade-off tốt nhất -> mặc định 7
FILTER_SIZE_CHOICES = (3, 5, 7, 9)   # [TEXT] Fig.4b
SKIP_CONV_SIZE   = 1  # [TABLE-I] "unit filters of 1×1" — KHÔNG đổi theo FILTER_SIZE
OUTPUT_CONV_SIZE = 1  # [TABLE-I] — MODANET_INSTRUCTION.md 1.3: luôn 1×1, KHÔNG đổi theo FILTER_SIZE.
#   (Ghi chú: đoạn chữ Fig.4b của bài nhắc cả "output block" khi tinh chỉnh filter size.
#    Người dùng xác nhận 2026-09-22: ƯU TIÊN instruction -> giữ 1×1.)
KERNEL_WIDTH = 1      # [TABLE-I] mọi kernel dạng k×1 -> không trượt qua trục I/Q
CONV_STRIDE  = 1      # [DERIVED] conv 'same', chiều dài chỉ giảm ở pool

# ---- Pooling (CỐ ĐỊNH, không đổi theo FILTER_SIZE) ----
POOL_BACKBONE_SIZE   = 3       # [TABLE-I] pool size 3×1
POOL_BACKBONE_STRIDE = 2       # [TABLE-I] stride (2,1)
POOL_BACKBONE_PAD    = 1       # [DERIVED] -> ceil(L/2)
POOL_SKIP_SIZE       = 5       # [TABLE-I] pool size 5×1
POOL_SKIP_STRIDE     = 4       # [TABLE-I] stride (4,1)
POOL_SKIP_PAD        = 2       # [DERIVED] -> ceil(L/4)
POOL_STRIDE_W        = 1       # [TABLE-I] thành phần thứ 2 của stride (s,1)
PAD_W                = 0       # [DERIVED] kernel rộng 1 -> không pad trục I/Q

# ---- Res-block ----
BACKBONE_GROUPS = 2   # [TEXT] "two four-layer groups" (conv-pool-norm-relu)
SKIP_GROUPS     = 1   # [TEXT] "one four-layer group is employed"

# ---- Topology ----
VARIANT = "A"                  # [TEXT] Type A cho kết quả tốt nhất
VARIANT_SHARED_BLOCKS = {"A": 1, "B": 2, "C": 3}   # [FIG 3a]
TOTAL_RESBLOCK_DEPTH  = 3      # [FIG 3a] mọi đường đi đều qua 3 res-block

# ---- Đầu ra ----
NUM_MOD_CLASSES = 12   # [TABLE-I] fc 12 neurons
NUM_DOA_CLASSES = 121  # [TABLE-I] fc 121 neurons
FLATTEN_START_DIM = 1  # [DERIVED] giữ trục batch khi flatten trước fc

# ---- Loss ----
LOSS_WEIGHT_AMC = 1.0  # [TEXT] "equivalent weights used for two tasks"
LOSS_WEIGHT_DOA = 1.0  # [TEXT]

# ---- Preset dung lượng (MODANET_INSTRUCTION.md 6B.2) ----
# [GAP] Không có trong bài; chỉ dùng khi train trên TẬP CON. Mặc định luôn là 'paper'.
PRESETS = {
    "paper":  {"K": 64, "k": 7, "variant": "A"},   # = cấu hình bài báo
    "medium": {"K": 64, "k": 5, "variant": "A"},
    "small":  {"K": 32, "k": 5, "variant": "A"},
    "tiny":   {"K": 16, "k": 3, "variant": "A"},
}
DEFAULT_PRESET = "paper"

# ---- Giá trị vàng để test đối chiếu (MODANET_INSTRUCTION.md 2.4 / 2.5 / 6B.2) ----
EXPECTED_PARAM_COUNT = {   # [DERIVED], đã verify — (variant, k) với K=64, N=1024
    ("A", 3): 292_293, ("A", 5): 374_853,
    ("A", 7): 457_413, ("A", 9): 539_973,
    ("B", 7): 395_397, ("C", 7): 333_381,
}
EXPECTED_PARAM_COUNT_BY_N = {   # [DERIVED] Type A, K=64, k=7
    128: 338_245, 256: 355_269, 512: 389_317, 1024: 457_413,
}
EXPECTED_PARAM_COUNT_PRESET = {  # [DERIVED] MODANET_INSTRUCTION.md 6B.2
    "paper": 457_413, "medium": 374_853, "small": 129_125, "tiny": 44_757,
}
EXPECTED_SHAPES_N1024 = {  # [DERIVED], đã verify — (C, L, IQ)
    "after_input":    (64, 512, 2),
    "after_shared":   (64, 128, 2),
    "after_branch_1": (64,  32, 2),
    "after_branch_2": (64,   8, 2),
    "flatten_dim":    1024,
}
EXPECTED_RESBLOCK_PARAMS_K7 = 62_016   # [DERIVED] 2*(64*64*7+64) + (64*64+64) + 3*(2*64)


# ---- Hàm suy ra (không phải hằng số tự do) ----
def same_pad(k):
    """[DERIVED] padding giữ nguyên chiều dài cho conv stride 1, kernel lẻ."""
    return (k - 1) // 2


def pool_out_len(L, size, stride, pad):
    """[DERIVED] công thức chiều dài ra của nn.MaxPool2d (ceil_mode=False)."""
    return (L + 2 * pad - size) // stride + 1
