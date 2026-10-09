"""Siêu tham số huấn luyện. Bài gốc KHÔNG nêu -> toàn bộ là [GAP].
Căn cứ chính: MCNet (Huynh-The et al., IEEE Comm. Letters 2020) — cùng nhóm tác giả.
"""

OPTIMIZER    = "sgd"    # [GAP] MCNet: SGD
LR           = 0.01     # [GAP] MCNet
MOMENTUM     = 0.9      # [GAP] MCNet
LR_DROP_EPOCH  = 30     # [GAP] MCNet: "dropped 90% after 30 epochs"
LR_DROP_FACTOR = 0.1    # [GAP] MCNet
EPOCHS       = 60       # [GAP] MCNet
BATCH_SIZE   = 128      # [GAP] MCNet — batch hiệu dụng = batch_size × accum_steps
SEED         = 42       # [GAP]

EARLY_STOPPING = True   # được phép bật mặc định (chỉ chọn checkpoint, không đổi model)
PATIENCE       = 8      # [GAP] theo val_loss

# Không có trong bài gốc -> KHÔNG dùng: dropout, weight decay, augmentation,
# label smoothing, AdamW, scheduler khác.

# ---- Micro-batch tự động theo VRAM (--batch-size auto) — batch hiệu dụng luôn = BATCH_SIZE ----
# MODANET_INSTRUCTION.md 9.3: 4 GB -> 64 (accum ×2), 2 GB -> 32 (accum ×4).
# [GAP] mở rộng cho server: >= 6 GB -> 128 (accum ×1); < 1.5 GB -> 16 (accum ×8).
# Ngưỡng hơi thấp hơn danh định vì GPU "4 GB" báo ~3.9 GB, "2 GB" báo ~1.94 GB.
AUTO_BATCH_BY_VRAM_GB = ((6.0, 128), (3.5, 64), (1.5, 32), (0.0, 16))

# ---- DataLoader ----
NUM_WORKERS = 4          # i7-9700 8 nhân, chừa 4 cho main process. 'auto' = min(4, số CPU - 1)
PIN_MEMORY  = True
PREFETCH_FACTOR = 4

# ---- Thứ tự đọc dữ liệu ----
# [GAP] Trên HDD, đọc ngẫu nhiên 360k mẫu/epoch từ memmap 9.2 GB (> RAM trống) rất chậm
# (~150 IOPS -> ~40 phút/epoch chỉ cho I/O). 'chunk' xáo trộn theo khối liền kề:
#   - chia chỉ số train (đã sắp) thành khối CHUNK_SIZE mẫu liền kề trên đĩa,
#   - xáo thứ tự khối, gom SHUFFLE_GROUP_CHUNKS khối thành một nhóm rồi xáo trộn mẫu trong nhóm.
# Mỗi batch 128 mẫu lấy từ ~64 vị trí ngẫu nhiên khác nhau trên đĩa -> gần iid, đọc gần tuần tự.
# 'random' = hoán vị ngẫu nhiên thuần (nên dùng khi dữ liệu nằm trên SSD / vừa RAM).
SHUFFLE_MODE = "chunk"
SHUFFLE_MODES = ("chunk", "random")
CHUNK_SIZE = 64
SHUFFLE_GROUP_CHUNKS = 64

# ---- Smoke test ----
SMOKE_SUBSET = 2000
SMOKE_EPOCHS = 2

# ---- Theo dõi overfitting (6B.3) ----
OVERFIT_WARN_EPOCHS = 5  # val_loss tăng liên tục N epoch trong khi train_loss giảm -> cảnh báo
