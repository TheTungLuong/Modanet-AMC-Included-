"""Hằng số dataset MoDANet.

Dataset IEEE DataPort của tác giả: 450,120 file .mat, mỗi file 1 tín hiệu, tổ chức
thành cây thư mục  SNR{+xx}dB/<MOD>/<±DOA>/dataXXXXXXXX.mat .
Nhãn giải mã được từ chỉ số trong tên file (xem src/data/labels.py) và được đối chiếu
chéo với tên thư mục (khớp 450,120/450,120 — đã kiểm trên máy này).
"""
import os
from pathlib import Path

# ---- Đường dẫn (ghi đè bằng biến môi trường nếu cần) ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get(                       # thư mục chứa SNR+xxdB/...
    "MODANET_DATA_ROOT", PROJECT_ROOT.parent / "MoDANet-dataset"))
PROCESSED_DIR = Path(os.environ.get(                   # memmap sau khi convert
    "MODANET_PROCESSED_DIR", PROJECT_ROOT / "data_processed"))
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"

# ---- Điều chế ----
# ⚠️ THỨ TỰ NÀY LÀ HỢP ĐỒNG DỮ LIỆU: MODULATIONS[mod_i] với mod_i giải mã từ TÊN FILE.
# Đã kiểm trên dataset đầy đủ: thứ tự trong chỉ số file KHÁC thứ tự liệt kê trong bài báo.
# Bằng chứng (2026-09-22):
#   (1) quét 450,120 file: mỗi thư mục <MOD> ánh xạ duy nhất tới 1 mod_i theo đúng bảng dưới;
#   (2) cumulant C42 (bất biến pha) ở SNR 20 dB: thư mục PSK/QPSK/8PSK/QFSK/8FSK/SSB-SC ≈ -0.98
#       (đường bao hằng), 16QAM ≈ -0.66 (lý thuyết -0.68), 64QAM ≈ -0.60 (lý thuyết -0.62)
#       -> tên thư mục đúng; bảng của bài báo (PSK=0, ...) sẽ gán nhầm tên 8/12 lớp.
# Đổi thứ tự này = sai tên lớp (không sai việc train, vì chỉ là hoán vị).
MODULATIONS = ["16QAM", "64QAM", "PSK", "QPSK", "8PSK", "QFSK",
               "8FSK", "16APSK", "4PAM", "LFM", "DSB-SC", "SSB-SC"]
# Thứ tự bài báo liệt kê (mục IV-A) — CHỈ để hiển thị/báo cáo, KHÔNG dùng làm chỉ số lớp.
PAPER_MODULATION_ORDER = ["PSK", "QPSK", "8PSK", "QFSK", "8FSK", "16APSK",
                          "16QAM", "64QAM", "4PAM", "LFM", "DSB-SC", "SSB-SC"]

# ---- DOA / SNR ----
DOA_MIN, DOA_MAX, DOA_STEP = -60, 60, 1                 # [TEXT]
DOA_ANGLES = list(range(DOA_MIN, DOA_MAX + DOA_STEP, DOA_STEP))   # 121 góc
SNR_MIN, SNR_MAX, SNR_STEP = -10, 20, 1                 # [TEXT]
SNR_LEVELS = list(range(SNR_MIN, SNR_MAX + SNR_STEP, SNR_STEP))   # 31 mức

N_MOD = len(MODULATIONS)          # 12
N_DOA = len(DOA_ANGLES)           # 121
N_SNR = len(SNR_LEVELS)           # 31
N_REP = 10                        # realization mỗi tổ hợp (kiểm trên dataset thật)
N_TOTAL = N_MOD * N_DOA * N_SNR * N_REP   # 450,120  [TEXT]

# ---- Định dạng file ----
MAT_VARIABLE = "data"             # đúng 1 biến trong mỗi file
RAW_SHAPE = (1024, 5, 2)          # (N, M, IQ) — KHÁC bài báo (N×2×M), đã kiểm
PERMUTE_RAW_TO_TORCH = (1, 0, 2)  # (N, M, IQ) -> (M, N, IQ). KHÔNG dùng (2,0,1)
SAMPLE_SHAPE = (5, 1024, 2)       # layout PyTorch mỗi mẫu (M, N, IQ)
FILENAME_REGEX = r"^data(\d+)\.mat$"

# ---- Nhãn trong labels.npy ----
LABEL_COLUMNS = ("mod_idx", "doa_idx", "snr_db")   # int16, shape (N_TOTAL, 3)
COL_MOD, COL_DOA, COL_SNR = 0, 1, 2

# ---- Chuẩn hóa ----
# [GAP] Bài báo không nêu chuẩn hóa -> giữ False. Đã kiểm trên dataset ĐẦY ĐỦ (2026-09-22):
#   công suất frame ≈ 1 + 10^(-SNR/10), tức tín hiệu có công suất đơn vị rồi mới cộng nhiễu
#   (median tỉ số = 1.008; -10 dB -> ~10.9, 0 dB -> ~2.0, 20 dB -> ~1.03); |x| max = 13.44.
#   (Instruction đo 0.99–1.23 / ±3.34 chỉ trên 10 file 4PAM @3 dB -> không đại diện.)
#   Thang đo bị chặn, fp16 dư sức, BN ngay sau conv đầu tự chuẩn hóa -> không cần chuẩn hóa thêm.
#   Chuẩn hóa từng frame sẽ xóa thông tin công suất (tương quan với SNR) — nếu thử, bật bằng cờ.
NORMALIZE = False

# ---- Chia tập ----
TEST_FRACTION = 0.2            # [TEXT] "randomly split with 80% for training"
VAL_FRACTION_OF_TRAIN = 0.1    # [GAP] bài không có val; tách từ train, không đụng test
SPLIT_SEED = 42                # [GAP]
SPLIT_MODES = ("random", "group")   # random = như bài; group = giữ 10 realization cùng phía

# ---- Hợp đồng giá trị (cảnh báo) — thực đo trên dataset đầy đủ ----
AMPLITUDE_LIMIT = 16.0             # mọi SNR; thực đo |x| max = 13.44 (SNR -10 dB)
AMPLITUDE_LIMIT_SNR_GE0 = 5.0      # SNR >= 0 dB; thực đo 4.81
SIGNAL_POWER_RANGE = (0.25, 2.0)   # công suất frame / (1 + 10^(-SNR/10)); thực đo 0.37–1.56, median 1.008
FRAME_POWER_RANGE = (0.9, 1.3)     # riêng 10 file mẫu 4PAM @3 dB (fixtures); thực đo 0.994–1.230


def expected_frame_power(snr_db):
    """Công suất tín hiệu đơn vị + công suất nhiễu 10^(-SNR/10)."""
    return 1.0 + 10.0 ** (-snr_db / 10.0)

# ---- Convert ----
CONVERT_BATCH = 2000           # số file mỗi lô khi convert
CONVERT_DTYPES = ("float16", "float32")
DEFAULT_CONVERT_DTYPE = "float16"   # 9.2 GB, sai số tương đối ~1e-3


def idx_to_angle(doa_idx):
    return DOA_ANGLES[0] + doa_idx * DOA_STEP


def angle_to_idx(angle):
    return (angle - DOA_ANGLES[0]) // DOA_STEP
