"""Kiểm tra vật lý trên mẫu thật — bắt lỗi permute sai trục và gán nhầm nhãn.

Mẫu ở layout PyTorch (M=5, N, IQ=2).
"""
import numpy as np

INTER_ELEMENT_PHASE = np.pi     # v = 2*pi*d/lambda với d = lambda/2 [TEXT] ct.(3)


def to_complex(sample):
    """(M, N, 2) -> (M, N) complex."""
    return sample[..., 0] + 1j * sample[..., 1]


def estimate_doa_deg(sample, v=INTER_ELEMENT_PHASE):
    """Ước lượng góc tới từ độ dốc pha qua các anten.
    Quy ước bài báo ct.(3): A[m] = exp(-j*v*m*sin(theta)) -> slope = -v*sin(theta)."""
    z = to_complex(sample)                                   # (M, N)
    ph = np.array([np.angle(np.vdot(z[0], z[m])) for m in range(z.shape[0])])
    slope = np.polyfit(np.arange(len(ph)), np.unwrap(ph), 1)[0]
    return float(np.degrees(np.arcsin(np.clip(-slope / v, -1, 1))))


def frame_power(sample):
    return float((sample.astype(np.float64) ** 2).sum(axis=-1).mean())


def c42(sample, antenna=0):
    """Cumulant chuẩn hóa C42 — bất biến với pha/lệch tần.
    Đường bao hằng (PSK, FSK, SSB 1 tone) ≈ -1; 16QAM ≈ -0.68; 64QAM ≈ -0.62."""
    z = to_complex(sample)[antenna]
    z = z - z.mean()
    z = z / np.sqrt(np.mean(np.abs(z) ** 2))
    c20 = np.mean(z ** 2)
    return float((np.mean(np.abs(z) ** 4) - np.abs(c20) ** 2 - 2).real)
