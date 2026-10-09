import numpy as np

from configs.data_config import idx_to_angle


def accuracy(pred, true):
    pred, true = np.asarray(pred), np.asarray(true)
    return float((pred == true).mean()) if pred.size else float("nan")


def rmse_deg(pred_doa_idx, true_doa_idx):
    """RMSE (độ). angle = idx - 60 nên hiệu góc = hiệu chỉ số (bước 1°)."""
    a = idx_to_angle(np.asarray(pred_doa_idx)) - idx_to_angle(np.asarray(true_doa_idx))
    return float(np.sqrt(np.mean(a.astype(np.float64) ** 2))) if a.size else float("nan")
