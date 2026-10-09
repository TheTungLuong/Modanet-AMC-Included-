"""Save/resume checkpoint — ghi nguyên tử (tmp + os.replace) để mất điện không hỏng file."""
import os

import torch

from src.models.modanet import MoDANet


def save_checkpoint(path, **state):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, map_location="cpu"):
    # checkpoint chứa rng_state (numpy) -> cần weights_only=False; chỉ nạp file của chính mình
    return torch.load(path, map_location=map_location, weights_only=False)


def build_model_from_checkpoint(ckpt):
    """Dựng lại đúng model từ config_snapshot['model'] (hparams của MoDANet)."""
    model = MoDANet(**ckpt["config_snapshot"]["model"])
    model.load_state_dict(ckpt["model_state"])
    return model
