import random

import numpy as np
import torch


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # benchmark=True nhanh hơn trên GPU nhưng không tất định từng bit
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def get_rng_state():
    st = {"python": random.getstate(), "numpy": np.random.get_state(),
          "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st):
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        # resume trên máy khác (số GPU khác / từ CPU sang GPU): khôi phục được bao nhiêu thì khôi phục
        n = min(len(st["cuda"]), torch.cuda.device_count())
        for i in range(n):
            torch.cuda.set_rng_state(st["cuda"][i], i)
