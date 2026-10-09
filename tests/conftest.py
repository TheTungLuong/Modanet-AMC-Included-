import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.model_config import M_ANTENNAS, N_SAMPLES, IQ_CHANNELS  # noqa: E402


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture
def x1024():
    return torch.randn(2, M_ANTENNAS, N_SAMPLES, IQ_CHANNELS)


def pytest_configure(config):
    config.addinivalue_line("markers", "fulldata: cần dataset đầy đủ hoặc memmap đã convert")
