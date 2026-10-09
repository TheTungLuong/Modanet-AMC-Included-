"""Chạy được trên mọi máy: tự chọn thiết bị / micro-batch / worker, resume khi đổi máy."""
import types

import pytest
import torch

from configs import train_config as TC
from src.utils import device as D


def test_arch_parser():
    assert D._arch_supports((6, 1), ["sm_50", "sm_60", "sm_70"])          # GTX 1050 dùng cubin sm_60
    assert not D._arch_supports((6, 1), ["sm_70", "sm_80"])
    assert D._arch_supports((12, 0), ["sm_90", "sm_100", "sm_120"])       # Blackwell
    assert not D._arch_supports((12, 0), ["sm_80", "sm_86", "sm_90"])
    assert D._arch_supports((8, 9), ["sm_80", "sm_86"])                   # 4090 chạy cubin sm_86
    assert D._arch_supports((9, 0), ["compute_80"])                       # PTX JIT


def test_auto_micro_batch_cpu():
    assert D.auto_micro_batch(torch.device("cpu")) == TC.BATCH_SIZE


@pytest.mark.parametrize("gb,micro", [(1.94, 32), (3.9, 64), (8, 128), (24, 128), (80, 128), (1.0, 16)])
def test_auto_micro_batch_table(monkeypatch, gb, micro):
    """MODANET_INSTRUCTION.md 9.3: 2 GB -> 32, 4 GB -> 64; server -> 128. Batch hiệu dụng luôn 128."""
    props = types.SimpleNamespace(total_memory=int(gb * 2**30))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda d: props)
    got = D.auto_micro_batch(torch.device("cuda"))
    assert got == micro and TC.BATCH_SIZE % got == 0


def test_resolve_batch_keeps_effective_on_resume(monkeypatch):
    import importlib.util
    import sys
    from configs.data_config import PROJECT_ROOT
    spec = importlib.util.spec_from_file_location("train_script", PROJECT_ROOT / "scripts" / "train.py")
    train = importlib.util.module_from_spec(spec)
    sys.modules["train_script"] = train
    spec.loader.exec_module(train)
    cpu = torch.device("cpu")
    a = train.parse_args([])
    assert train.resolve_batch(a, cpu) == (128, 1)
    a = train.parse_args(["--batch-size", "64", "--accum-steps", "2"])
    assert train.resolve_batch(a, cpu) == (64, 2)                      # instruction 8.1
    a = train.parse_args(["--batch-size", "64"])
    assert train.resolve_batch(a, cpu) == (64, 1)                      # -> lr hạ tuyến tính
    # resume: checkpoint GTX 1050 (32×4) sang server: micro tính lại, batch hiệu dụng giữ 128
    monkeypatch.setattr(train, "auto_micro_batch", lambda d, eff=128: min(128, eff))
    a = train.parse_args(["--resume", "x.pt"])
    assert train.resolve_batch(a, cpu, saved_effective=128) == (128, 1)
    a = train.parse_args(["--resume", "x.pt", "--batch-size", "16"])
    assert train.resolve_batch(a, cpu, saved_effective=128) == (16, 8)
    a = train.parse_args(["--resume", "x.pt", "--batch-size", "48"])
    with pytest.raises(SystemExit):
        train.resolve_batch(a, cpu, saved_effective=128)


def test_auto_num_workers_respects_shm(monkeypatch):
    monkeypatch.setattr(D, "available_cpus", lambda: 32)
    monkeypatch.setattr(D, "shm_free_bytes", lambda: 64 * 2**20)        # docker mặc định 64 MB
    batch = 128 * 5 * 1024 * 2 * 4                                       # 5.2 MB
    n = D.auto_num_workers(batch, prefetch=4, verbose=False)
    assert 0 <= n < TC.NUM_WORKERS and n * 4 * batch * 2 <= 64 * 2**20
    monkeypatch.setattr(D, "shm_free_bytes", lambda: 16 * 2**30)
    assert D.auto_num_workers(batch, verbose=False) == TC.NUM_WORKERS
    monkeypatch.setattr(D, "available_cpus", lambda: 2)
    assert D.auto_num_workers(batch, verbose=False) == 1


@pytest.mark.parametrize("set_method,default,share", [
    ("fork", "spawn", True), ("spawn", "fork", False), ("forkserver", "fork", False),
    (None, "fork", True),          # Linux, Python < 3.14
    (None, "forkserver", False),   # Linux, Python >= 3.14
    (None, "spawn", False),        # macOS, Windows
])
def test_workers_share_memory(monkeypatch, set_method, default, share):
    import multiprocessing
    monkeypatch.setattr(torch.multiprocessing, "get_start_method", lambda allow_none=False: set_method)
    monkeypatch.setattr(multiprocessing, "get_all_start_methods", lambda: [default, "spawn"])
    assert D.workers_share_memory() is share


def test_resolve_num_workers(monkeypatch):
    """--in-memory: fork -> vẫn dùng worker (auto hoặc số tự đặt); spawn/forkserver -> 0."""
    batch = 128 * 5 * 1024 * 2 * 4
    monkeypatch.setattr(D, "auto_num_workers", lambda b, verbose=True: 4)
    monkeypatch.setattr(D, "workers_share_memory", lambda: True)
    assert D.resolve_num_workers("auto", batch, in_memory=False) == 4
    assert D.resolve_num_workers("auto", batch, in_memory=True) == 4
    assert D.resolve_num_workers(6, batch, in_memory=True) == 6
    assert D.resolve_num_workers(0, batch, in_memory=True) == 0
    monkeypatch.setattr(D, "workers_share_memory", lambda: False)
    assert D.resolve_num_workers("auto", batch, in_memory=False) == 4    # memmap: pickle chỉ đường dẫn
    assert D.resolve_num_workers(3, batch, in_memory=False) == 3
    assert D.resolve_num_workers("auto", batch, in_memory=True, verbose=False) == 0
    assert D.resolve_num_workers(6, batch, in_memory=True, verbose=False) == 0


def test_checkpoint_loads_on_cpu(tmp_path):
    """Checkpoint train trên GPU phải nạp được trên máy chỉ có CPU (map_location='cpu')."""
    from src.models.modanet import MoDANet
    from src.utils.checkpoint import build_model_from_checkpoint, load_checkpoint, save_checkpoint
    m = MoDANet()
    if torch.cuda.is_available():
        m = m.cuda()
    save_checkpoint(str(tmp_path / "c.pt"), epoch=1, model_state=m.state_dict(),
                    config_snapshot={"model": m.hparams})
    m2 = build_model_from_checkpoint(load_checkpoint(str(tmp_path / "c.pt")))
    assert next(m2.parameters()).device.type == "cpu"
    assert all(torch.equal(a.cpu(), b) for a, b in zip(m.state_dict().values(), m2.state_dict().values()))


def test_no_absolute_user_paths_in_code():
    """Không được có đường dẫn cứng tới máy cá nhân trong code (để chạy trên server)."""
    from configs.data_config import PROJECT_ROOT
    bad = []
    for p in list((PROJECT_ROOT / "src").rglob("*.py")) + list((PROJECT_ROOT / "configs").rglob("*.py")) \
            + list((PROJECT_ROOT / "scripts").glob("*.py")) + list(PROJECT_ROOT.glob("*.sh")):
        if "/home/" in p.read_text() or "C:\\\\" in p.read_text():
            bad.append(p.name)
    assert not bad, bad
