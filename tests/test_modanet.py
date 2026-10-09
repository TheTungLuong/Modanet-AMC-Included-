"""7.6 — toàn mạng MoDANet."""
import warnings

import pytest
import torch

from configs.model_config import (
    EXPECTED_PARAM_COUNT, EXPECTED_PARAM_COUNT_BY_N, EXPECTED_PARAM_COUNT_PRESET,
    EXPECTED_SHAPES_N1024, LOSS_WEIGHT_AMC, LOSS_WEIGHT_DOA, VARIANT_SHARED_BLOCKS, PRESETS,
)
from src.models.modanet import MoDANet, MultiTaskLoss, count_params


def test_output_shapes(x1024):
    a, d = MoDANet()(x1024)
    assert a.shape == (2, 12) and d.shape == (2, 121)


def test_stage_shapes(x1024):
    m = MoDANet()
    rec = {}
    m.input_block.register_forward_hook(lambda *a: rec.__setitem__("after_input", a[2].shape[1:]))
    m.shared.register_forward_hook(lambda *a: rec.__setitem__("after_shared", a[2].shape[1:]))
    m.amc_br[0].register_forward_hook(lambda *a: rec.__setitem__("after_branch_1", a[2].shape[1:]))
    m.amc_br[1].register_forward_hook(lambda *a: rec.__setitem__("after_branch_2", a[2].shape[1:]))
    m.amc_head.fc.register_forward_pre_hook(lambda mod, i: rec.__setitem__("flatten_dim", i[0].shape[1]))
    m(x1024)
    for k, v in EXPECTED_SHAPES_N1024.items():
        assert (tuple(rec[k]) if k != "flatten_dim" else rec[k]) == v, k


@pytest.mark.parametrize("variant,k", sorted(EXPECTED_PARAM_COUNT))
def test_param_count(variant, k):
    """MODANET_INSTRUCTION.md 2.5 — fail cứng."""
    assert count_params(MoDANet(k=k, variant=variant)) == EXPECTED_PARAM_COUNT[(variant, k)]


@pytest.mark.parametrize("N", sorted(EXPECTED_PARAM_COUNT_BY_N))
def test_param_count_by_N(N):
    m = MoDANet(N=N)
    assert count_params(m) == EXPECTED_PARAM_COUNT_BY_N[N]
    a, d = m(torch.randn(2, 5, N, 2))
    assert a.shape == (2, 12) and d.shape == (2, 121)


@pytest.mark.parametrize("preset", list(PRESETS))
def test_presets(preset):
    m = MoDANet.from_preset(preset)
    assert count_params(m) == EXPECTED_PARAM_COUNT_PRESET[preset]
    a, d = m(torch.randn(2, 5, 1024, 2))
    assert a.shape == (2, 12) and d.shape == (2, 121)


def test_param_count_in_paper_range():
    """MODANET_INSTRUCTION.md 2.5: Fig.6b chỉ đọc gần đúng -> ngưỡng CẢNH BÁO, không fail."""
    n = count_params(MoDANet())
    if not 0.4e6 <= n <= 0.7e6:
        warnings.warn(f"{n:,} tham số nằm ngoài vùng đọc từ Fig.6b")


@pytest.mark.parametrize("variant", list(VARIANT_SHARED_BLOCKS))
def test_y_depth(variant):
    m = MoDANet(variant=variant)
    assert m.n_shared + m.n_branch == 3
    assert len(m.shared) == VARIANT_SHARED_BLOCKS[variant]
    assert len(m.amc_br) == len(m.doa_br) == m.n_branch


def _losses(m, x):
    ym = torch.randint(0, 12, (x.shape[0],))
    yd = torch.randint(0, 121, (x.shape[0],))
    a, d = m(x)
    return MultiTaskLoss()(a, d, ym, yd)


def test_trunk_gets_gradient_from_both_heads(x1024):
    m = MoDANet()
    _, l_amc, _ = _losses(m, x1024)
    l_amc.backward()
    g1 = m.input_block.conv.weight.grad.clone()
    m.zero_grad()
    _, _, l_doa = _losses(m, x1024)
    l_doa.backward()
    g2 = m.input_block.conv.weight.grad.clone()
    assert g1.abs().sum() > 0 and g2.abs().sum() > 0
    assert not torch.allclose(g1, g2)


def test_branches_independent(x1024):
    m = MoDANet()
    _, l_amc, _ = _losses(m, x1024)
    l_amc.backward()
    for p in list(m.doa_br.parameters()) + list(m.doa_head.parameters()):
        assert p.grad is None or p.grad.abs().sum() == 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.amc_br.parameters())


def test_total_loss_weights(x1024):
    assert LOSS_WEIGHT_AMC == LOSS_WEIGHT_DOA == 1.0
    total, l_amc, l_doa = _losses(MoDANet(), x1024)
    assert torch.allclose(total, 1.0 * l_amc + 1.0 * l_doa)


def test_overfit_one_batch():
    torch.manual_seed(1)
    m = MoDANet()
    x = torch.randn(8, 5, 1024, 2)
    ym, yd = torch.randint(0, 12, (8,)), torch.randint(0, 121, (8,))
    opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9)
    crit = MultiTaskLoss()
    first = None
    for _ in range(50):
        a, d = m(x)
        loss, _, _ = crit(a, d, ym, yd)
        first = loss.item() if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5 * first, (first, loss.item())


def test_reproducible_init():
    torch.manual_seed(42)
    a = MoDANet().state_dict()
    torch.manual_seed(42)
    b = MoDANet().state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_eval_mode_deterministic(x1024):
    m = MoDANet()
    m.train()
    m(x1024)                       # cập nhật running stats
    m.eval()
    with torch.no_grad():
        o1, o2 = m(x1024), m(x1024)
    assert torch.equal(o1[0], o2[0]) and torch.equal(o1[1], o2[1])


def test_invalid_config():
    with pytest.raises(ValueError):
        MoDANet(k=4)
    with pytest.raises(ValueError):
        MoDANet(variant="D")


def _cuda_ok():
    try:
        return torch.cuda.is_available() and torch.zeros(1, device="cuda").sum().item() == 0
    except RuntimeError:
        return False


@pytest.fixture
def no_tf32():
    """GPU Ampere trở lên: cuDNN mặc định dùng TF32 cho conv (mantissa 10 bit) -> sai số ~1e-3.
    Tắt TF32 khi so với CPU theo ngưỡng 1e-4 của instruction 7.6, rồi khôi phục."""
    old = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = old


@pytest.mark.skipif(not _cuda_ok(), reason="không có CUDA dùng được")
def test_cpu_gpu_consistent(x1024, no_tf32):
    m = MoDANet().eval()
    with torch.no_grad():
        a_cpu, d_cpu = m(x1024)
        mg = m.cuda()
        a_gpu, d_gpu = mg(x1024.cuda())
    assert (a_cpu - a_gpu.cpu()).abs().max() < 1e-4
    assert (d_cpu - d_gpu.cpu()).abs().max() < 1e-4


def _mps_ok():
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


@pytest.mark.skipif(not _mps_ok(), reason="không có Apple MPS")
def test_cpu_mps_consistent(x1024):
    m = MoDANet().eval()
    with torch.no_grad():
        a_cpu, d_cpu = m(x1024)
        a_m, d_m = m.to("mps")(x1024.to("mps"))
    assert (a_cpu - a_m.cpu()).abs().max() < 1e-3
    assert (d_cpu - d_m.cpu()).abs().max() < 1e-3


def test_train_step_on_every_available_device():
    """1 bước train (forward + backward + SGD) chạy được trên mọi thiết bị có trên máy."""
    devs = ["cpu"] + (["cuda"] if _cuda_ok() else []) + (["mps"] if _mps_ok() else [])
    for dev in devs:
        m = MoDANet().to(dev)
        opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9)
        x = torch.randn(4, 5, 1024, 2, device=dev)
        a, d = m(x)
        loss, _, _ = MultiTaskLoss()(a, d, torch.zeros(4, dtype=torch.long, device=dev),
                                     torch.zeros(4, dtype=torch.long, device=dev))
        loss.backward()
        opt.step()
        assert torch.isfinite(loss).item(), dev
