"""7.1 — hằng số khớp Bảng I + quét số magic trong src/models."""
import ast
from pathlib import Path

import pytest

from configs import model_config as MC
from configs import data_config as DC


def test_table_I_constants():
    assert MC.K_FILTERS == 64
    assert MC.NUM_MOD_CLASSES == 12
    assert MC.NUM_DOA_CLASSES == 121
    assert MC.POOL_BACKBONE_SIZE == 3 and MC.POOL_BACKBONE_STRIDE == 2
    assert MC.POOL_SKIP_SIZE == 5 and MC.POOL_SKIP_STRIDE == 4
    assert MC.SKIP_CONV_SIZE == 1 and MC.OUTPUT_CONV_SIZE == 1
    assert MC.KERNEL_WIDTH == 1 and MC.POOL_STRIDE_W == 1
    assert MC.BACKBONE_GROUPS == 2 and MC.SKIP_GROUPS == 1
    assert (MC.M_ANTENNAS, MC.N_SAMPLES, MC.IQ_CHANNELS) == (5, 1024, 2)
    assert MC.LOSS_WEIGHT_AMC == 1.0 and MC.LOSS_WEIGHT_DOA == 1.0


def test_filter_size_choice():
    assert MC.FILTER_SIZE in MC.FILTER_SIZE_CHOICES
    assert MC.FILTER_SIZE == 7
    assert MC.FILTER_SIZE_CHOICES == (3, 5, 7, 9)


def test_class_lists():
    assert len(DC.DOA_ANGLES) == MC.NUM_DOA_CLASSES == 121
    assert DC.DOA_ANGLES[0] == -60 and DC.DOA_ANGLES[-1] == 60
    assert len(DC.MODULATIONS) == MC.NUM_MOD_CLASSES == 12
    assert len(set(DC.MODULATIONS)) == 12
    assert sorted(DC.MODULATIONS) == sorted(DC.PAPER_MODULATION_ORDER)
    assert len(DC.SNR_LEVELS) == 31
    assert DC.N_TOTAL == 450_120


@pytest.mark.parametrize("variant", ["A", "B", "C"])
def test_variant_depth(variant):
    n_shared = MC.VARIANT_SHARED_BLOCKS[variant]
    n_branch = MC.TOTAL_RESBLOCK_DEPTH - n_shared
    assert n_shared + n_branch == MC.TOTAL_RESBLOCK_DEPTH == 3
    assert n_branch >= 0


def test_derived_padding():
    assert MC.POOL_BACKBONE_PAD == (MC.POOL_BACKBONE_SIZE - 1) // 2
    assert MC.POOL_SKIP_PAD == (MC.POOL_SKIP_SIZE - 1) // 2
    for k in MC.FILTER_SIZE_CHOICES:
        assert MC.same_pad(k) == (k - 1) // 2


def test_presets_valid():
    assert MC.PRESETS[MC.DEFAULT_PRESET] == {"K": 64, "k": 7, "variant": "A"}
    for p in MC.PRESETS.values():
        assert p["k"] in MC.FILTER_SIZE_CHOICES and p["variant"] in MC.VARIANT_SHARED_BLOCKS


MODEL_FILES = sorted((Path(__file__).resolve().parents[1] / "src" / "models").glob("*.py"))


@pytest.mark.parametrize("path", MODEL_FILES, ids=lambda p: p.name)
def test_no_magic_numbers_in_models(path):
    """Không một hằng số số học nào trong src/models — mọi số phải import từ configs."""
    tree = ast.parse(path.read_text())
    bad = [(n.lineno, n.value) for n in ast.walk(tree)
           if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
           and not isinstance(n.value, bool)]
    assert not bad, f"{path.name}: số magic tại (dòng, giá trị) {bad}"
