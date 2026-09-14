from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from probe_cifar10_top500_asr import parse_floats, select_probe_topk


def test_select_probe_topk_is_stable_and_descending():
    assert select_probe_topk(np.asarray([1.0, 3.0, 3.0, 2.0]), [10, 11, 12, 13], 3) == [11, 12, 13]


def test_select_probe_topk_rejects_invalid_size():
    with pytest.raises(ValueError):
        select_probe_topk(np.ones(2), [0, 1], 3)


def test_analysis_epsilon_parser():
    assert parse_floats("1,1.5") == (1.0, 1.5)
