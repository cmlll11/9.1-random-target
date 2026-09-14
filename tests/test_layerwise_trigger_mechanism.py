from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from layerwise_trigger_mechanism import LAYERS, concentration, cosine_rows, validate_endpoint_arrays


def test_concentration_excludes_diagonal():
    assert concentration(np.eye(3, dtype=np.float32)) == 0.0


def test_concentration_shared_direction_is_one():
    vectors = np.asarray([[1.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float32)
    assert np.isclose(concentration(vectors), 1.0)


def test_cosine_rows_marks_zero_norm_as_nan():
    result = cosine_rows(
        np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
    )
    assert np.isnan(result[0])
    assert np.isclose(result[1], 1.0)


def test_layer_order_is_stable():
    assert LAYERS == ("pixel", "conv1", "layer1", "layer2", "layer3", "layer4", "avgpool")


def test_endpoint_validation_requires_exact_reused_cohort():
    image = np.zeros((100, 3, 32, 32), dtype=np.float32)
    arrays = {
        "badnet_sample_indices": np.arange(99, dtype=np.int64),
        "badnet_original": image,
        "badnet_clean_adv": image,
        "badnet_backdoor_adv": image,
        "badnet_trigger": image,
    }
    with pytest.raises(ValueError, match="exactly 100"):
        validate_endpoint_arrays(arrays)


def test_endpoint_validation_accepts_pixel_archive_contract():
    image = np.zeros((100, 3, 32, 32), dtype=np.float32)
    arrays = {
        "badnet_sample_indices": np.arange(100, dtype=np.int64),
        "badnet_original": image,
        "badnet_clean_adv": image,
        "badnet_backdoor_adv": image,
        "badnet_trigger": image,
    }
    validated = validate_endpoint_arrays(arrays)
    assert validated[0].shape == (100,)
    assert all(value.shape == (100, 3, 32, 32) for value in validated[1:])
