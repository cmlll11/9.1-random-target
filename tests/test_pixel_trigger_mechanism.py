from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from pixel_trigger_mechanism import concentration, cosine
from mdluap.official_triggers import SSBAEncoderTrigger, TriggerStatus


def test_pixel_cosine_and_concentration_exclude_diagonal():
    vectors = np.eye(3, dtype=np.float32)
    assert cosine(vectors[0], vectors[0]) == 1.0
    assert concentration(vectors) == 0.0


def test_pixel_concentration_uses_shared_direction():
    vectors = np.asarray([[1.0, 0.0], [2.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    # Pairwise cosines are 1, -1, -1; their mean is -1/3.
    assert np.isclose(concentration(vectors), -1.0 / 3.0)


def test_ssba_encoder_accepts_cifar10_test_split():
    import torch
    from torch import nn

    class DummyEncoder(nn.Module):
        def forward(self, _fingerprints, images):
            return images

    adapter = SSBAEncoderTrigger(
        TriggerStatus("ssba", "test", True),
        DummyEncoder(),
        {"fingerprint_length": 1, "fingerprint_values": [[1.0]], "use_residual": 0, "quantize_uint8": False},
        bdb_root=ROOT / "third_party" / "BackdoorBench",
        device=torch.device("cpu"),
    )
    images = torch.zeros(1, 3, 32, 32)
    # Identity is not a real SSBA encoder, but verifies the CIFAR-10 split
    # contract without loading a checkpoint.
    output = adapter.apply(images, sample_indices=[0], split="cifar10_test")
    assert tuple(output.shape) == (1, 3, 32, 32)
