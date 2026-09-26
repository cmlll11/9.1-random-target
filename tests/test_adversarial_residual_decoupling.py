from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from adversarial_residual_decoupling import (  # noqa: E402
    pairwise_cosine_stats,
    projection_residual,
)


def test_pairwise_concentration_excludes_diagonal_and_keeps_negative_values():
    vectors = np.asarray([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    stats = pairwise_cosine_stats(vectors)

    assert stats["valid_pair_count"] == 3
    assert stats["valid_vector_count"] == 3
    assert stats["mean"] == -1 / 3


def test_projection_residual_is_orthogonal_to_paired_reference():
    delta = np.asarray([[3.0, 4.0], [0.0, 2.0]])
    reference = np.asarray([[1.0, 0.0], [1.0, 0.0]])
    residual, alpha, projection_ratio, orthogonality = projection_residual(delta, reference)

    np.testing.assert_allclose(residual, [[0.0, 4.0], [0.0, 2.0]], atol=1e-10)
    np.testing.assert_allclose(alpha, [3.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(projection_ratio, [0.6, 0.0], atol=1e-10)
    np.testing.assert_allclose(orthogonality[:, 0], [0.0, 0.0], atol=1e-10)
