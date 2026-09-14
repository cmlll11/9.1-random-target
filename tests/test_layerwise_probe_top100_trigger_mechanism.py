from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from layerwise_probe_top100_trigger_mechanism import metric_rows


def test_probe_layer_metric_rows_have_clean_and_badnet_gaps():
    vectors = np.asarray([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    residuals = {
        "clean0": {"adv": {"pixel": vectors}, "trigger": {"pixel": vectors}},
        "badnet0": {"adv": {"pixel": vectors}, "trigger": {"pixel": vectors}},
    }
    # The helper expects all protocol layers; use the same vectors for the
    # small unit test so no model or checkpoint is loaded.
    for alias in residuals:
        for kind in residuals[alias]:
            residuals[alias][kind] = {layer: vectors for layer in ("pixel", "conv1", "layer1", "layer2", "layer3", "layer4", "avgpool")}
    metrics, samples = metric_rows(residuals, np.asarray([10, 11, 12]))
    assert [row["layer"] for row in metrics] == ["pixel", "conv1", "layer1", "layer2", "layer3", "layer4", "avgpool"]
    assert all(np.isclose(row["clean_adv_concentration"], 1.0) for row in metrics)
    assert len(samples) == 7 * 3 * 2
