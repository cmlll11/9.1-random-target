from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from layerwise_probe_top100_trigger_mechanism import load_selection_file, metric_rows


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


def test_shared_selection_file_requires_exact_unique_top100(tmp_path):
    path = tmp_path / "selected_probe_top100.csv"
    path.write_text("rank,sample_index\n" + "".join(f"{i + 1},{i}\n" for i in range(100)), encoding="utf-8")
    indices, rows = load_selection_file(path, 100)
    assert indices == list(range(100))
    assert len(rows) == 100

    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text("rank,sample_index\n" + "".join(f"{i + 1},{0 if i == 99 else i}\n" for i in range(100)), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_selection_file(duplicate, 100)

    short = tmp_path / "short.csv"
    short.write_text("rank,sample_index\n1,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 100"):
        load_selection_file(short, 100)
