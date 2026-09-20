from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from badnet_adv_finetune import (  # noqa: E402
    MINIMUM_COHORT_SIZE,
    TARGET,
    TRAIN_COHORT_SIZE,
    select_successful_cohort,
)


def test_successful_cohort_preserves_probe_rank_and_filters_target_zero():
    rows = [
        {"rank": str(rank), "sample_index": str(rank), "true_label": "0" if rank == 3 else "1"}
        for rank in range(1, MINIMUM_COHORT_SIZE + 85)
    ]
    attacks = {
        rank: {"success": rank != 2}
        for rank in range(1, MINIMUM_COHORT_SIZE + 85)
    }
    selected = select_successful_cohort(rows, attacks)

    assert len(selected) == MINIMUM_COHORT_SIZE + 82
    assert all(int(row["true_label"]) != TARGET for row in selected)
    selected_ranks = [int(row["rank"]) for row in selected]
    assert selected_ranks == sorted(selected_ranks)
    assert 2 not in selected_ranks and 3 not in selected_ranks


def test_fine_tune_and_heldout_sizes_are_fixed():
    assert TRAIN_COHORT_SIZE == 200
    assert MINIMUM_COHORT_SIZE == TRAIN_COHORT_SIZE
