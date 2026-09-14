"""Score Clean0 with an existing target-0 Probe and save one reusable Top-100."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mdluap.data import cifar10_dataset
from mdluap.models import load_modelzoo_classifier
from mdluap.probes import RidgeProbe, target_conditioned_logits_features
from pilot_common import batch_images, seed_everything, timestamp_run_dir, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-zoo-root", required=True)
    parser.add_argument("--probe-archive", required=True)
    parser.add_argument("--output-root", default="results/stage1d_probe_top100_selection")
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=20260914)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.target != 0 or args.top_k != 100:
        raise ValueError("this selector is fixed to target=0 and Top-100")
    seed_everything(args.random_seed)
    device = torch.device(args.device)
    data_root = Path(args.data_root).expanduser().resolve()
    probe_path = Path(args.probe_archive).expanduser().resolve()
    if not probe_path.is_file():
        raise FileNotFoundError(probe_path)
    test_data = cifar10_dataset(data_root, train=False)
    indices = list(range(len(test_data)))
    probe = RidgeProbe.load(probe_path)
    model, info = load_modelzoo_classifier("clean0", model_zoo_root=str(Path(args.model_zoo_root).expanduser().resolve()), device=device)
    logits_rows: list[torch.Tensor] = []
    features_rows: list[np.ndarray] = []
    labels: list[int] = []
    for batch_indices, images, batch_labels in batch_images(test_data, indices, batch_size=args.batch_size, device=device):
        logits = model(images).detach()
        logits_rows.append(logits.cpu())
        features_rows.append(target_conditioned_logits_features(logits, args.target).cpu().numpy())
        labels.extend(int(value) for value in batch_labels.cpu().tolist())
    logits = torch.cat(logits_rows)
    features = np.concatenate(features_rows, axis=0)
    predictions = logits.argmax(dim=1).numpy()
    scores = probe.predict(features)
    positions = np.argsort(-scores, kind="mergesort")[:args.top_k]
    selected_indices = [int(indices[position]) for position in positions]
    output = timestamp_run_dir(args.output_root, "probe_selection")
    selected_set = set(selected_indices)
    score_rows = [{
        "sample_index": index,
        "true_label": labels[position],
        "clean0_original_prediction": int(predictions[position]),
        "clean0_eligible": int(predictions[position]) != args.target,
        "probe_score": float(scores[position]),
        "selected_top100": index in selected_set,
        "selection_rank": selected_indices.index(index) + 1 if index in selected_set else None,
    } for position, index in enumerate(indices)]
    positions = {index: position for position, index in enumerate(indices)}
    selected_rows = [{
        "rank": rank,
        "sample_index": index,
        "true_label": labels[positions[index]],
        "clean0_original_prediction": int(predictions[positions[index]]),
        "clean0_eligible": int(predictions[positions[index]]) != args.target,
        "probe_score": float(scores[positions[index]]),
        "target": args.target,
    } for rank, index in enumerate(selected_indices, start=1)]
    write_csv(output / "probe_selection_scores.csv", score_rows)
    write_csv(output / "selected_probe_top100.csv", selected_rows)
    write_json(output / "selection_metadata.json", {
        "protocol": "clean123-probe-artifact-clean0-rescore-top100-v1",
        "probe_archive": str(probe_path),
        "probe_archive_sha256": sha256_file(probe_path),
        "selection_model": "clean0",
        "selected_count": len(selected_indices),
        "selected_indices": selected_indices,
        "target": args.target,
        "probe_feature_names": list(probe.feature_names),
        "clean0_model_info": info,
    })
    print(f"Probe selection complete: {output}", flush=True)
    print(f"selected_count={len(selected_indices)}", flush=True)


if __name__ == "__main__":
    main()
