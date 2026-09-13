"""Measure successful target-PGD direction concentration on a fixed Top-100.

This is deliberately separate from the trigger-alignment experiment.  The
selected CIFAR-100 Top-100 is fixed before this script starts, and no
trigger-success condition is used to define the attack cohort.  For each
Model Zoo classifier and epsilon, concentration is computed only over samples
that are eligible and successfully targeted by PGD at that same epsilon.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mdluap.data import cifar100_dataset
from mdluap.models import load_modelzoo_classifier
from mdluap.targeted_pgd import targeted_pgd_endpoint
from pilot_common import batch_images, seed_everything, timestamp_run_dir, write_csv, write_json


DEFAULT_ALIASES = (
    "clean0",
    "badnet0",
    "blended0",
    "wanet0",
    "inputaware0",
    "ssba0",
    "adaptive_blend01",
)
DEFAULT_EPSILON_PIXELS = (0.75, 1.0, 1.25)


class AvgPoolFeatures:
    """Extract the pre-classifier feature for both supported architectures."""

    def __init__(self, classifier: nn.Module):
        backbone = getattr(classifier, "model", classifier)
        layer = getattr(backbone, "avgpool", None)
        self._spatial = layer is None
        if self._spatial:
            layer = getattr(backbone, "layer4", None)
        if layer is None:
            raise AttributeError("model must expose avgpool or layer4")
        self.value: torch.Tensor | None = None
        self.handle = layer.register_forward_hook(self._hook)
        self.classifier = classifier

    def _hook(self, _module, _inputs, output):
        self.value = output[0] if isinstance(output, (tuple, list)) else output

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.classifier(images)
        if self.value is None:
            raise RuntimeError("feature hook did not capture a value")
        value = self.value
        if self._spatial and value.ndim == 4:
            value = torch.nn.functional.adaptive_avg_pool2d(value, 1)
        return logits.detach(), value.flatten(1).detach()

    def close(self) -> None:
        self.handle.remove()


def parse_floats(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or tuple(sorted(values)) != values or any(value <= 0 for value in values):
        raise ValueError("epsilon values must be a positive ascending list")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-zoo-root", required=True)
    parser.add_argument("--model-zoo-source-root", default=None)
    parser.add_argument("--selected-samples", required=True,
                        help="selected_targeted_robust_samples.csv from the fixed Clean0 Probe run")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-aliases", default=",".join(DEFAULT_ALIASES))
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--epsilon-pixels", default=",".join(str(x) for x in DEFAULT_EPSILON_PIXELS))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--random-seed", type=int, default=20260913)
    return parser.parse_args()


def load_selected_indices(path: Path, target: int) -> list[int]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row.get("target", target)) == int(target):
                rows.append(row)
    rows.sort(key=lambda row: int(row.get("rank", len(rows))))
    indices = [int(row["sample_index"]) for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError(f"selection contains duplicate sample indices: {path}")
    if len(indices) != 100:
        raise ValueError(f"expected exactly 100 selected samples, found {len(indices)} in {path}")
    return indices


def target_margin(logits: torch.Tensor, target: int) -> torch.Tensor:
    other = logits.clone()
    other[:, int(target)] = float("-inf")
    return other.max(dim=1).values - logits[:, int(target)]


def concentration(vectors: np.ndarray) -> float | None:
    if len(vectors) < 2:
        return None
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    valid = norms[:, 0] > 1e-12
    if valid.sum() < 2:
        return None
    normalized = vectors[valid] / norms[valid]
    similarity = normalized @ normalized.T
    tri = np.triu_indices(len(normalized), k=1)
    return float(np.mean(similarity[tri])) if len(tri[0]) else None


def write_yaml(path: Path, payload: dict) -> None:
    import yaml

    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(path: Path | None) -> str | None:
    if path is None or not (path / ".git").exists():
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def model_zoo_provenance(root: Path, source_root: Path | None, aliases: tuple[str, ...]) -> dict:
    """Record public Model Zoo metadata without touching checkpoint files directly."""

    os.environ["MODEL_ZOO_ROOT"] = str(root)
    from modelzoo import get_model_info, list_models

    registry_candidates = [root / "registry.yaml", root / "registry.yml"]
    registry = next((path for path in registry_candidates if path.is_file()), None)
    if registry is None:
        matches = sorted(root.rglob("registry.yaml"))
        registry = matches[0] if matches else None
    try:
        package_version = importlib.metadata.version("backdoor-model-zoo")
    except importlib.metadata.PackageNotFoundError:
        package_version = None
    return {
        "package_version": package_version,
        "source_root": str(source_root) if source_root else None,
        "source_git_commit": git_commit(source_root),
        "registry_path": str(registry.resolve()) if registry else None,
        "registry_sha256": sha256_file(registry) if registry else None,
        "list_models": list_models(),
        "infos": {alias: get_model_info(alias) for alias in aliases},
    }


def main() -> None:
    args = parse_args()
    if args.target != 0:
        raise ValueError("this mechanism check is fixed to target=0")
    if args.steps <= 0 or args.restarts <= 0:
        raise ValueError("steps and restarts must be positive")
    epsilon_pixels = parse_floats(args.epsilon_pixels)
    aliases = tuple(item.strip() for item in args.model_aliases.split(",") if item.strip())
    if not aliases:
        raise ValueError("at least one Model Zoo alias is required")

    seed_everything(args.random_seed)
    device = torch.device(args.device)
    data_root = Path(args.data_root).expanduser().resolve()
    model_zoo_root = Path(args.model_zoo_root).expanduser().resolve()
    model_zoo_source_root = Path(args.model_zoo_source_root).expanduser().resolve() if args.model_zoo_source_root else None
    selection_path = Path(args.selected_samples).expanduser().resolve()
    if not selection_path.is_file():
        raise FileNotFoundError(selection_path)
    selected_indices = load_selected_indices(selection_path, args.target)
    dataset = cifar100_dataset(data_root, train=False)
    output = timestamp_run_dir(args.output_root, "target0_adv_concentration")
    provenance = model_zoo_provenance(model_zoo_root, model_zoo_source_root, aliases)

    config = vars(args).copy()
    config.update({
        "protocol": "stage1d-target0-fixed-top100-success-concentration-v1",
        "selected_sample_count": len(selected_indices),
        "selected_samples_file": str(selection_path),
        "epsilon_pixels": list(epsilon_pixels),
        "alpha_pixels": [float(eps / 10.0) for eps in epsilon_pixels],
        "model_aliases": list(aliases),
        "normalization": "CIFAR10 normalization applied once by load_modelzoo_classifier",
        "trigger_filtering": False,
        "concentration_definition": "mean pairwise cosine of feature deltas among eligible PGD-success samples at fixed epsilon",
        "model_zoo_provenance": provenance,
    })
    write_yaml(output / "config.resolved.yaml", config)
    write_csv(output / "selected_samples.csv", [
        {"rank": rank, "sample_index": index, "target": args.target, "source_file": str(selection_path)}
        for rank, index in enumerate(selected_indices, start=1)
    ])

    all_records: list[dict] = []
    metric_rows: list[dict] = []
    model_infos: dict[str, dict] = {}
    direction_arrays: dict[str, np.ndarray] = {}

    for alias_index, alias in enumerate(aliases):
        print(f"Loading Model Zoo alias: {alias}", flush=True)
        model, info = load_modelzoo_classifier(alias, model_zoo_root=str(model_zoo_root), device=device)
        model_infos[alias] = info
        extractor = AvgPoolFeatures(model)
        try:
            for epsilon_index, epsilon_pixels_value in enumerate(epsilon_pixels):
                seed_everything(args.random_seed + alias_index * 1000 + epsilon_index)
                eps = epsilon_pixels_value / 255.0
                success_directions: list[np.ndarray] = []
                eligible_count = 0
                success_count = 0
                for batch_indices, images, _labels in batch_images(
                    dataset, selected_indices, batch_size=args.batch_size, device=device
                ):
                    logits, clean_features = extractor(images)
                    original_prediction = logits.argmax(dim=1)
                    eligible = original_prediction.ne(args.target)
                    margin = target_margin(logits, args.target)
                    batch_success = torch.zeros(len(batch_indices), dtype=torch.bool, device=device)
                    batch_final_prediction = original_prediction.clone()
                    batch_linf = torch.full((len(batch_indices),), float("nan"), device=device)
                    batch_loss = torch.full((len(batch_indices),), float("nan"), device=device)
                    batch_adv_features = torch.full_like(clean_features, float("nan"))

                    eligible_positions = torch.nonzero(eligible, as_tuple=False).flatten()
                    eligible_count += int(eligible.sum().item())
                    if len(eligible_positions):
                        endpoint = targeted_pgd_endpoint(
                            model,
                            images[eligible_positions],
                            target=args.target,
                            epsilon=eps,
                            steps=args.steps,
                            alpha=eps / 10.0,
                            random_start=True,
                            restarts=args.restarts,
                        )
                        _endpoint_logits, endpoint_features = extractor(endpoint.endpoint)
                        batch_success[eligible_positions] = endpoint.success
                        batch_final_prediction[eligible_positions] = endpoint.endpoint_prediction
                        batch_linf[eligible_positions] = endpoint.endpoint_linf
                        batch_loss[eligible_positions] = endpoint.target_loss
                        batch_adv_features[eligible_positions] = endpoint_features
                        for position in range(len(eligible_positions)):
                            if bool(endpoint.success[position]):
                                index = int(eligible_positions[position])
                                success_directions.append(
                                    (endpoint_features[position] - clean_features[index]).cpu().numpy()
                                )
                        success_count += int(endpoint.success.sum().item())

                    for position, sample_index in enumerate(batch_indices):
                        all_records.append({
                            "model_alias": alias,
                            "target": args.target,
                            "sample_index": int(sample_index),
                            "epsilon_pixels": float(epsilon_pixels_value),
                            "original_prediction": int(original_prediction[position].item()),
                            "eligible": bool(eligible[position].item()),
                            "pgd_status": "success" if bool(batch_success[position]) else ("failure" if bool(eligible[position]) else "ineligible_original_target"),
                            "success": bool(batch_success[position].item()) if bool(eligible[position]) else None,
                            "final_prediction": int(batch_final_prediction[position].item()),
                            "actual_linf": float(batch_linf[position].item()) if bool(eligible[position]) else None,
                            "targeted_loss": float(batch_loss[position].item()) if bool(eligible[position]) else None,
                            "target_resistance_margin": float(margin[position].item()),
                            "feature_delta_norm": float(torch.linalg.vector_norm(batch_adv_features[position] - clean_features[position]).item()) if bool(eligible[position]) else None,
                        })

                directions = np.stack(success_directions) if success_directions else np.empty((0, 0), dtype=np.float32)
                direction_arrays[f"{alias}__eps{epsilon_pixels_value:g}"] = directions
                asr = success_count / eligible_count if eligible_count else None
                metric_rows.append({
                    "model_alias": alias,
                    "target": args.target,
                    "epsilon_pixels": float(epsilon_pixels_value),
                    "eligible_count": eligible_count,
                    "success_count": success_count,
                    "asr": asr,
                    "success_direction_count": len(directions),
                    "adv_concentration_success": concentration(directions),
                    "concentration_definition": "pairwise cosine among eligible PGD-success feature deltas",
                    "trigger_filtering": False,
                })
                print(
                    f"{alias} eps={epsilon_pixels_value:g}: eligible={eligible_count} "
                    f"success={success_count} ASR={asr if asr is not None else 'NA'} "
                    f"C_adv_success={metric_rows[-1]['adv_concentration_success']}",
                    flush=True,
                )
        finally:
            extractor.close()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_csv(output / "attack_records.csv", all_records)
    write_csv(output / "group_metrics.csv", metric_rows)
    np.savez_compressed(output / "success_direction_vectors.npz", **direction_arrays)

    try:
        import matplotlib.pyplot as plt

        figure = output / "figures" / "success_direction_concentration.png"
        plt.figure(figsize=(9, 5))
        for alias in aliases:
            values = [
                next(row["adv_concentration_success"] for row in metric_rows if row["model_alias"] == alias and row["epsilon_pixels"] == eps)
                for eps in epsilon_pixels
            ]
            plt.plot(epsilon_pixels, values, marker="o", label=alias)
        plt.xlabel("epsilon (pixels / 255)")
        plt.ylabel("C_adv among PGD-success samples")
        plt.xticks(epsilon_pixels)
        plt.grid(alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(figure, dpi=160)
        plt.close()
    except Exception as exc:  # pragma: no cover - plotting is supplementary
        print(f"WARNING: figure generation failed: {exc}", file=sys.stderr)

    summary = {
        **config,
        "selected_indices": selected_indices,
        "model_info": model_infos,
        "metrics": metric_rows,
        "outputs": {
            "attack_records": str((output / "attack_records.csv").resolve()),
            "group_metrics": str((output / "group_metrics.csv").resolve()),
            "direction_vectors": str((output / "success_direction_vectors.npz").resolve()),
        },
    }
    write_json(output / "summary.json", summary)
    print(f"Stage 1D concentration complete: {output.resolve()}")


if __name__ == "__main__":
    main()
