"""Train a target-0 Probe on CIFAR-10 and compare low-budget PGD ASR.

Clean seeds 1--3 provide the only Probe training labels.  Clean0 scores the
full CIFAR-10 test split and defines one shared Probe Top-500.  Every
Model-Zoo classifier is then evaluated on exactly those images at the same
targeted-PGD budgets.  No backdoor model or trigger is used during selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from mdluap.data import cifar10_dataset
from mdluap.models import load_modelzoo_classifier
from mdluap.probes import RidgeProbe, target_conditioned_logits_features, target_feature_names
from mdluap.targeted_pgd import targeted_pgd
from pilot_common import batch_images, seed_everything, timestamp_run_dir, write_csv, write_json


REFERENCE_SEEDS = (1, 2, 3)
MODEL_ALIASES = (
    "clean0",
    "badnet0",
    "blended0",
    "wanet0",
    "inputaware0",
    "ssba0",
    "adaptive_blend01",
)
TRAIN_EPSILON_PIXELS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0, 16.0, 32.0)
ANALYSIS_EPSILON_PIXELS = (1.0, 1.5)


def parse_floats(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or tuple(sorted(values)) != values or any(value <= 0 for value in values):
        raise ValueError("epsilon values must be a positive ascending list")
    return values


def parse_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("at least one integer is required")
    return values


def select_probe_topk(scores: np.ndarray, indices: list[int], top_k: int) -> list[int]:
    """Return the highest-scoring samples using stable descending ordering."""

    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) != len(indices):
        raise ValueError("scores and indices must be one-dimensional arrays of equal length")
    if top_k <= 0 or top_k > len(indices):
        raise ValueError("top_k must be within the score array length")
    positions = np.argsort(-scores, kind="mergesort")[: int(top_k)]
    return [int(indices[position]) for position in positions]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-zoo-root", required=True)
    parser.add_argument("--model-zoo-source-root", default=None)
    parser.add_argument("--output-root", default="results/stage1e_cifar10_probe_top500_asr")
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--train-count", type=int, default=1000)
    parser.add_argument("--train-eps-pixels", default=",".join(map(str, TRAIN_EPSILON_PIXELS)))
    parser.add_argument("--analysis-eps-pixels", default=",".join(map(str, ANALYSIS_EPSILON_PIXELS)))
    parser.add_argument("--train-steps", type=int, default=30)
    parser.add_argument("--analysis-steps", type=int, default=100)
    parser.add_argument("--analysis-restarts", type=int, default=3)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--split-seed", type=int, default=20260914)
    parser.add_argument("--random-seed", type=int, default=20260914)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    import yaml

    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def sha256_file(path: Path | None) -> str | None:
    if path is None or not path.is_file():
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


def model_zoo_provenance(root: Path, source_root: Path | None, aliases: tuple[str, ...]) -> dict[str, Any]:
    os.environ["MODEL_ZOO_ROOT"] = str(root)
    from modelzoo import get_model_info, list_models

    registry = next((root / name for name in ("registry.yaml", "registry.yml") if (root / name).is_file()), None)
    if registry is None:
        matches = sorted(root.rglob("registry.yaml")) if root.exists() else []
        registry = matches[0] if matches else None
    try:
        package_version = importlib.metadata.version("backdoor-model-zoo")
    except importlib.metadata.PackageNotFoundError:
        package_version = None
    infos = {alias: get_model_info(alias) for alias in aliases}
    return {
        "package_version": package_version,
        "source_root": str(source_root) if source_root else None,
        "source_git_commit": git_commit(source_root),
        "registry_path": str(registry.resolve()) if registry else None,
        "registry_sha256": sha256_file(registry),
        "list_models": list_models(),
        "infos": infos,
    }


@torch.inference_mode()
def logits_and_features(model, dataset, indices: list[int], *, target: int, batch_size: int, device: torch.device):
    logits_rows: list[torch.Tensor] = []
    feature_rows: list[np.ndarray] = []
    label_rows: list[int] = []
    for _batch_indices, images, labels in batch_images(dataset, indices, batch_size=batch_size, device=device):
        logits = model(images).detach()
        logits_rows.append(logits.cpu())
        feature_rows.append(target_conditioned_logits_features(logits, target).cpu().numpy())
        label_rows.extend(int(value) for value in labels.cpu().tolist())
    return torch.cat(logits_rows), np.concatenate(feature_rows), np.asarray(label_rows, dtype=np.int64)


def probe_labels(
    model,
    dataset,
    indices: list[int],
    *,
    target: int,
    epsilons: tuple[float, ...],
    steps: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]], torch.Tensor, np.ndarray]:
    """Create first-success target-PGD labels for one Clean reference model."""

    seed_everything(seed)
    logits, features, labels = logits_and_features(model, dataset, indices, target=target, batch_size=batch_size, device=device)
    before = logits.argmax(dim=1).numpy()
    radius = np.where(before == target, 0.0, np.inf).astype(np.float64)
    rows: list[dict[str, Any]] = []
    for epsilon_index, epsilon_pixels in enumerate(epsilons):
        seed_everything(seed + epsilon_index)
        success_values: list[bool] = []
        prediction_values: list[int] = []
        for _batch_indices, images, _batch_labels in batch_images(dataset, indices, batch_size=batch_size, device=device):
            result = targeted_pgd(
                model,
                images,
                target=target,
                epsilon=epsilon_pixels / 255.0,
                steps=steps,
                alpha=epsilon_pixels / 255.0 / 10.0,
                random_start=False,
                restarts=1,
            )
            success_values.extend(bool(value) for value in result.success.cpu().tolist())
            prediction_values.extend(int(value) for value in result.best_prediction.cpu().tolist())
        for position, sample_index in enumerate(indices):
            success = success_values[position]
            if success and not math.isfinite(radius[position]):
                radius[position] = epsilon_pixels
            rows.append({
                "reference_clean_seed": seed,
                "target": target,
                "sample_index": sample_index,
                "true_label": int(labels[position]),
                "epsilon_pixels": epsilon_pixels,
                "before_prediction": int(before[position]),
                "after_prediction": prediction_values[position],
                "success": success,
                "robustness_radius_pixels": None if not math.isfinite(radius[position]) else float(radius[position]),
                "censored": not math.isfinite(radius[position]),
                "steps": steps,
                "random_start": False,
                "restarts": 1,
            })
    return radius, rows, logits, features


def attack_selected(
    model,
    dataset,
    selected_indices: list[int],
    *,
    target: int,
    epsilon_pixels: float,
    steps: int,
    restarts: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[dict[int, dict[str, Any]], dict[int, int]]:
    """Run targeted PGD only on samples not initially predicted as target."""

    seed_everything(seed)
    original_predictions: dict[int, int] = {}
    with torch.no_grad():
        for batch_indices, images, _labels in batch_images(dataset, selected_indices, batch_size=batch_size, device=device):
            predictions = model(images).argmax(dim=1).cpu().tolist()
            original_predictions.update({int(index): int(prediction) for index, prediction in zip(batch_indices, predictions)})
    eligible = [index for index in selected_indices if original_predictions[index] != target]
    records: dict[int, dict[str, Any]] = {}
    for batch_indices, images, _labels in batch_images(dataset, eligible, batch_size=batch_size, device=device):
        result = targeted_pgd(
            model,
            images,
            target=target,
            epsilon=epsilon_pixels / 255.0,
            steps=steps,
            alpha=epsilon_pixels / 255.0 / 10.0,
            random_start=True,
            restarts=restarts,
        )
        for position, index in enumerate(batch_indices):
            records[int(index)] = {
                "success": bool(result.success[position].item()),
                "best_linf": float(result.best_linf[position].item()),
                "best_prediction": int(result.best_prediction[position].item()),
            }
    return records, original_predictions


def make_quality_rows(provenance: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for alias in MODEL_ALIASES:
        info = provenance["infos"][alias]
        rows.append({
            "model_alias": alias,
            "model_type": info.get("model_type"),
            "attack": info.get("attack"),
            "target_class": info.get("target_class"),
            "classifier_seed": info.get("classifier_seed"),
            "clean_acc": info.get("clean_acc"),
            "native_asr": info.get("asr"),
            "clean_trigger_asr": info.get("clean_trigger_asr"),
            "quality_gate": info.get("quality_gate"),
        })
    return rows


def plot_asr(output: Path, metric_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    aliases = list(MODEL_ALIASES)
    epsilons = sorted({float(row["epsilon_pixels"]) for row in metric_rows})
    fig, axes = plt.subplots(1, len(epsilons), figsize=(5 * len(epsilons), 5), squeeze=False)
    for axis, epsilon in zip(axes[0], epsilons):
        rows = [row for row in metric_rows if float(row["epsilon_pixels"]) == epsilon]
        values = {row["model_alias"]: float(row["asr_eligible"]) for row in rows}
        axis.bar(range(len(aliases)), [values.get(alias, float("nan")) for alias in aliases])
        axis.set_xticks(range(len(aliases)), aliases, rotation=55, ha="right")
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("targeted PGD ASR among eligible samples")
        axis.set_title(f"epsilon={epsilon}/255")
    fig.tight_layout()
    fig.savefig(output / "figures" / "pgd_asr_comparison.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.target != 0:
        raise ValueError("this experiment is fixed to target=0")
    if args.top_k <= 0 or args.train_count <= 0 or args.train_steps <= 0 or args.analysis_steps <= 0 or args.analysis_restarts <= 0:
        raise ValueError("top-k, train-count, steps, and restarts must be positive")
    train_epsilons = parse_floats(args.train_eps_pixels)
    analysis_epsilons = parse_floats(args.analysis_eps_pixels)
    if tuple(analysis_epsilons) != ANALYSIS_EPSILON_PIXELS:
        raise ValueError("analysis epsilon must be exactly 1,1.5 / 255")

    seed_everything(args.random_seed)
    device = torch.device(args.device)
    data_root = Path(args.data_root).expanduser().resolve()
    model_zoo_root = Path(args.model_zoo_root).expanduser().resolve()
    source_root = Path(args.model_zoo_source_root).expanduser().resolve() if args.model_zoo_source_root else None
    train_data = cifar10_dataset(data_root, train=True)
    test_data = cifar10_dataset(data_root, train=False)
    if args.train_count > len(train_data):
        raise ValueError(f"train-count {args.train_count} exceeds CIFAR-10 train size {len(train_data)}")
    if args.top_k > len(test_data):
        raise ValueError(f"top-k {args.top_k} exceeds CIFAR-10 test size {len(test_data)}")
    train_rng = np.random.default_rng(args.split_seed)
    train_indices = [int(value) for value in train_rng.permutation(len(train_data))[: args.train_count]]
    test_indices = list(range(len(test_data)))
    provenance = model_zoo_provenance(model_zoo_root, source_root, MODEL_ALIASES)
    output = timestamp_run_dir(args.output_root, "cifar10_probe_top500_asr")
    log_path = output / "run.log"

    def log(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    config = vars(args).copy()
    config.update({
        "protocol": "cifar10-clean123-probe-clean0-top500-target0-pgd-asr-v1",
        "reference_clean_seeds": list(REFERENCE_SEEDS),
        "train_pool": "CIFAR-10 train deterministic subset",
        "train_count": len(train_indices),
        "test_pool": "complete CIFAR-10 test split",
        "test_count": len(test_indices),
        "probe_selection": "Clean0 target-conditioned logits features, descending Ridge predicted radius",
        "probe_training_excludes": ["Clean0", "all backdoor models", "CIFAR-10 test PGD labels"],
        "analysis_epsilons_pixels": list(analysis_epsilons),
        "pgd": {"train_steps": args.train_steps, "analysis_steps": args.analysis_steps, "analysis_restarts": args.analysis_restarts, "random_start": True, "alpha_fraction": 0.1},
        "asr_definition": "success_count / eligible_count, where eligible means original_prediction != target",
        "model_zoo_provenance": provenance,
    })
    write_yaml(output / "config.resolved.yaml", config)
    write_csv(output / "candidate_pool.csv", [
        {"split": "cifar10_train", "sample_index": index, "pool_role": "probe_train"} for index in train_indices
    ] + [
        {"split": "cifar10_test", "sample_index": index, "pool_role": "probe_selection"} for index in test_indices
    ])

    log(f"CIFAR-10 Probe Top-500 experiment started: {output}")
    reference_features: list[np.ndarray] = []
    reference_labels: list[np.ndarray] = []
    probe_rows: list[dict[str, Any]] = []
    feature_names = target_feature_names(args.target, 10)
    for seed_index, seed in enumerate(REFERENCE_SEEDS):
        alias = f"clean{seed}"
        log(f"Loading Model Zoo alias: {alias} for Probe training")
        model, _info = load_modelzoo_classifier(alias, model_zoo_root=str(model_zoo_root), device=device)
        radius, rows, _logits, features = probe_labels(
            model,
            train_data,
            train_indices,
            target=args.target,
            epsilons=train_epsilons,
            steps=args.train_steps,
            batch_size=args.batch_size,
            device=device,
            seed=args.random_seed + 100 * seed_index,
        )
        finite = np.isfinite(radius)
        reference_features.append(features[finite])
        reference_labels.append(radius[finite])
        for row in rows:
            probe_rows.append(row)
        log(f"{alias}: finite Probe labels={int(finite.sum())}/{len(radius)}")
        del model

    pooled_features = np.concatenate(reference_features, axis=0)
    pooled_labels = np.concatenate(reference_labels, axis=0)
    probe = RidgeProbe.fit(pooled_features, pooled_labels, alpha=args.ridge_alpha, feature_names=feature_names)
    write_csv(output / "probe_training_records.csv", probe_rows)
    write_json(output / "probe_parameters.json", {
        "target": args.target,
        "reference_clean_aliases": [f"clean{seed}" for seed in REFERENCE_SEEDS],
        "train_sample_indices": train_indices,
        "finite_label_count": int(len(pooled_labels)),
        "feature_names": list(probe.feature_names),
        "feature_mean": probe.feature_mean.tolist(),
        "feature_std": probe.feature_std.tolist(),
        "weight": probe.weight.tolist(),
        "bias": float(probe.bias),
        "alpha": probe.alpha,
    })
    probe.save(output / "probe_target0.npz")

    log("Loading Model Zoo alias: clean0 for Top-500 selection")
    clean0, _clean0_info = load_modelzoo_classifier("clean0", model_zoo_root=str(model_zoo_root), device=device)
    test_logits, test_features, test_labels = logits_and_features(
        clean0, test_data, test_indices, target=args.target, batch_size=args.batch_size, device=device
    )
    scores = probe.predict(test_features)
    clean0_predictions = test_logits.argmax(dim=1).numpy()
    selected_indices = select_probe_topk(scores, test_indices, args.top_k)
    position_by_index = {index: position for position, index in enumerate(test_indices)}
    selected_positions = [position_by_index[index] for index in selected_indices]
    write_csv(output / "probe_selection_scores.csv", [
        {
            "sample_index": index,
            "true_label": int(test_labels[position]),
            "clean0_original_prediction": int(clean0_predictions[position]),
            "clean0_eligible": int(clean0_predictions[position]) != args.target,
            "probe_score": float(scores[position]),
            "selected_top500": position in set(selected_positions),
            "selection_rank": selected_positions.index(position) + 1 if position in set(selected_positions) else None,
        }
        for position, index in enumerate(test_indices)
    ])
    write_csv(output / "selected_probe_top500.csv", [
        {
            "rank": rank,
            "sample_index": index,
            "true_label": int(test_labels[position_by_index[index]]),
            "clean0_original_prediction": int(clean0_predictions[position_by_index[index]]),
            "clean0_eligible": int(clean0_predictions[position_by_index[index]]) != args.target,
            "probe_score": float(scores[position_by_index[index]]),
            "target": args.target,
        }
        for rank, index in enumerate(selected_indices, start=1)
    ])
    log(f"Clean0 selected Probe Top-{len(selected_indices)} from complete CIFAR-10 test")

    quality_rows = make_quality_rows(provenance)
    write_csv(output / "model_quality.csv", quality_rows)
    metric_rows: list[dict[str, Any]] = []
    attack_rows: list[dict[str, Any]] = []
    for alias_index, alias in enumerate(MODEL_ALIASES):
        log(f"Loading Model Zoo alias: {alias} for PGD ASR")
        model, _info = load_modelzoo_classifier(alias, model_zoo_root=str(model_zoo_root), device=device)
        for epsilon_index, epsilon_pixels in enumerate(analysis_epsilons):
            records, original_predictions = attack_selected(
                model,
                test_data,
                selected_indices,
                target=args.target,
                epsilon_pixels=epsilon_pixels,
                steps=args.analysis_steps,
                restarts=args.analysis_restarts,
                batch_size=args.batch_size,
                device=device,
                seed=args.random_seed + 10000 * alias_index + epsilon_index,
            )
            eligible_count = sum(original_predictions[index] != args.target for index in selected_indices)
            success_count = sum(records.get(index, {}).get("success", False) for index in selected_indices)
            metric_rows.append({
                "model_alias": alias,
                "epsilon_pixels": epsilon_pixels,
                "selected_count": len(selected_indices),
                "original_target_count": len(selected_indices) - eligible_count,
                "eligible_count": eligible_count,
                "success_count": success_count,
                "asr_eligible": float(success_count / eligible_count) if eligible_count else None,
                "asr_selected": float(success_count / len(selected_indices)),
            })
            for index in selected_indices:
                record = records.get(index)
                attack_rows.append({
                    "model_alias": alias,
                    "target": args.target,
                    "epsilon_pixels": epsilon_pixels,
                    "sample_index": index,
                    "original_prediction": original_predictions[index],
                    "eligible": original_predictions[index] != args.target,
                    "pgd_status": "ineligible_original_target" if original_predictions[index] == args.target else ("success" if record and record["success"] else "failure"),
                    "success": None if original_predictions[index] == args.target else bool(record and record["success"]),
                    "final_prediction": None if record is None else record["best_prediction"],
                    "actual_linf": None if record is None else record["best_linf"],
                    "actual_linf_pixels": None if record is None else record["best_linf"] * 255.0,
                })
            current = metric_rows[-1]
            log(f"{alias} eps={epsilon_pixels}: eligible={eligible_count} success={success_count} ASR={current['asr_eligible']}")
        del model

    write_csv(output / "attack_records.csv", attack_rows)
    write_csv(output / "group_metrics.csv", metric_rows)
    plot_asr(output, metric_rows)
    summary = {
        "protocol": config["protocol"],
        "output_directory": str(output),
        "target": args.target,
        "probe_reference_clean_aliases": [f"clean{seed}" for seed in REFERENCE_SEEDS],
        "selection_model": "clean0",
        "selected_count": len(selected_indices),
        "selected_indices": selected_indices,
        "metrics": metric_rows,
        "model_quality": quality_rows,
        "model_zoo_provenance": provenance,
        "asr_definition": config["asr_definition"],
    }
    write_json(output / "summary.json", summary)
    log(f"CIFAR-10 Probe Top-500 ASR experiment complete: {output}")


if __name__ == "__main__":
    main()
