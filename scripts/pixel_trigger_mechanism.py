"""CIFAR-10 pixel-space trigger/targeted-PGD mechanism experiment.

This runner deliberately does not use Probe scores or a precomputed robust
sample set.  Each backdoor family gets its own cohort: samples must have a
non-target ground-truth label, succeed under targeted PGD on Clean0 and the
corresponding backdoor model, and be classified as target 0 by that model
after its own official trigger is applied.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
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
from mdluap.official_triggers import TriggerAdapter, build_trigger_adapters
from mdluap.targeted_pgd import targeted_pgd_endpoint
from pilot_common import batch_images, seed_everything, timestamp_run_dir, write_csv, write_json


GROUPS: tuple[tuple[str, str, str], ...] = (
    ("badnet", "badnet0", "badnet"),
    ("blended", "blended0", "blended"),
    ("wanet", "wanet0", "wanet"),
    ("inputaware", "inputaware0", "inputaware"),
    ("adaptive_blend", "adaptive_blend01", "adaptive_blend"),
    ("ssba", "ssba0", "ssba"),
)
EPSILON_PIXELS: tuple[float, ...] = (1.0, 1.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-zoo-root", required=True)
    parser.add_argument("--model-zoo-source-root", default=None)
    parser.add_argument("--trigger-artifact-root", required=True)
    parser.add_argument("--backdoorbench-root", required=True)
    parser.add_argument("--output-root", default="results/stage1d_pixel_trigger_mechanism")
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--cohort-size", type=int, default=100)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--random-seed", type=int, default=20260913)
    parser.add_argument("--blended-alpha", type=float, default=0.2)
    parser.add_argument("--wanet-s", type=float, default=0.5)
    parser.add_argument("--wanet-grid-rescale", type=float, default=1.0)
    parser.add_argument("--badnet-trigger-path", default=None)
    parser.add_argument("--blended-trigger-path", default=None)
    parser.add_argument("--wanet-identity-path", default=None)
    parser.add_argument("--wanet-noise-path", default=None)
    parser.add_argument("--inputaware-state-path", default=None)
    parser.add_argument("--adaptive-blend-trigger-path", default=None)
    parser.add_argument("--ssba-encoder-path", default=None)
    parser.add_argument("--ssba-config-path", default=None)
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


def model_zoo_provenance(root: Path, source_root: Path | None, aliases: list[str]) -> dict[str, Any]:
    import os

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
    return {
        "package_version": package_version,
        "source_root": str(source_root) if source_root else None,
        "source_git_commit": git_commit(source_root),
        "registry_path": str(registry.resolve()) if registry else None,
        "registry_sha256": sha256_file(registry),
        "list_models": list_models(),
        "infos": {alias: get_model_info(alias) for alias in aliases},
    }


def parse_trigger_paths(args: argparse.Namespace) -> dict[str, Path | None]:
    return {
        "badnet": Path(args.badnet_trigger_path).expanduser().resolve() if args.badnet_trigger_path else None,
        "blended": Path(args.blended_trigger_path).expanduser().resolve() if args.blended_trigger_path else None,
        "wanet_identity": Path(args.wanet_identity_path).expanduser().resolve() if args.wanet_identity_path else None,
        "wanet_noise": Path(args.wanet_noise_path).expanduser().resolve() if args.wanet_noise_path else None,
        "inputaware": Path(args.inputaware_state_path).expanduser().resolve() if args.inputaware_state_path else None,
        "adaptive_blend": Path(args.adaptive_blend_trigger_path).expanduser().resolve() if args.adaptive_blend_trigger_path else None,
        "ssba_encoder": Path(args.ssba_encoder_path).expanduser().resolve() if args.ssba_encoder_path else None,
        "ssba_config": Path(args.ssba_config_path).expanduser().resolve() if args.ssba_config_path else None,
    }


def predictions_for_indices(model, dataset, indices: list[int], *, batch_size: int, device: torch.device) -> dict[int, int]:
    result: dict[int, int] = {}
    with torch.no_grad():
        for batch_indices, images, _labels in batch_images(dataset, indices, batch_size=batch_size, device=device):
            predictions = model(images).argmax(dim=1).detach().cpu().tolist()
            result.update({int(index): int(prediction) for index, prediction in zip(batch_indices, predictions)})
    return result


def trigger_predictions(
    model,
    adapter: TriggerAdapter,
    dataset,
    indices: list[int],
    *,
    split: str,
    batch_size: int,
    device: torch.device,
) -> dict[int, int]:
    result: dict[int, int] = {}
    with torch.no_grad():
        for batch_indices, images, _labels in batch_images(dataset, indices, batch_size=batch_size, device=device):
            triggered = adapter.apply(images, sample_indices=batch_indices, split=split)
            predictions = model(triggered).argmax(dim=1).detach().cpu().tolist()
            result.update({int(index): int(prediction) for index, prediction in zip(batch_indices, predictions)})
    return result


def attack_indices(
    model,
    dataset,
    indices: list[int],
    *,
    target: int,
    epsilon_pixels: float,
    steps: int,
    restarts: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[int, dict[str, Any]]:
    seed_everything(seed)
    output: dict[int, dict[str, Any]] = {}
    epsilon = float(epsilon_pixels) / 255.0
    alpha = epsilon / 10.0
    for batch_indices, images, _labels in batch_images(dataset, indices, batch_size=batch_size, device=device):
        result = targeted_pgd_endpoint(
            model,
            images,
            target=target,
            epsilon=epsilon,
            steps=steps,
            alpha=alpha,
            random_start=True,
            restarts=restarts,
        )
        endpoints = result.endpoint.detach().cpu().numpy().astype(np.float32)
        success = result.success.detach().cpu().numpy().astype(bool)
        endpoint_prediction = result.endpoint_prediction.detach().cpu().numpy().astype(np.int64)
        endpoint_linf = result.endpoint_linf.detach().cpu().numpy().astype(np.float32)
        target_loss = result.target_loss.detach().cpu().numpy().astype(np.float32)
        original = images.detach().cpu().numpy().astype(np.float32)
        l2 = np.sqrt(((endpoints - original) ** 2).reshape(len(batch_indices), -1).sum(axis=1)).astype(np.float32)
        for position, index in enumerate(batch_indices):
            output[int(index)] = {
                "endpoint": endpoints[position],
                "success": bool(success[position]),
                "endpoint_prediction": int(endpoint_prediction[position]),
                "actual_linf": float(endpoint_linf[position]),
                "actual_l2": float(l2[position]),
                "targeted_loss": float(target_loss[position]),
            }
    return output


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    numerator = float(np.dot(a.reshape(-1), b.reshape(-1)))
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return numerator / denominator if denominator > 1e-12 else float("nan")


def concentration(vectors: np.ndarray) -> float | None:
    if len(vectors) < 2:
        return None
    flat = vectors.reshape(len(vectors), -1).astype(np.float64)
    norms = np.linalg.norm(flat, axis=1)
    valid = norms > 1e-12
    if int(valid.sum()) < 2:
        return None
    normalized = flat[valid] / norms[valid, None]
    similarity = normalized @ normalized.T
    upper = np.triu_indices(len(normalized), k=1)
    return float(similarity[upper].mean()) if len(upper[0]) else None


def apply_trigger_batch(adapter: TriggerAdapter, dataset, indices: list[int], *, device: torch.device, batch_size: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    for batch_indices, images, _labels in batch_images(dataset, indices, batch_size=batch_size, device=device):
        triggered = adapter.apply(images, sample_indices=batch_indices, split="cifar10_test")
        rows.append(triggered.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(rows, axis=0)


def make_attack_row(group: str, alias: str, epsilon: float, index: int, label: int, original_prediction: int, attack: dict[str, Any]) -> dict[str, Any]:
    return {
        "backdoor_group": group,
        "model_alias": alias,
        "epsilon_pixels": epsilon,
        "sample_index": index,
        "true_label": label,
        "original_prediction": original_prediction,
        "eligible": original_prediction != 0,
        "pgd_status": "success" if attack["success"] else "failure",
        "success": attack["success"],
        "endpoint_prediction": attack["endpoint_prediction"],
        "actual_linf": attack["actual_linf"],
        "actual_linf_pixels": attack["actual_linf"] * 255.0,
        "actual_l2": attack["actual_l2"],
        "actual_l2_pixels": attack["actual_l2"] * 255.0,
        "targeted_loss": attack["targeted_loss"],
    }


def plot_metrics(output: Path, metrics: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    valid = [row for row in metrics if row.get("status") == "complete"]
    if not valid:
        return
    labels = [f"{row['backdoor_group']}\n{row['epsilon_pixels']}" for row in valid]
    positions = np.arange(len(valid))
    width = 0.35
    for filename, title, left, right in (
        ("asr_comparison.png", "Targeted PGD ASR on each independent cohort", "clean_asr", "backdoor_asr"),
        ("residual_concentration.png", "Pixel PGD residual concentration", "clean_adv_concentration", "backdoor_adv_concentration"),
        ("trigger_alignment.png", "Pixel PGD residual vs official trigger residual", "clean_trigger_alignment_mean", "backdoor_trigger_alignment_mean"),
    ):
        fig, axis = plt.subplots(figsize=(max(8, len(valid) * 1.5), 5))
        axis.bar(positions - width / 2, [row[left] for row in valid], width, label="Clean0")
        axis.bar(positions + width / 2, [row[right] for row in valid], width, label="Backdoor")
        axis.set_xticks(positions, labels, rotation=35, ha="right")
        axis.set_title(title)
        axis.set_ylim(-1.0, 1.0)
        axis.legend()
        fig.tight_layout()
        fig.savefig(output / "figures" / filename, dpi=160)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.target != 0:
        raise ValueError("this mechanism experiment is fixed to target=0")
    if args.cohort_size <= 0 or args.steps <= 0 or args.restarts <= 0 or args.batch_size <= 0:
        raise ValueError("cohort-size, steps, restarts, and batch-size must be positive")

    seed_everything(args.random_seed)
    device = torch.device(args.device)
    data_root = Path(args.data_root).expanduser().resolve()
    model_zoo_root = Path(args.model_zoo_root).expanduser().resolve()
    source_root = Path(args.model_zoo_source_root).expanduser().resolve() if args.model_zoo_source_root else None
    trigger_root = Path(args.trigger_artifact_root).expanduser().resolve()
    bdb_root = Path(args.backdoorbench_root).expanduser().resolve()
    dataset = cifar10_dataset(data_root, train=False)
    all_indices = list(range(len(dataset)))
    labels = {index: int(dataset[index][1]) for index in all_indices}
    candidate_indices = [index for index in all_indices if labels[index] != args.target]
    rng = np.random.default_rng(args.random_seed)
    shuffled_candidates = [int(value) for value in rng.permutation(candidate_indices)]

    aliases = ["clean0", *(alias for _group, alias, _trigger in GROUPS)]
    provenance = model_zoo_provenance(model_zoo_root, source_root, aliases)
    output = timestamp_run_dir(args.output_root, "pixel_trigger_mechanism")
    trigger_paths = parse_trigger_paths(args)
    adapters = build_trigger_adapters(
        model_root=trigger_root,
        backdoorbench_root=bdb_root,
        explicit=trigger_paths,
        device=device,
        blended_alpha=args.blended_alpha,
        wanet_s=args.wanet_s,
        wanet_grid_rescale=args.wanet_grid_rescale,
    )
    trigger_status = {
        name: {
            "trigger_type": adapter.status.trigger_type,
            "source": adapter.status.source,
            "available": adapter.status.available,
            "reason": adapter.status.reason,
        }
        for name, adapter in adapters.items()
    }

    config = vars(args).copy()
    config.update({
        "protocol": "stage1d-cifar10-pixel-trigger-mechanism-v1",
        "dataset": "CIFAR10 test",
        "target": 0,
        "candidate_count": len(shuffled_candidates),
        "candidate_order_seed": args.random_seed,
        "epsilon_attempts_pixels": list(EPSILON_PIXELS),
        "pgd": {"steps": args.steps, "restarts": args.restarts, "random_start": True, "alpha_fraction": 0.1},
        "selection_conditions": [
            "true_label != target",
            "Clean0 original_prediction != target and targeted PGD success",
            "corresponding backdoor original_prediction != target and targeted PGD success",
            "corresponding official trigger prediction == target on backdoor model",
        ],
        "cohort_policy": "one independent cohort per backdoor type; no cross-epsilon pooling",
        "pixel_residual_definition": "endpoint - original image, flattened as 3072 dimensions",
        "model_zoo_provenance": provenance,
        "trigger_status": trigger_status,
        "ssba_note": "SSBA uses the official encoder/provenance configuration; the documented tiny reproduction quantization discrepancy is retained as provenance.",
    })
    write_yaml(output / "config.resolved.yaml", config)
    write_csv(output / "candidate_pool.csv", [
        {
            "candidate_rank": rank,
            "sample_index": index,
            "true_label": labels[index],
            "excluded_true_target": labels[index] == args.target,
        }
        for rank, index in enumerate(shuffled_candidates, start=1)
    ])

    print(f"CIFAR-10 candidate count (true label != 0): {len(shuffled_candidates)}", flush=True)
    print(f"Output: {output}", flush=True)
    print("Loading Model Zoo alias: clean0", flush=True)
    clean_model, clean_info = load_modelzoo_classifier("clean0", model_zoo_root=str(model_zoo_root), device=device)
    clean_predictions = predictions_for_indices(clean_model, dataset, shuffled_candidates, batch_size=args.batch_size, device=device)
    clean_eligible = [index for index in shuffled_candidates if clean_predictions[index] != args.target]
    print(f"Clean0 eligible candidates: {len(clean_eligible)}", flush=True)

    model_quality = [{
        "model_alias": "clean0",
        "model_type": clean_info.get("model_type"),
        "attack": clean_info.get("attack"),
        "target_class": clean_info.get("target_class"),
        "classifier_seed": clean_info.get("classifier_seed"),
        "clean_acc": clean_info.get("clean_acc"),
        "asr": clean_info.get("asr"),
    }]
    for group, alias, _trigger_type in GROUPS:
        info = provenance["infos"].get(alias, {})
        model_quality.append({
            "model_alias": alias,
            "model_type": info.get("model_type"),
            "attack": info.get("attack"),
            "target_class": info.get("target_class"),
            "classifier_seed": info.get("classifier_seed"),
            "clean_acc": info.get("clean_acc"),
            "asr": info.get("asr"),
        })
    write_csv(output / "model_quality.csv", model_quality)

    clean_attack_cache: dict[float, dict[int, dict[str, Any]]] = {}
    metrics: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    attack_rows: list[dict[str, Any]] = []
    trigger_rows: list[dict[str, Any]] = []
    similarity_rows: list[dict[str, Any]] = []
    endpoint_arrays: dict[str, np.ndarray] = {}

    for group_index, (group, backdoor_alias, trigger_type) in enumerate(GROUPS):
        adapter = adapters[trigger_type]
        print(f"=== {group} ({backdoor_alias}) ===", flush=True)
        if not adapter.status.available:
            print(f"SKIP {group}: trigger unavailable: {adapter.status.reason}", flush=True)
            metrics.append({
                "backdoor_group": group,
                "backdoor_alias": backdoor_alias,
                "status": "trigger_unavailable",
                "epsilon_pixels": None,
                "selected_count": 0,
                "trigger_source": adapter.status.source,
                "trigger_unavailable_reason": adapter.status.reason,
            })
            continue

        print(f"Loading Model Zoo alias: {backdoor_alias}", flush=True)
        backdoor_model, _backdoor_info = load_modelzoo_classifier(
            backdoor_alias, model_zoo_root=str(model_zoo_root), device=device
        )
        bd_predictions = predictions_for_indices(backdoor_model, dataset, shuffled_candidates, batch_size=args.batch_size, device=device)
        bd_eligible = [index for index in shuffled_candidates if bd_predictions[index] != args.target]
        clean_trigger_predictions = trigger_predictions(
            clean_model, adapter, dataset, shuffled_candidates, split="cifar10_test", batch_size=args.batch_size, device=device
        )
        bd_trigger_predictions = trigger_predictions(
            backdoor_model, adapter, dataset, shuffled_candidates, split="cifar10_test", batch_size=args.batch_size, device=device
        )
        trigger_success_indices = [index for index in shuffled_candidates if bd_trigger_predictions[index] == args.target]

        selected: list[int] = []
        chosen_epsilon: float | None = None
        chosen_clean_attack: dict[int, dict[str, Any]] = {}
        chosen_bd_attack: dict[int, dict[str, Any]] = {}
        attempt_counts: list[dict[str, Any]] = []
        for epsilon_index, epsilon_pixels in enumerate(EPSILON_PIXELS):
            if epsilon_pixels not in clean_attack_cache:
                print(f"Running Clean0 PGD at epsilon={epsilon_pixels}/255", flush=True)
                clean_attack_cache[epsilon_pixels] = attack_indices(
                    clean_model,
                    dataset,
                    clean_eligible,
                    target=args.target,
                    epsilon_pixels=epsilon_pixels,
                    steps=args.steps,
                    restarts=args.restarts,
                    batch_size=args.batch_size,
                    device=device,
                    seed=args.random_seed + 1000 + epsilon_index,
                )
            clean_attack = clean_attack_cache[epsilon_pixels]
            bd_attack_indices = [index for index in shuffled_candidates if bd_predictions[index] != args.target]
            print(f"Running {backdoor_alias} PGD at epsilon={epsilon_pixels}/255 on {len(bd_attack_indices)} eligible samples", flush=True)
            bd_attack = attack_indices(
                backdoor_model,
                dataset,
                bd_attack_indices,
                target=args.target,
                epsilon_pixels=epsilon_pixels,
                steps=args.steps,
                restarts=args.restarts,
                batch_size=args.batch_size,
                device=device,
                seed=args.random_seed + 10000 * (group_index + 1) + 1000 + epsilon_index,
            )
            qualified = [
                index for index in shuffled_candidates
                if index in clean_attack
                and index in bd_attack
                and clean_attack[index]["success"]
                and bd_attack[index]["success"]
                and bd_trigger_predictions[index] == args.target
            ]
            selected = qualified[: args.cohort_size]
            attempt_counts.append({
                "epsilon_pixels": epsilon_pixels,
                "clean_eligible_count": len(clean_eligible),
                "backdoor_eligible_count": len(bd_eligible),
                "backdoor_trigger_success_count": len(trigger_success_indices),
                "qualified_count": len(qualified),
                "selected_count": len(selected),
            })
            for rank, index in enumerate(shuffled_candidates, start=1):
                clean_attack_row = clean_attack.get(index)
                bd_attack_row = bd_attack.get(index)
                selection_rows.append({
                    "backdoor_group": group,
                    "backdoor_alias": backdoor_alias,
                    "epsilon_pixels": epsilon_pixels,
                    "candidate_rank": rank,
                    "sample_index": index,
                    "true_label": labels[index],
                    "clean_original_prediction": clean_predictions[index],
                    "backdoor_original_prediction": bd_predictions[index],
                    "clean_eligible": clean_predictions[index] != args.target,
                    "backdoor_eligible": bd_predictions[index] != args.target,
                    "clean_success": clean_attack_row["success"] if clean_attack_row else False,
                    "backdoor_success": bd_attack_row["success"] if bd_attack_row else False,
                    "trigger_prediction_backdoor": bd_trigger_predictions[index],
                    "trigger_success_backdoor": bd_trigger_predictions[index] == args.target,
                    "qualified": index in qualified,
                    "selected": index in selected,
                })
            if len(selected) >= args.cohort_size:
                chosen_epsilon = epsilon_pixels
                chosen_clean_attack = clean_attack
                chosen_bd_attack = bd_attack
                break

        if chosen_epsilon is None:
            print(f"FAILED {group}: fewer than {args.cohort_size} qualified samples", flush=True)
            metrics.append({
                "backdoor_group": group,
                "backdoor_alias": backdoor_alias,
                "status": "insufficient_qualified_samples",
                "epsilon_pixels": None,
                "selected_count": len(selected),
                "attempt_counts": json.dumps(attempt_counts, ensure_ascii=False),
                "trigger_source": adapter.status.source,
            })
            del backdoor_model
            continue

        selected = selected[: args.cohort_size]
        print(f"Selected {len(selected)} samples at epsilon={chosen_epsilon}/255", flush=True)
        trigger_images = apply_trigger_batch(adapter, dataset, selected, device=device, batch_size=args.batch_size)
        original_images = np.stack([dataset[index][0].numpy().astype(np.float32) for index in selected])
        clean_endpoints = np.stack([chosen_clean_attack[index]["endpoint"] for index in selected])
        backdoor_endpoints = np.stack([chosen_bd_attack[index]["endpoint"] for index in selected])
        clean_adv_residual = clean_endpoints - original_images
        backdoor_adv_residual = backdoor_endpoints - original_images
        trigger_residual = trigger_images - original_images

        clean_alignments = np.asarray([cosine(clean_adv_residual[i], trigger_residual[i]) for i in range(len(selected))])
        backdoor_alignments = np.asarray([cosine(backdoor_adv_residual[i], trigger_residual[i]) for i in range(len(selected))])
        group_metrics = {
            "backdoor_group": group,
            "backdoor_alias": backdoor_alias,
            "status": "complete",
            "epsilon_pixels": chosen_epsilon,
            "selected_count": len(selected),
            "clean_eligible_count": len(clean_eligible),
            "backdoor_eligible_count": len(bd_eligible),
            "clean_asr": float(sum(chosen_clean_attack[index]["success"] for index in clean_eligible) / len(clean_eligible)) if clean_eligible else None,
            "backdoor_asr": float(sum(chosen_bd_attack[index]["success"] for index in bd_eligible) / len(bd_eligible)) if bd_eligible else None,
            "backdoor_trigger_success_rate_all_candidates": float(sum(bd_trigger_predictions[index] == args.target for index in shuffled_candidates) / len(shuffled_candidates)),
            "backdoor_trigger_success_rate_eligible": float(sum(bd_trigger_predictions[index] == args.target for index in bd_eligible) / len(bd_eligible)) if bd_eligible else None,
            "clean_trigger_success_rate_selected": float(sum(clean_trigger_predictions[index] == args.target for index in selected) / len(selected)),
            "backdoor_trigger_success_rate_selected": float(sum(bd_trigger_predictions[index] == args.target for index in selected) / len(selected)),
            "clean_adv_concentration": concentration(clean_adv_residual),
            "backdoor_adv_concentration": concentration(backdoor_adv_residual),
            "trigger_concentration": concentration(trigger_residual),
            "clean_trigger_alignment_mean": float(np.nanmean(clean_alignments)),
            "clean_trigger_alignment_median": float(np.nanmedian(clean_alignments)),
            "backdoor_trigger_alignment_mean": float(np.nanmean(backdoor_alignments)),
            "backdoor_trigger_alignment_median": float(np.nanmedian(backdoor_alignments)),
            "attempt_counts": json.dumps(attempt_counts, ensure_ascii=False),
            "trigger_source": adapter.status.source,
            "trigger_unavailable_reason": adapter.status.reason,
        }
        metrics.append(group_metrics)
        endpoint_arrays[f"{group}_sample_indices"] = np.asarray(selected, dtype=np.int64)
        endpoint_arrays[f"{group}_original"] = original_images
        endpoint_arrays[f"{group}_clean_adv"] = clean_endpoints
        endpoint_arrays[f"{group}_backdoor_adv"] = backdoor_endpoints
        endpoint_arrays[f"{group}_trigger"] = trigger_images

        for rank, index in enumerate(selected, start=1):
            selected_rows.append({
                "backdoor_group": group,
                "backdoor_alias": backdoor_alias,
                "trigger_type": trigger_type,
                "epsilon_pixels": chosen_epsilon,
                "cohort_rank": rank,
                "sample_index": index,
                "true_label": labels[index],
                "clean_original_prediction": clean_predictions[index],
                "backdoor_original_prediction": bd_predictions[index],
                "trigger_prediction_clean": clean_trigger_predictions[index],
                "trigger_prediction_backdoor": bd_trigger_predictions[index],
                "trigger_success_backdoor": bd_trigger_predictions[index] == args.target,
            })
            attack_rows.append(make_attack_row(group, "clean0", chosen_epsilon, index, labels[index], clean_predictions[index], chosen_clean_attack[index]))
            attack_rows.append(make_attack_row(group, backdoor_alias, chosen_epsilon, index, labels[index], bd_predictions[index], chosen_bd_attack[index]))
            trigger_rows.append({
                "backdoor_group": group,
                "trigger_type": trigger_type,
                "model_alias": "clean0",
                "epsilon_pixels": chosen_epsilon,
                "sample_index": index,
                "trigger_prediction": clean_trigger_predictions[index],
                "trigger_success_true_target": clean_trigger_predictions[index] == args.target,
                "trigger_source": adapter.status.source,
            })
            trigger_rows.append({
                "backdoor_group": group,
                "trigger_type": trigger_type,
                "model_alias": backdoor_alias,
                "epsilon_pixels": chosen_epsilon,
                "sample_index": index,
                "trigger_prediction": bd_trigger_predictions[index],
                "trigger_success_true_target": bd_trigger_predictions[index] == args.target,
                "trigger_source": adapter.status.source,
            })
            similarity_rows.append({
                "backdoor_group": group,
                "backdoor_alias": backdoor_alias,
                "epsilon_pixels": chosen_epsilon,
                "sample_index": index,
                "clean_trigger_alignment": float(clean_alignments[rank - 1]),
                "backdoor_trigger_alignment": float(backdoor_alignments[rank - 1]),
                "clean_adv_l2": chosen_clean_attack[index]["actual_l2"],
                "backdoor_adv_l2": chosen_bd_attack[index]["actual_l2"],
            })
        del backdoor_model

    np.savez_compressed(output / "endpoint_arrays.npz", **endpoint_arrays)
    write_csv(output / "cohort_selection_records.csv", selection_rows)
    write_csv(output / "selected_cohorts.csv", selected_rows)
    write_csv(output / "attack_records.csv", attack_rows)
    write_csv(output / "trigger_records.csv", trigger_rows)
    write_csv(output / "pixel_similarity_records.csv", similarity_rows)
    write_csv(output / "group_metrics.csv", metrics)
    plot_metrics(output, metrics)

    summary = {
        "protocol": config["protocol"],
        "output_directory": str(output),
        "target": args.target,
        "candidate_order_seed": args.random_seed,
        "candidate_count_true_label_not_target": len(shuffled_candidates),
        "cohort_size_requested": args.cohort_size,
        "epsilon_attempts_pixels": list(EPSILON_PIXELS),
        "groups": metrics,
        "trigger_status": trigger_status,
        "model_quality": model_quality,
        "model_zoo_provenance": provenance,
        "model_infos": provenance["infos"],
        "ssba_note": config["ssba_note"],
        "selection_conditions": config["selection_conditions"],
    }
    write_json(output / "summary.json", summary)
    print(f"Pixel trigger mechanism experiment complete: {output}", flush=True)


if __name__ == "__main__":
    main()
