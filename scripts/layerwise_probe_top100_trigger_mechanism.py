"""Probe-selected CIFAR-10 Top-100 layerwise trigger-path analysis.

Clean1--3 train the target-0 Probe, Clean0 scores the complete CIFAR-10 test
split and selects one shared Top-100, then Clean0 and BadNet0 are attacked on
exactly those images.  The resulting endpoints are analyzed layer by layer.
This is a parallel protocol to the fixed joint-success cohort experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mdluap.data import cifar10_dataset
from mdluap.models import load_modelzoo_classifier
from mdluap.official_triggers import build_trigger_adapters
from mdluap.probes import RidgeProbe, target_feature_names
from mdluap.targeted_pgd import targeted_pgd_endpoint
from pilot_common import batch_images, seed_everything, timestamp_run_dir, write_csv, write_json
from probe_cifar10_top500_asr import logits_and_features, probe_labels, select_probe_topk
from layerwise_trigger_mechanism import (
    LAYERS,
    concentration,
    cosine_rows,
    model_feature_residuals,
    model_zoo_provenance,
    plot_curves,
)


REFERENCE_SEEDS = (1, 2, 3)
TRAIN_EPSILON_PIXELS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0, 16.0, 32.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-zoo-root", required=True)
    parser.add_argument("--model-zoo-source-root", default=None)
    parser.add_argument("--backdoorbench-root", required=True)
    parser.add_argument("--trigger-artifact-root", default=None)
    parser.add_argument("--backdoor-alias", default="badnet0", choices=("badnet0", "wanet0", "ssba0"))
    parser.add_argument("--badnet-trigger-path", default=None)
    parser.add_argument("--wanet-state-path", default=None)
    parser.add_argument("--ssba-encoder-path", default=None)
    parser.add_argument("--ssba-config-path", default=None)
    parser.add_argument("--output-root", default="results/stage1d_layerwise_probe_top100")
    parser.add_argument("--selection-file", default=None, help="Existing selected_probe_top100.csv; skips Probe training and selection.")
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--train-count", type=int, default=1000)
    parser.add_argument("--train-eps-pixels", default=",".join(map(str, TRAIN_EPSILON_PIXELS)))
    parser.add_argument("--train-steps", type=int, default=30)
    parser.add_argument("--analysis-epsilon-pixels", type=float, default=1.0)
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


def load_selection_file(path: Path, top_k: int) -> tuple[list[int], list[dict[str, Any]]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != top_k:
        raise ValueError(f"selection file must contain exactly {top_k} rows, got {len(rows)}")
    if not rows or "sample_index" not in rows[0]:
        raise ValueError("selection file must contain a sample_index column")
    indices = [int(row["sample_index"]) for row in rows]
    if len(set(indices)) != len(indices):
        raise ValueError("selection file contains duplicate sample indices")
    return indices, rows


def apply_trigger(adapter, dataset, indices: list[int], *, batch_size: int, device: torch.device) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for batch_indices, images, _labels in batch_images(dataset, indices, batch_size=batch_size, device=device):
        triggered = adapter.apply(images, sample_indices=batch_indices, split="cifar10_test")
        outputs.append(triggered.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def attack_endpoints(
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
) -> tuple[dict[int, dict[str, Any]], dict[int, int]]:
    seed_everything(seed)
    predictions: dict[int, int] = {}
    records: dict[int, dict[str, Any]] = {}
    with torch.no_grad():
        for batch_indices, images, _labels in batch_images(dataset, indices, batch_size=batch_size, device=device):
            predictions.update({int(index): int(value) for index, value in zip(batch_indices, model(images).argmax(dim=1).cpu().tolist())})
    for batch_indices, images, _labels in batch_images(dataset, indices, batch_size=batch_size, device=device):
        result = targeted_pgd_endpoint(
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
            endpoint = result.endpoint[position].cpu().numpy().astype(np.float32)
            original = images[position].cpu().numpy().astype(np.float32)
            records[int(index)] = {
                "endpoint": endpoint,
                "success": bool(result.success[position].item()),
                "best_prediction": int(result.endpoint_prediction[position].item()),
                "targeted_loss": float(result.target_loss[position].item()),
                "actual_linf": float(result.endpoint_linf[position].item()),
                "actual_linf_pixels": float(result.endpoint_linf[position].item() * 255.0),
                "actual_l2": float(np.linalg.norm((endpoint - original).reshape(-1))),
            }
    return records, predictions


def metric_rows(
    residuals: dict[str, dict[str, dict[str, np.ndarray]]],
    sample_indices: np.ndarray,
    backdoor_alias: str = "badnet0",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    for layer in LAYERS:
        clean_adv = residuals["clean0"]["adv"][layer]
        clean_trigger = residuals["clean0"]["trigger"][layer]
        bad_adv = residuals[backdoor_alias]["adv"][layer]
        bad_trigger = residuals[backdoor_alias]["trigger"][layer]
        clean_alignment = cosine_rows(clean_adv, clean_trigger)
        bad_alignment = cosine_rows(bad_adv, bad_trigger)
        clean_adv_l2 = np.linalg.norm(clean_adv, axis=1)
        bad_adv_l2 = np.linalg.norm(bad_adv, axis=1)
        clean_trigger_l2 = np.linalg.norm(clean_trigger, axis=1)
        bad_trigger_l2 = np.linalg.norm(bad_trigger, axis=1)
        clean_adv_concentration = concentration(clean_adv)
        bad_adv_concentration = concentration(bad_adv)
        clean_trigger_concentration = concentration(clean_trigger)
        bad_trigger_concentration = concentration(bad_trigger)
        clean_mean = float(np.nanmean(clean_alignment))
        bad_mean = float(np.nanmean(bad_alignment))
        metrics.append({
            "layer": layer,
            "sample_count": int(len(sample_indices)),
            "clean_alignment_mean": clean_mean,
            "clean_alignment_median": float(np.nanmedian(clean_alignment)),
            "badnet_alignment_mean": bad_mean,
            "badnet_alignment_median": float(np.nanmedian(bad_alignment)),
            "alignment_gap_badnet_minus_clean": bad_mean - clean_mean,
            "clean_adv_concentration": clean_adv_concentration,
            "badnet_adv_concentration": bad_adv_concentration,
            "adv_concentration_gap_badnet_minus_clean": None if clean_adv_concentration is None or bad_adv_concentration is None else bad_adv_concentration - clean_adv_concentration,
            "clean_trigger_concentration": clean_trigger_concentration,
            "badnet_trigger_concentration": bad_trigger_concentration,
            "trigger_concentration_gap_badnet_minus_clean": None if clean_trigger_concentration is None or bad_trigger_concentration is None else bad_trigger_concentration - clean_trigger_concentration,
            "clean_adv_l2_mean": float(clean_adv_l2.mean()),
            "badnet_adv_l2_mean": float(bad_adv_l2.mean()),
            "clean_trigger_l2_mean": float(clean_trigger_l2.mean()),
            "badnet_trigger_l2_mean": float(bad_trigger_l2.mean()),
            "clean_adv_zero_norm_count": int((clean_adv_l2 <= 1e-12).sum()),
            "badnet_adv_zero_norm_count": int((bad_adv_l2 <= 1e-12).sum()),
            "clean_trigger_zero_norm_count": int((clean_trigger_l2 <= 1e-12).sum()),
            "badnet_trigger_zero_norm_count": int((bad_trigger_l2 <= 1e-12).sum()),
        })
        for position, sample_index in enumerate(sample_indices):
            samples.extend((
                {"model_alias": "clean0", "layer": layer, "sample_index": int(sample_index), "adv_alignment": float(clean_alignment[position]), "adv_delta_l2": float(clean_adv_l2[position]), "trigger_delta_l2": float(clean_trigger_l2[position]), "adv_zero_norm": bool(clean_adv_l2[position] <= 1e-12), "trigger_zero_norm": bool(clean_trigger_l2[position] <= 1e-12)},
                {"model_alias": backdoor_alias, "layer": layer, "sample_index": int(sample_index), "adv_alignment": float(bad_alignment[position]), "adv_delta_l2": float(bad_adv_l2[position]), "trigger_delta_l2": float(bad_trigger_l2[position]), "adv_zero_norm": bool(bad_adv_l2[position] <= 1e-12), "trigger_zero_norm": bool(bad_trigger_l2[position] <= 1e-12)},
            ))
    return metrics, samples


def main() -> None:
    args = parse_args()
    if args.target != 0 or args.top_k != 100:
        raise ValueError("this protocol is fixed to target=0 and Top-100")
    if min(args.train_count, args.train_steps, args.analysis_steps, args.analysis_restarts, args.batch_size) <= 0:
        raise ValueError("counts, steps, restarts, and batch size must be positive")
    if args.analysis_epsilon_pixels <= 0:
        raise ValueError("analysis epsilon must be positive")
    train_epsilons = tuple(float(value) for value in args.train_eps_pixels.split(",") if value.strip())
    seed_everything(args.random_seed)
    device = torch.device(args.device)
    data_root = Path(args.data_root).expanduser().resolve()
    model_zoo_root = Path(args.model_zoo_root).expanduser().resolve()
    source_root = Path(args.model_zoo_source_root).expanduser().resolve() if args.model_zoo_source_root else None
    backdoorbench_root = Path(args.backdoorbench_root).expanduser().resolve()
    trigger_artifact_root = Path(args.trigger_artifact_root).expanduser().resolve() if args.trigger_artifact_root else backdoorbench_root
    train_data = None if args.selection_file else cifar10_dataset(data_root, train=True)
    test_data = cifar10_dataset(data_root, train=False)
    if (train_data is not None and args.train_count > len(train_data)) or args.top_k > len(test_data):
        raise ValueError("requested pool size exceeds CIFAR-10 split size")
    train_rng = np.random.default_rng(args.split_seed)
    train_indices = [int(value) for value in train_rng.permutation(len(train_data))[:args.train_count]] if train_data is not None else []
    test_indices = list(range(len(test_data)))
    aliases = ("clean0", "clean1", "clean2", "clean3", args.backdoor_alias)
    provenance = model_zoo_provenance(model_zoo_root, source_root, aliases)
    output = timestamp_run_dir(args.output_root, "layerwise_probe_top100")
    log_path = output / "run.log"

    def log(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    config = vars(args).copy()
    config.update({
        "protocol": f"stage1d-layerwise-clean123-probe-clean0-top100-clean0-{args.backdoor_alias}-v1",
        "probe_reference_clean_aliases": [f"clean{seed}" for seed in REFERENCE_SEEDS],
        "selection_model": "clean0",
        "models": ["clean0", args.backdoor_alias],
        "selection_pool": "complete CIFAR-10 test split",
        "train_pool": "deterministic CIFAR-10 train subset" if args.selection_file is None else None,
        "probe_selection_mode": "reuse_existing_selection" if args.selection_file else "fit_clean123_and_select_clean0",
        "selection_file": str(Path(args.selection_file).expanduser().resolve()) if args.selection_file else None,
        "pgd_reused_without_rerun": False,
        "pgd": {"target": args.target, "epsilon_pixels": args.analysis_epsilon_pixels, "steps": args.analysis_steps, "restarts": args.analysis_restarts, "random_start": True, "alpha_fraction": 0.1},
        "feature_definition": "h_l(x_adv)-h_l(x) and h_l(T(x))-h_l(x); convolution maps flattened without pooling",
        "model_zoo_provenance": provenance,
    })
    write_yaml(output / "config.resolved.yaml", config)
    candidate_rows = ([{"split": "cifar10_train", "sample_index": index, "pool_role": "probe_train"} for index in train_indices] if train_data is not None else []) + [{"split": "cifar10_test", "sample_index": index, "pool_role": "probe_selection"} for index in test_indices]
    write_csv(output / "candidate_pool.csv", candidate_rows)
    log(f"Probe Top-100 layerwise experiment started: {output}")

    if args.selection_file:
        selection_path = Path(args.selection_file).expanduser().resolve()
        if not selection_path.is_file():
            raise FileNotFoundError(selection_path)
        selected_indices, selected_rows = load_selection_file(selection_path, args.top_k)
        if min(selected_indices) < 0 or max(selected_indices) >= len(test_data):
            raise ValueError("selection file contains an index outside the CIFAR-10 test split")
        write_csv(output / "selected_probe_top100.csv", selected_rows)
        scores_path = selection_path.parent / "probe_selection_scores.csv"
        if scores_path.is_file():
            with scores_path.open("r", newline="", encoding="utf-8") as handle:
                write_csv(output / "probe_selection_scores.csv", list(csv.DictReader(handle)))
        metadata_path = selection_path.parent / "selection_metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            write_json(output / "probe_selection_source.json", metadata)
            probe_archive = metadata.get("probe_archive")
            if probe_archive and Path(probe_archive).is_file():
                shutil.copy2(probe_archive, output / "probe_target0.npz")
            for artifact_name in ("probe_training_records.csv", "probe_parameters.json"):
                source_artifact = selection_path.parent / artifact_name
                if source_artifact.is_file():
                    shutil.copy2(source_artifact, output / artifact_name)
            write_json(output / "probe_reuse_metadata.json", {
                "mode": "reused_existing_probe_and_clean0_selection",
                "selection_file": str(selection_path),
                "selection_metadata": str(metadata_path),
                "probe_archive": probe_archive,
                "probe_archive_sha256": metadata.get("probe_archive_sha256"),
            })
        log(f"Reusing existing Probe Top-{len(selected_indices)} selection: {selection_path}")
    else:
        reference_features: list[np.ndarray] = []
        reference_labels: list[np.ndarray] = []
        probe_records: list[dict[str, Any]] = []
        for seed_index, seed in enumerate(REFERENCE_SEEDS):
            alias = f"clean{seed}"
            model, _info = load_modelzoo_classifier(alias, model_zoo_root=str(model_zoo_root), device=device)
            radius, rows, _logits, features = probe_labels(model, train_data, train_indices, target=args.target, epsilons=train_epsilons, steps=args.train_steps, batch_size=args.batch_size, device=device, seed=args.random_seed + 100 * seed_index)
            finite = np.isfinite(radius)
            reference_features.append(features[finite])
            reference_labels.append(radius[finite])
            probe_records.extend(rows)
            log(f"{alias}: finite Probe labels={int(finite.sum())}/{len(radius)}")
            del model
        pooled_features = np.concatenate(reference_features, axis=0)
        pooled_labels = np.concatenate(reference_labels, axis=0)
        probe = RidgeProbe.fit(pooled_features, pooled_labels, alpha=args.ridge_alpha, feature_names=target_feature_names(args.target, 10))
        write_csv(output / "probe_training_records.csv", probe_records)
        write_json(output / "probe_parameters.json", {"target": args.target, "reference_clean_aliases": [f"clean{seed}" for seed in REFERENCE_SEEDS], "train_sample_indices": train_indices, "finite_label_count": int(len(pooled_labels)), "feature_names": list(probe.feature_names), "feature_mean": probe.feature_mean.tolist(), "feature_std": probe.feature_std.tolist(), "weight": probe.weight.tolist(), "bias": float(probe.bias), "alpha": probe.alpha})
        probe.save(output / "probe_target0.npz")
        clean0, _info = load_modelzoo_classifier("clean0", model_zoo_root=str(model_zoo_root), device=device)
        test_logits, test_features, test_labels = logits_and_features(clean0, test_data, test_indices, target=args.target, batch_size=args.batch_size, device=device)
        scores = probe.predict(test_features)
        predictions = test_logits.argmax(dim=1).numpy()
        selected_indices = select_probe_topk(scores, test_indices, args.top_k)
        positions = {index: position for position, index in enumerate(test_indices)}
        selected_set = set(selected_indices)
        write_csv(output / "probe_selection_scores.csv", [{"sample_index": index, "true_label": int(test_labels[position]), "clean0_original_prediction": int(predictions[position]), "clean0_eligible": int(predictions[position]) != args.target, "probe_score": float(scores[position]), "selected_top100": index in selected_set, "selection_rank": selected_indices.index(index) + 1 if index in selected_set else None} for position, index in enumerate(test_indices)])
        write_csv(output / "selected_probe_top100.csv", [{"rank": rank, "sample_index": index, "true_label": int(test_labels[positions[index]]), "clean0_original_prediction": int(predictions[positions[index]]), "clean0_eligible": int(predictions[positions[index]]) != args.target, "probe_score": float(scores[positions[index]]), "target": args.target} for rank, index in enumerate(selected_indices, start=1)])
        log(f"Clean0 selected Probe Top-{len(selected_indices)}")

    trigger_type = {"badnet0": "badnet", "wanet0": "wanet", "ssba0": "ssba"}[args.backdoor_alias]
    explicit = {
        "badnet": Path(args.badnet_trigger_path).expanduser().resolve() if args.badnet_trigger_path else None,
        "wanet_identity": None,
        "wanet_noise": None,
        "ssba_encoder": Path(args.ssba_encoder_path).expanduser().resolve() if args.ssba_encoder_path else None,
        "ssba_config": Path(args.ssba_config_path).expanduser().resolve() if args.ssba_config_path else None,
        "blended": None,
        "inputaware": None,
        "adaptive_blend": None,
    }
    if args.wanet_state_path:
        state_path = Path(args.wanet_state_path).expanduser().resolve()
        if state_path.name == "state_identity_grid.pt":
            explicit["wanet_identity"] = state_path
        elif state_path.name == "state_noise_grid.pt":
            explicit["wanet_noise"] = state_path
    adapter_map = build_trigger_adapters(model_root=trigger_artifact_root, backdoorbench_root=backdoorbench_root, explicit=explicit, device=device, blended_alpha=0.2, wanet_s=0.5, wanet_grid_rescale=1.0)
    adapter = adapter_map[trigger_type]
    if not adapter.status.available:
        raise RuntimeError(f"BadNet trigger unavailable: {adapter.status.reason}")
    trigger_images = apply_trigger(adapter, test_data, selected_indices, batch_size=args.batch_size, device=device)
    original_images = np.stack([test_data[index][0].numpy().astype(np.float32) for index in selected_indices])

    attack_records: list[dict[str, Any]] = []
    endpoints: dict[str, np.ndarray] = {}
    original_predictions: dict[str, dict[int, int]] = {}
    for alias_index, alias in enumerate(("clean0", args.backdoor_alias)):
        model, _info = load_modelzoo_classifier(alias, model_zoo_root=str(model_zoo_root), device=device)
        records, predictions_map = attack_endpoints(model, test_data, selected_indices, target=args.target, epsilon_pixels=args.analysis_epsilon_pixels, steps=args.analysis_steps, restarts=args.analysis_restarts, batch_size=args.batch_size, device=device, seed=args.random_seed + 10000 * alias_index)
        endpoints[alias] = np.stack([records[index]["endpoint"] for index in selected_indices])
        original_predictions[alias] = predictions_map
        for index in selected_indices:
            record = records[index]
            attack_records.append({"model_alias": alias, "target": args.target, "epsilon_pixels": args.analysis_epsilon_pixels, "sample_index": index, "original_prediction": predictions_map[index], "eligible": predictions_map[index] != args.target, "success": record["success"], "pgd_status": "ineligible_original_target" if predictions_map[index] == args.target else ("success" if record["success"] else "failure"), "final_prediction": record["best_prediction"], "targeted_loss": record["targeted_loss"], "actual_linf": record["actual_linf"], "actual_linf_pixels": record["actual_linf_pixels"], "actual_l2": record["actual_l2"]})
        log(f"{alias}: success={sum(records[index]['success'] for index in selected_indices)}/{len(selected_indices)}")
        del model

    residuals: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for alias in ("clean0", args.backdoor_alias):
        model, _info = load_modelzoo_classifier(alias, model_zoo_root=str(model_zoo_root), device=device)
        adv_values, trigger_values = model_feature_residuals(model, original_images, endpoints[alias], trigger_images, batch_size=args.batch_size, device=device)
        residuals[alias] = {"adv": adv_values, "trigger": trigger_values}
        del model
    metrics, sample_rows = metric_rows(residuals, np.asarray(selected_indices, dtype=np.int64), args.backdoor_alias)
    write_csv(output / "attack_records.csv", attack_records)
    write_csv(output / "layer_metrics.csv", metrics)
    write_csv(output / "per_sample_layer_records.csv", sample_rows)
    np.savez_compressed(output / "endpoint_arrays.npz", sample_indices=np.asarray(selected_indices, dtype=np.int64), original=original_images, clean0_adv=endpoints["clean0"], backdoor_adv=endpoints[args.backdoor_alias], backdoor_trigger=trigger_images)
    write_csv(output / "model_quality.csv", [{"model_alias": alias, "model_type": provenance["infos"][alias].get("model_type"), "attack": provenance["infos"][alias].get("attack"), "classifier_seed": provenance["infos"][alias].get("classifier_seed"), "clean_acc": provenance["infos"][alias].get("clean_acc"), "native_asr": provenance["infos"][alias].get("asr")} for alias in aliases])
    plot_curves(output, metrics)
    write_json(output / "summary.json", {"protocol": config["protocol"], "output_directory": str(output), "selected_count": len(selected_indices), "selected_indices": selected_indices, "models": ["clean0", args.backdoor_alias], "backdoor_alias": args.backdoor_alias, "trigger_type": trigger_type, "trigger_source": adapter.status.source, "metrics": metrics, "model_zoo_provenance": provenance, "attack_success_counts": {alias: sum(row["success"] for row in attack_records if row["model_alias"] == alias) for alias in ("clean0", args.backdoor_alias)}, "interpretation_boundary": "Probe-selected Top-100 representation-direction alignment and concentration; not a causal proof."})
    log(f"Probe Top-100 layerwise experiment complete: {output}")


if __name__ == "__main__":
    main()
