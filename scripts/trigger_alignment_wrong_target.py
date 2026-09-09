"""Stage 1D-WT: wrong-target PGD and official trigger-direction analysis.

Clean seeds 1--3 train one Ridge Probe for each wrong target.  Clean seed 0
uses each Probe to select one shared CIFAR-100 Top-100 set.  That same set is
then sent to Clean0 and each seed-0 backdoor model so the mechanism comparison
is paired by image.  The attack target is 1, 3, or 7, while the true backdoor
target remains 0.

This script deliberately does not claim a literal class-transition path.  It
tests whether a successful wrong-target PGD endpoint has a feature change
direction similar to the official trigger transformation for that attack.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mdluap.data import cifar100_dataset
from mdluap.models import load_backdoor_toolbox_resnet18
from mdluap.official_triggers import build_trigger_adapters
from mdluap.probes import RidgeProbe, target_conditioned_logits_features, target_feature_names, target_margin
from mdluap.targeted_pgd import targeted_pgd, targeted_pgd_endpoint
from pilot_common import batch_images, load_model, timestamp_run_dir, write_csv, write_json


TRAIN_EPS_PIXELS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0, 16.0, 32.0)
ANALYSIS_EPS_PIXELS = (1.0, 1.5)
DEFAULT_TARGETS = (1, 3, 7)
BACKDOOR_GROUPS = ("badnet", "blended", "wanet", "inputaware", "adaptive_blend")


class AvgPoolFeatures:
    """Capture the 512-dimensional classifier avgpool output."""

    def __init__(self, classifier: nn.Module):
        backbone = getattr(classifier, "model", None)
        layer = getattr(backbone, "avgpool", None)
        if layer is None:
            raise AttributeError("model must expose model.avgpool for Stage 1D-WT")
        self.value: torch.Tensor | None = None
        self.handle = layer.register_forward_hook(self._hook)
        self.classifier = classifier

    def _hook(self, _module, _inputs, output):
        self.value = output[0] if isinstance(output, (tuple, list)) else output

    @torch.inference_mode()
    def __call__(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.classifier(images).detach()
        if self.value is None:
            raise RuntimeError("avgpool hook did not capture a feature")
        return logits, self.value.flatten(1).detach()

    def close(self) -> None:
        self.handle.remove()


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_floats(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or tuple(sorted(values)) != values or any(value <= 0 for value in values):
        raise ValueError("epsilon values must be a positive ascending list")
    return values


def checkpoint_path(root: Path, group: str, seed: int) -> Path:
    aliases = {
        "inputaware": ("inputaware", "input_aware", "input-aware"),
        "adaptive_blend": ("adaptive_blend", "adaptive-blend", "adaptiveblend"),
    }
    for name in aliases.get(group, (group,)):
        candidate = root / name / f"seed{seed}" / "attack_result.pt"
        if candidate.is_file():
            return candidate
    return root / group / f"seed{seed}" / "attack_result.pt"


def load_any_model(root: Path, group: str, seed: int, *, bdb_root: Path, adaptive_root: Path | None, device: torch.device):
    if group == "adaptive_blend":
        if adaptive_root is None:
            raise ValueError("Adaptive-Blend requires --adaptive-blend-root")
        path = root / group / f"seed{seed}" / "official_model.pt"
        return load_backdoor_toolbox_resnet18(str(path), backdoor_toolbox_root=str(adaptive_root), device=device)
    return load_model(checkpoint_path(root, group, seed), str(bdb_root), device)


@torch.inference_mode()
def logits_and_features(model, dataset, indices: list[int], *, target: int, batch_size: int, device: torch.device):
    logits_rows, feature_rows, labels = [], [], []
    for _, images, batch_labels in batch_images(dataset, indices, batch_size=batch_size, device=device):
        logits = model(images).detach()
        logits_rows.append(logits.cpu())
        feature_rows.append(target_conditioned_logits_features(logits, target).cpu().numpy())
        labels.extend(batch_labels.cpu().tolist())
    logits = torch.cat(logits_rows)
    return logits, np.concatenate(feature_rows), np.asarray(labels, dtype=np.int64)


def coarse_probe_labels(model, dataset, indices: list[int], *, target: int, eps_pixels: tuple[float, ...], steps: int, batch_size: int, device: torch.device, seed: int):
    """Create full-grid first-success labels for one reference Clean model."""

    logits, _, _ = logits_and_features(model, dataset, indices, target=target, batch_size=batch_size, device=device)
    before = logits.argmax(dim=1).numpy()
    # A sample already predicted as the attack target has a zero-radius
    # targeted attack.  Reference labels retain this fact; only Clean0's
    # final selection pool excludes such samples.
    first = np.where(before == int(target), 0.0, np.inf).astype(np.float64)
    rows = []
    for eps in eps_pixels:
        successes, predictions = [], []
        for _, images, _ in batch_images(dataset, indices, batch_size=batch_size, device=device):
            result = targeted_pgd(model, images, target=target, epsilon=eps / 255.0, steps=steps,
                                  alpha=eps / 255.0 / 10.0, random_start=False, restarts=1)
            successes.extend(result.success.cpu().tolist())
            predictions.extend(result.best_prediction.cpu().tolist())
        for pos, sample_index in enumerate(indices):
            success = bool(successes[pos])
            if success and not math.isfinite(first[pos]):
                first[pos] = eps
            rows.append({
                "phase": "probe_label", "reference_clean_seed": seed, "target": target,
                "sample_index": sample_index, "epsilon_pixels": eps,
                "before_prediction": int(before[pos]), "after_prediction": int(predictions[pos]),
                "success": success, "robustness_radius_pixels": None if not math.isfinite(first[pos]) else float(first[pos]),
                "censored": not math.isfinite(first[pos]), "steps": steps, "random_start": False, "restarts": 1,
            })
    return first, rows


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    result = np.full(len(left), np.nan, dtype=np.float64)
    valid = denominator > 1e-12
    result[valid] = (left[valid] * right[valid]).sum(axis=1) / denominator[valid]
    return result


def concentration(vectors: np.ndarray) -> float | None:
    if len(vectors) < 2:
        return None
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.maximum(norms, 1e-12)
    values = (normalized @ normalized.T)[np.triu_indices(len(vectors), k=1)]
    return float(values[np.isfinite(values)].mean()) if np.isfinite(values).any() else None


def prototype(vectors: np.ndarray) -> np.ndarray | None:
    """Return the normalized mean direction for a shared-trigger summary."""

    if len(vectors) == 0:
        return None
    normalized = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    value = normalized.mean(axis=0)
    norm = np.linalg.norm(value)
    return value / norm if norm > 1e-12 else None


def shuffled_alignment(adv: np.ndarray, trigger: np.ndarray, *, seed: int, repeats: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(int(repeats)):
        order = rng.permutation(len(trigger))
        while len(trigger) > 1 and np.any(order == np.arange(len(trigger))):
            order = rng.permutation(len(trigger))
        values.append(cosine_rows(adv, trigger[order]))
    return np.nanmean(np.stack(values), axis=0)


def first_success_endpoint(endpoint_by_eps: dict[float, dict], eps_pixels: tuple[float, ...]):
    """Choose the first successful endpoint; use the largest-budget endpoint for censored samples."""

    n = len(endpoint_by_eps[eps_pixels[0]]["success"])
    first_eps = np.full(n, np.inf, dtype=np.float64)
    last = endpoint_by_eps[eps_pixels[-1]]
    selected = {key: value.copy() for key, value in last.items()}
    for eps in eps_pixels:
        current = endpoint_by_eps[eps]
        newly = np.isinf(first_eps) & current["success"]
        first_eps[newly] = eps
        for key in selected:
            selected[key][newly] = current[key][newly]
    return first_eps, selected


def load_quality(path: str | None) -> list[dict]:
    if not path:
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    records = payload.get("rows", payload.get("models", payload.get("records", payload.get("results", [])))) if isinstance(payload, dict) else []
    output = []
    for key, row in records.items() if isinstance(records, dict) else enumerate(records):
        item = dict(row)
        if "group" not in item and isinstance(key, str):
            item["group"] = key.rsplit("_seed", 1)[0]
        output.append(item)
    return output


def quality_status(rows: list[dict], group: str, seed: int) -> str | None:
    for row in rows:
        if str(row.get("group")) == group and int(row.get("seed", -1)) == seed:
            return str(row.get("status", "unknown"))
    return None


def write_figures(output: Path, rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    usable = [row for row in rows if row.get("alignment_available") and row.get("same_alignment") is not None]
    if usable:
        groups = defaultdict(list)
        for row in usable:
            groups[f"{row['model_alias']}:{row['trigger_type']}:{row['sample_group']}"] .append(float(row["same_alignment"]))
        labels = list(groups)
        plt.figure(figsize=(max(10, len(labels) * 0.6), 5))
        plt.boxplot([groups[label] for label in labels], labels=labels, showfliers=False)
        plt.xticks(rotation=65, ha="right")
        plt.ylabel("same-trigger avgpool alignment")
        plt.tight_layout()
        plt.savefig(output / "figures/alignment_boxplot.png", dpi=160)
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--backdoorbench-root", required=True)
    parser.add_argument("--adaptive-blend-root", default=None)
    parser.add_argument("--output-root", default="results/stage1d_wrong_target_trigger_alignment")
    parser.add_argument("--clean-group", default="clean_select_shared")
    parser.add_argument("--backdoor-groups", default=",".join(BACKDOOR_GROUPS))
    parser.add_argument("--reference-clean-seeds", default="1,2,3")
    parser.add_argument("--test-clean-seed", type=int, default=0)
    parser.add_argument("--targets", default="1,3,7")
    parser.add_argument("--train-count", type=int, default=1000)
    parser.add_argument("--test-count", type=int, default=1000)
    parser.add_argument("--split-seed", type=int, default=2031)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--train-eps-pixels", default=",".join(map(str, TRAIN_EPS_PIXELS)))
    parser.add_argument("--analysis-eps-pixels", default=",".join(map(str, ANALYSIS_EPS_PIXELS)))
    parser.add_argument("--train-steps", type=int, default=30)
    parser.add_argument("--analysis-steps", type=int, default=100)
    parser.add_argument("--analysis-restarts", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--quality-report", default=None)
    parser.add_argument("--ssba-test-path", default=None)
    parser.add_argument("--inputaware-state-path", default=None)
    parser.add_argument("--adaptive-blend-trigger-path", default=None)
    parser.add_argument("--blended-alpha", type=float, default=0.2)
    parser.add_argument("--wanet-s", type=float, default=0.5)
    parser.add_argument("--wanet-grid-rescale", type=float, default=1.0)
    parser.add_argument("--shuffle-repeats", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = parse_ints(args.targets)
    if not targets or any(target == 0 or target < 0 or target > 9 for target in targets):
        raise ValueError("Stage 1D-WT targets must be nonzero CIFAR-10 classes")
    reference_seeds = parse_ints(args.reference_clean_seeds)
    if args.test_clean_seed in reference_seeds:
        raise ValueError("test Clean seed must be excluded from Probe reference seeds")
    train_eps = parse_floats(args.train_eps_pixels)
    analysis_eps = parse_floats(args.analysis_eps_pixels)
    if tuple(analysis_eps) != ANALYSIS_EPS_PIXELS:
        raise ValueError("Stage 1D-WT mechanism analysis is fixed to epsilon 1 and 1.5 / 255")
    groups = tuple(item.strip() for item in args.backdoor_groups.split(",") if item.strip())
    if set(groups) - set(BACKDOOR_GROUPS):
        raise ValueError(f"unsupported backdoor groups: {sorted(set(groups) - set(BACKDOOR_GROUPS))}")

    output = timestamp_run_dir(args.output_root, "wrong_target_alignment")
    # Keep the exact official training provenance next to the analysis result.
    for name in ("training_configs", "training_logs"):
        source = Path(args.model_root) / name
        if source.is_dir():
            shutil.copytree(source, output / name, dirs_exist_ok=True)
    device = torch.device(args.device)
    data_root, model_root, bdb_root = Path(args.data_root), Path(args.model_root), Path(args.backdoorbench_root)
    adaptive_root = Path(args.adaptive_blend_root) if args.adaptive_blend_root else None
    train_data, test_data = cifar100_dataset(data_root, train=True), cifar100_dataset(data_root, train=False)
    rng = np.random.default_rng(args.split_seed)
    train_indices = rng.permutation(len(train_data))[:args.train_count].astype(int).tolist()
    test_indices = rng.permutation(len(test_data))[:args.test_count].astype(int).tolist()
    write_csv(output / "candidate_pool.csv", [
        {"split": "cifar100_train", "position": pos, "sample_index": index} for pos, index in enumerate(train_indices)
    ] + [{"split": "cifar100_test", "position": pos, "sample_index": index} for pos, index in enumerate(test_indices)])
    config = vars(args).copy()
    config.update({"protocol": "stage1d-wrong-target-trigger-alignment-v1", "targets": list(targets), "reference_clean_seeds": list(reference_seeds), "analysis_eps_pixels": list(analysis_eps), "train_eps_pixels": list(train_eps)})
    output.joinpath("config.resolved.yaml").write_text(__import__("yaml").safe_dump(config, sort_keys=False), encoding="utf-8")
    log_path = output / "run.log"
    def log(message: str):
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")
    log(f"Stage 1D-WT started: {output.resolve()}")

    reference_models = {seed: load_any_model(model_root, args.clean_group, seed, bdb_root=bdb_root, adaptive_root=None, device=device)[0] for seed in reference_seeds}
    probes, probe_rows = {}, []
    for target in targets:
        feature_rows, label_rows = [], []
        names = target_feature_names(target, 10)
        for seed, model in reference_models.items():
            logits, features, _ = logits_and_features(model, train_data, train_indices, target=target, batch_size=args.batch_size, device=device)
            radius, coarse_rows = coarse_probe_labels(model, train_data, train_indices, target=target, eps_pixels=train_eps, steps=args.train_steps, batch_size=args.batch_size, device=device, seed=seed)
            finite = np.isfinite(radius)
            feature_rows.extend(features[finite])
            label_rows.extend(radius[finite])
            for pos, sample_index in enumerate(train_indices):
                row = {
                    "phase": "probe_label",
                    "reference_clean_seed": seed,
                    "target": target,
                    "sample_index": sample_index,
                    "robustness_radius_pixels": None if not finite[pos] else float(radius[pos]),
                    "censored": not bool(finite[pos]),
                    "original_prediction": int(logits[pos].argmax().item()),
                }
                row.update({f"feature_{name}": float(features[pos, col]) for col, name in enumerate(names)})
                probe_rows.append(row)
        probe = RidgeProbe.fit(np.asarray(feature_rows), np.asarray(label_rows), alpha=args.ridge_alpha, feature_names=names)
        probe.save(output / f"probe_target_{target}.npz")
        probes[target] = probe
        log(f"target={target}: Probe fitted on {len(label_rows)} finite reference records")
    write_csv(output / "probe_training_records.csv", probe_rows)
    write_json(output / "probe_parameters.json", {str(target): {"feature_names": list(probe.feature_names), "feature_mean": probe.feature_mean.tolist(), "feature_std": probe.feature_std.tolist(), "weight": probe.weight.tolist(), "bias": float(probe.bias), "alpha": probe.alpha} for target, probe in probes.items()})

    clean0, _ = load_any_model(model_root, args.clean_group, args.test_clean_seed, bdb_root=bdb_root, adaptive_root=None, device=device)
    selected_by_target, selection_rows, score_rows = {}, [], []
    for target, probe in probes.items():
        logits, features, _ = logits_and_features(clean0, test_data, test_indices, target=target, batch_size=args.batch_size, device=device)
        predictions = logits.argmax(dim=1).numpy()
        scores = probe.predict(features)
        eligible = predictions != target
        positions = [pos for pos in np.argsort(-scores, kind="mergesort") if eligible[pos]][:args.top_k]
        selected = [test_indices[pos] for pos in positions]
        selected_by_target[target] = selected
        for pos, index in enumerate(test_indices):
            score_rows.append({"target": target, "sample_index": index, "original_prediction": int(predictions[pos]), "probe_score": float(scores[pos]), "eligible": bool(eligible[pos])})
        for rank, pos in enumerate(positions, start=1):
            selection_rows.append({"target": target, "rank": rank, "sample_index": test_indices[pos], "original_prediction": int(predictions[pos]), "probe_score": float(scores[pos]), "selector": "probe_clean0_shared"})
    write_csv(output / "probe_selection_scores.csv", score_rows)
    write_csv(output / "selected_targeted_robust_samples.csv", selection_rows)

    explicit = {"ssba": Path(args.ssba_test_path) if args.ssba_test_path else None, "inputaware": Path(args.inputaware_state_path) if args.inputaware_state_path else None, "adaptive_blend": Path(args.adaptive_blend_trigger_path) if args.adaptive_blend_trigger_path else None, "badnet": None, "blended": None, "wanet_identity": None, "wanet_noise": None}
    adapters = build_trigger_adapters(model_root=model_root, backdoorbench_root=bdb_root, explicit=explicit, device=device, blended_alpha=args.blended_alpha, wanet_s=args.wanet_s, wanet_grid_rescale=args.wanet_grid_rescale)
    quality = load_quality(args.quality_report)
    clean_status = quality_status(quality, args.clean_group, args.test_clean_seed)
    if clean_status in {"gate_failed", "failed"}:
        raise RuntimeError(f"Clean seed{args.test_clean_seed} failed the model-quality gate")
    analysis_groups = [group for group in groups if quality_status(quality, group, 0) not in {"gate_failed", "failed"}]
    excluded_groups = {group: quality_status(quality, group, 0) for group in groups if group not in analysis_groups}
    if excluded_groups:
        log(f"excluded gate-failed groups: {excluded_groups}")
    model_cache = {"clean": clean0}
    for group in analysis_groups:
        model_cache[group] = load_any_model(model_root, group, 0, bdb_root=bdb_root, adaptive_root=adaptive_root, device=device)[0]

    attack_rows, trigger_rows, alignment_rows = [], [], []
    trigger_prototypes = {}
    for target, selected_indices in selected_by_target.items():
        for trigger_type in analysis_groups:
            adapter = adapters[trigger_type]
            for model_alias in ("clean", trigger_type):
                model = model_cache[model_alias]
                extractor = AvgPoolFeatures(model)
                try:
                    base_parts, trigger_parts, endpoint_by_eps = [], [], {eps: [] for eps in analysis_eps}
                    for batch_indices, images, _ in batch_images(test_data, selected_indices, batch_size=args.batch_size, device=device):
                        base_logits, base_features = extractor(images)
                        base_parts.append({"logits": base_logits.cpu().numpy(), "features": base_features.cpu().numpy()})
                        if adapter.status.available:
                            triggered = adapter.apply(images, sample_indices=batch_indices, split="cifar100_test")
                            _, trig_features = extractor(triggered)
                            trigger_parts.append(trig_features.cpu().numpy())
                        for eps in analysis_eps:
                            endpoint = targeted_pgd_endpoint(model, images, target=target, epsilon=eps / 255.0, steps=args.analysis_steps, alpha=eps / 255.0 / 10.0, random_start=True, restarts=args.analysis_restarts)
                            _, endpoint_features = extractor(endpoint.endpoint)
                            endpoint_by_eps[eps].append({"success": endpoint.success.cpu().numpy(), "prediction": endpoint.endpoint_prediction.cpu().numpy(), "linf": endpoint.endpoint_linf.cpu().numpy() * 255.0, "loss": endpoint.target_loss.cpu().numpy(), "features": endpoint_features.cpu().numpy()})
                    base_logits = np.concatenate([part["logits"] for part in base_parts])
                    base_features = np.concatenate([part["features"] for part in base_parts])
                    base_predictions = base_logits.argmax(axis=1)
                    trigger_available = adapter.status.available
                    trigger_delta = np.concatenate(trigger_parts) - base_features if trigger_available else None
                    trigger_predictions = np.full(len(selected_indices), -1, dtype=np.int64)
                    trigger_success = np.full(len(selected_indices), False, dtype=bool)
                    if trigger_available:
                        trigger_logits_parts = []
                        for batch_indices, images, _ in batch_images(test_data, selected_indices, batch_size=args.batch_size, device=device):
                            triggered = adapter.apply(images, sample_indices=batch_indices, split="cifar100_test")
                            trigger_logits_parts.append(model(triggered).detach().cpu().numpy())
                        trigger_predictions = np.concatenate(trigger_logits_parts).argmax(axis=1)
                        trigger_success = trigger_predictions == 0
                    trigger_concentration = concentration(trigger_delta) if trigger_available else None
                    if trigger_available:
                        trigger_vector = prototype(trigger_delta)
                        if trigger_vector is not None:
                            trigger_prototypes[f"{trigger_type}:{model_alias}:target{target}"] = trigger_vector.tolist()
                    for pos, sample_index in enumerate(selected_indices):
                        trigger_rows.append({"model_group": model_alias, "model_seed": 0, "trigger_type": trigger_type, "attack_target": target, "true_backdoor_target": 0, "sample_index": sample_index, "trigger_source": adapter.status.source, "trigger_available": trigger_available, "alignment_available": trigger_available, "alignment_unavailable_reason": adapter.status.reason if not trigger_available else None, "original_prediction": int(base_predictions[pos]), "trigger_prediction": int(trigger_predictions[pos]), "trigger_success_true_target": bool(trigger_success[pos]) if trigger_available else None})
                    endpoint_arrays = {eps: {key: np.concatenate([part[key] for part in parts]) for key in parts[0]} for eps, parts in endpoint_by_eps.items()}
                    first_eps, first_data = first_success_endpoint(endpoint_arrays, analysis_eps)
                    target_resistance = target_margin(torch.from_numpy(base_logits), target).numpy()
                    for eps in analysis_eps:
                        data = endpoint_arrays[eps]
                        for pos, sample_index in enumerate(selected_indices):
                            attack_rows.append({"model_group": model_alias, "model_seed": 0, "trigger_type": trigger_type, "attack_target": target, "true_backdoor_target": 0, "sample_index": sample_index, "epsilon_pixels": eps, "before_prediction": int(base_predictions[pos]), "after_prediction": int(data["prediction"][pos]), "original_prediction": int(base_predictions[pos]), "final_prediction": int(data["prediction"][pos]), "success": bool(data["success"][pos]), "targeted_loss": float(data["loss"][pos]), "actual_linf_pixels": float(data["linf"][pos]), "first_success_epsilon_pixels": None if math.isinf(first_eps[pos]) else float(first_eps[pos]), "censored": bool(math.isinf(first_eps[pos])), "target_resistance_margin": float(target_resistance[pos]), "steps": args.analysis_steps, "random_start": True, "restarts": args.analysis_restarts})
                    protocols = [("first_success", first_data["features"], np.isfinite(first_eps))]
                    protocols.extend((f"fixed_{eps:g}", endpoint_arrays[eps]["features"], endpoint_arrays[eps]["success"]) for eps in analysis_eps)
                    for protocol, adv_features, success_values in protocols:
                        adv_delta = adv_features - base_features
                        same = cosine_rows(adv_delta, trigger_delta) if trigger_available else np.full(len(selected_indices), np.nan)
                        shuffled = shuffled_alignment(adv_delta, trigger_delta, seed=2031 + target, repeats=args.shuffle_repeats) if trigger_available else np.full(len(selected_indices), np.nan)
                        for pos, sample_index in enumerate(selected_indices):
                            pre_target = bool(base_predictions[pos] == target)
                            sample_group = "pre_target" if pre_target else ("success" if bool(success_values[pos]) else "failure")
                            subset = (~(base_predictions == target)) & (success_values if sample_group == "success" else ~success_values)
                            alignment_rows.append({"model_group": model_alias, "model_seed": 0, "trigger_type": trigger_type, "attack_target": target, "true_backdoor_target": 0, "sample_index": sample_index, "protocol": protocol, "sample_group": sample_group, "success": bool(success_values[pos]), "pre_target": pre_target, "trigger_success_true_target": bool(trigger_success[pos]) if trigger_available else None, "trigger_available": trigger_available, "alignment_available": trigger_available, "same_alignment": float(same[pos]) if math.isfinite(float(same[pos])) else None, "shuffled_alignment": float(shuffled[pos]) if math.isfinite(float(shuffled[pos])) else None, "same_minus_shuffled": float(same[pos] - shuffled[pos]) if math.isfinite(float(same[pos])) and math.isfinite(float(shuffled[pos])) else None, "adversarial_feature_norm": float(np.linalg.norm(adv_delta[pos])), "trigger_feature_norm": float(np.linalg.norm(trigger_delta[pos])) if trigger_available else None, "target_resistance_margin": float(target_resistance[pos]), "first_success_epsilon_pixels": None if math.isinf(first_eps[pos]) else float(first_eps[pos]), "censored": bool(math.isinf(first_eps[pos])), "adv_concentration": concentration(adv_delta[subset]), "trigger_concentration": trigger_concentration})
                finally:
                    extractor.close()
        log(f"target={target}: completed paired mechanism analysis")

    metric_rows = []
    grouped = defaultdict(list)
    for row in alignment_rows:
        grouped[(row["target"], row["trigger_type"], row["model_group"], row["protocol"], row["sample_group"])].append(row)
    for key, rows in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        target, trigger_type, model_group, protocol, sample_group = key
        values = [float(row["same_alignment"]) for row in rows if row["same_alignment"] is not None]
        shuffled = [float(row["shuffled_alignment"]) for row in rows if row["shuffled_alignment"] is not None]
        metric_rows.append({"target": target, "trigger_type": trigger_type, "model_group": model_group, "protocol": protocol, "sample_group": sample_group, "n": len(rows), "success_rate": float(np.mean([row["success"] for row in rows])), "trigger_success_rate": float(np.mean([row["trigger_success_true_target"] for row in rows])) if rows and all(row["trigger_success_true_target"] is not None for row in rows) else None, "alignment_mean": float(np.mean(values)) if values else None, "shuffled_alignment_mean": float(np.mean(shuffled)) if shuffled else None, "same_minus_shuffled_mean": float(np.mean(values) - np.mean(shuffled)) if values and shuffled else None, "adv_concentration": rows[0]["adv_concentration"], "trigger_concentration": rows[0]["trigger_concentration"], "alignment_available": bool(values)})

    quality_rows = quality or [{"group": args.clean_group, "seed": args.test_clean_seed, "status": "not_provided"}] + [{"group": group, "seed": 0, "status": "not_provided"} for group in groups]
    write_csv(output / "trigger_records.csv", trigger_rows)
    write_csv(output / "attack_records.csv", attack_rows)
    write_csv(output / "feature_alignment_records.csv", alignment_rows)
    write_csv(output / "group_metrics.csv", metric_rows)
    write_csv(output / "model_quality.csv", quality_rows)
    write_figures(output, alignment_rows)
    summary = {**config, "train_indices": train_indices, "test_indices": test_indices, "selected_indices_by_target": selected_by_target, "trigger_status": {group: vars(adapter.status) for group, adapter in adapters.items()}, "trigger_prototypes": trigger_prototypes, "analysis_backdoor_groups": analysis_groups, "excluded_gate_failed_groups": excluded_groups, "model_quality": quality_rows, "outputs": {"attack_records": str((output / "attack_records.csv").resolve()), "alignment_records": str((output / "feature_alignment_records.csv").resolve())}, "interpretation": "wrong-target attack to official-trigger-related feature direction; not a literal A->B->C class-transition claim"}
    write_json(output / "summary.json", summary)
    log(f"Stage 1D-WT complete: {output.resolve()}")


if __name__ == "__main__":
    main()
