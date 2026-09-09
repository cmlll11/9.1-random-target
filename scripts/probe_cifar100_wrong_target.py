"""Stage WT: target-specific Probe and wrong-target targeted-PGD pilot.

The true BadNet target is class 0.  The pilot additionally evaluates fixed
wrong targets (1, 3, and 7 by default).  Every model selects its own Probe
and margin samples from the model-specific eligible pool, while a fixed
random order provides the no-selection baseline.  PGD is never used to
select deployment samples.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mdluap.data import cifar100_dataset
from mdluap.probes import RidgeProbe, target_conditioned_logits_features, target_feature_names
from mdluap.targeted_pgd import targeted_pgd
from pilot_common import batch_images, load_model, seed_everything, timestamp_run_dir, write_csv, write_json


TRAIN_EPS_PIXELS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0, 16.0, 32.0)
TEST_EPS_PIXELS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
LOW_EPS_PIXELS = (0.5, 1.0, 1.5)


def parse_ints(value: str) -> list[int]:
    """Parse comma-separated integer arguments."""

    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_floats(value: str) -> tuple[float, ...]:
    """Parse a positive ascending epsilon grid expressed in pixel units."""

    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or tuple(sorted(values)) != values or any(value <= 0 for value in values):
        raise ValueError("epsilon grid must be a positive ascending list")
    return values


def checkpoint_path(model_root: Path, group: str, seed: int) -> Path:
    """Resolve a packaged BackdoorBench checkpoint."""

    return model_root / group / f"seed{seed}" / "attack_result.pt"


@torch.inference_mode()
def logits_for_indices(model, dataset, indices: list[int], *, batch_size: int, device: torch.device):
    """Compute logits and labels in stable dataset-index order."""

    logits_rows = []
    label_rows = []
    for _, images, labels in batch_images(dataset, indices, batch_size=batch_size, device=device):
        logits_rows.append(model(images).detach().cpu())
        label_rows.append(labels.detach().cpu())
    return torch.cat(logits_rows), torch.cat(label_rows)


def run_targeted_grid(
    model,
    dataset,
    indices: list[int],
    *,
    target: int,
    eps_pixels: tuple[float, ...],
    steps: int,
    random_start: bool,
    restarts: int,
    batch_size: int,
    device: torch.device,
    phase: str,
    model_group: str,
    model_seed: int,
) -> tuple[list[dict], dict[int, int]]:
    """Run a targeted PGD grid and return long rows plus robustness levels."""

    before_logits, _ = logits_for_indices(model, dataset, indices, batch_size=batch_size, device=device)
    before = before_logits.argmax(dim=1).tolist()
    first_level = {index: 0 if prediction == int(target) else math.inf for index, prediction in zip(indices, before)}
    rows: list[dict] = []

    for level, epsilon_pixels in enumerate(eps_pixels, start=1):
        epsilon = float(epsilon_pixels) / 255.0
        if epsilon == 0.0:
            # The deployment pool already excludes samples whose original
            # prediction is the target.  Keeping epsilon=0 in the output is
            # still useful as an explicit ASR@0 sanity check.
            successes = [prediction == int(target) for prediction in before]
            norms = [0.0] * len(indices)
            predictions = list(before)
        else:
            successes = []
            norms = []
            predictions = []
            for _, images, _ in batch_images(dataset, indices, batch_size=batch_size, device=device):
                result = targeted_pgd(
                    model,
                    images,
                    target=int(target),
                    epsilon=epsilon,
                    steps=int(steps),
                    alpha=epsilon / 10.0,
                    random_start=bool(random_start),
                    restarts=int(restarts),
                )
                successes.append(result.success.detach().cpu())
                norms.append(result.best_linf.detach().cpu())
                predictions.append(result.best_prediction.detach().cpu())
            successes = torch.cat(successes).tolist()
            norms = torch.cat(norms).tolist()
            predictions = torch.cat(predictions).tolist()
        for position, sample_index in enumerate(indices):
            success = bool(successes[position])
            if success and math.isinf(float(first_level[sample_index])):
                first_level[sample_index] = level
            rows.append(
                {
                    "phase": phase,
                    "model_group": model_group,
                    "model_seed": int(model_seed),
                    "target": int(target),
                    "sample_index": int(sample_index),
                    "epsilon_pixels": float(epsilon_pixels),
                    "before_prediction": int(before[position]),
                    "after_prediction": int(predictions[position]),
                    "success": success,
                    "actual_best_linf_pixels": float(norms[position]) * 255.0 if math.isfinite(float(norms[position])) else None,
                    "steps": int(steps),
                    "random_start": bool(random_start),
                    "restarts": int(restarts),
                }
            )

    max_level = len(eps_pixels) + 1
    return rows, {index: int(level if math.isfinite(float(level)) else max_level) for index, level in first_level.items()}


def select_top(indices: list[int], scores: dict[int, float], count: int) -> list[int]:
    """Select largest scores with deterministic index tie-breaking."""

    return sorted(indices, key=lambda index: (-float(scores[index]), int(index)))[: int(count)]


def select_random_eligible(indices: list[int], order: list[int], eligible: set[int], count: int) -> list[int]:
    """Select from a fixed random order after applying model-specific eligibility."""

    selected = [index for index in order if index in eligible][: int(count)]
    if len(selected) < int(count):
        raise RuntimeError(f"only {len(selected)} eligible random samples are available; need {count}")
    return selected


def attach_selector(rows: list[dict], membership: dict[int, list[str]], **context) -> list[dict]:
    """Duplicate raw attack rows for every selector containing the image."""

    output = []
    for row in rows:
        for selector in membership.get(int(row["sample_index"]), []):
            output.append({**row, **context, "selector": selector})
    return output


def summarize_asr(rows: list[dict]) -> list[dict]:
    """Summarize ASR for each target, seed, selector and model."""

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["analysis_part"], int(row["target"]), int(row["selection_seed"]),
                row["selector"], row["evaluation_group"], int(row["evaluation_seed"]),
                float(row["epsilon_pixels"]),
            )
        ].append(row)
    output = []
    for key, values in sorted(grouped.items()):
        part, target, seed, selector, group, model_seed, epsilon = key
        successes = sum(bool(row["success"]) for row in values)
        output.append(
            {
                "metric": "asr",
                "analysis_part": part,
                "target": target,
                "selection_seed": seed,
                "selector": selector,
                "evaluation_group": group,
                "evaluation_seed": model_seed,
                "epsilon_pixels": epsilon,
                "n": len(values),
                "success_count": successes,
                "asr": successes / len(values),
                "asr_at_zero_expected": epsilon == 0.0,
            }
        )
    return output


def spearman(left: list[float], right: list[float]) -> float | None:
    """Compute Spearman correlation without scipy."""

    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    rx = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort")
    ry = np.argsort(np.argsort(y, kind="mergesort"), kind="mergesort")
    return float(np.corrcoef(rx, ry)[0, 1])


def load_quality_report(path: str) -> list[dict]:
    """Load the existing model-gate JSON when supplied."""

    if not path:
        return []
    payload = json_load(Path(path))
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        records = payload.get("models", payload.get("records", payload.get("results", [])))
        return [dict(row) for row in records if isinstance(row, dict)]
    return []


def json_load(path: Path):
    """Read JSON without adding a second file utility dependency."""

    import json

    return json.loads(path.read_text(encoding="utf-8"))


def write_run_log(output: Path, message: str) -> None:
    """Print and append a progress line."""

    print(message, flush=True)
    with (output / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def parse_args() -> argparse.Namespace:
    """Parse the fixed pilot protocol and server paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--backdoorbench-root", required=True)
    parser.add_argument("--output-root", default="results/stage_wt_wrong_target")
    parser.add_argument("--clean-group", default="clean_select_shared")
    parser.add_argument("--backdoor-group", default="badnet")
    parser.add_argument("--reference-clean-seeds", default="3,4")
    parser.add_argument("--target-seeds", default="0,1,2")
    parser.add_argument("--targets", default="0,1,3,7")
    parser.add_argument("--train-count", type=int, default=1000)
    parser.add_argument("--test-count", type=int, default=1000)
    parser.add_argument("--split-seed", type=int, default=2030)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--train-eps-pixels", default=",".join(map(str, TRAIN_EPS_PIXELS)))
    parser.add_argument("--test-eps-pixels", default=",".join(map(str, TEST_EPS_PIXELS)))
    parser.add_argument("--train-steps", type=int, default=30)
    parser.add_argument("--test-steps", type=int, default=100)
    parser.add_argument("--test-restarts", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--quality-report", default="")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    """Train target-specific Probes and run the deployment-style pilot."""

    args = parse_args()
    targets = parse_ints(args.targets)
    reference_seeds = parse_ints(args.reference_clean_seeds)
    target_seeds = parse_ints(args.target_seeds)
    train_eps = parse_floats(args.train_eps_pixels)
    test_eps = parse_floats(args.test_eps_pixels)
    if 0 not in targets:
        raise ValueError("target 0 must be included as the known-target positive control")
    if set(reference_seeds) & set(target_seeds):
        raise ValueError("reference and target Clean seeds must be disjoint")

    output = timestamp_run_dir(args.output_root, "wrong_target")
    device = torch.device(args.device)
    model_root = Path(args.model_root)
    train_dataset = cifar100_dataset(args.data_root, train=True)
    test_dataset = cifar100_dataset(args.data_root, train=False)
    rng = np.random.default_rng(args.split_seed)
    train_indices = rng.permutation(len(train_dataset))[: int(args.train_count)].astype(int).tolist()
    test_indices = rng.permutation(len(test_dataset))[: int(args.test_count)].astype(int).tolist()
    split_rows = [
        {"split": "probe_train", "position": position, "sample_index": index, "dataset": "CIFAR100_train"}
        for position, index in enumerate(train_indices)
    ] + [
        {"split": "target_test", "position": position, "sample_index": index, "dataset": "CIFAR100_test"}
        for position, index in enumerate(test_indices)
    ]
    write_csv(output / "cifar100_split.csv", split_rows)

    config = vars(args).copy()
    config.update({
        "targets": targets,
        "reference_clean_seeds": reference_seeds,
        "target_seeds": target_seeds,
        "train_epsilon_pixels": list(train_eps),
        "test_epsilon_pixels": list(test_eps),
        "protocol": "stage-wt-wrong-target-v1",
    })
    (output / "config.resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    write_run_log(output, f"Stage WT started: {output.resolve()}")

    reference_models = {}
    for seed in reference_seeds:
        path = checkpoint_path(model_root, args.clean_group, seed)
        reference_models[seed] = load_model(path, args.backdoorbench_root, device)[0]

    probes: dict[int, RidgeProbe] = {}
    probe_rows: list[dict] = []
    for target in targets:
        features_parts = []
        labels_parts = []
        for seed, model in reference_models.items():
            logits, _ = logits_for_indices(model, train_dataset, train_indices, batch_size=args.batch_size, device=device)
            features = target_conditioned_logits_features(logits, target).numpy()
            attack_rows, levels = run_targeted_grid(
                model,
                train_dataset,
                train_indices,
                target=target,
                eps_pixels=train_eps,
                steps=args.train_steps,
                random_start=False,
                restarts=1,
                batch_size=args.batch_size,
                device=device,
                phase="probe_label",
                model_group=args.clean_group,
                model_seed=seed,
            )
            for position, sample_index in enumerate(train_indices):
                level = int(levels[sample_index])
                row = {
                    "target": target,
                    "reference_clean_seed": seed,
                    "sample_index": sample_index,
                    "robustness_level": level,
                    "censored": level == len(train_eps) + 1,
                    "original_prediction": int(logits[position].argmax().item()),
                }
                row.update({f"feature_{name}": float(features[position, column]) for column, name in enumerate(target_feature_names(target, logits.shape[1]))})
                probe_rows.append(row)
                features_parts.append(features[position])
                labels_parts.append(level)
        probe = RidgeProbe.fit(
            np.asarray(features_parts),
            np.asarray(labels_parts, dtype=np.float64),
            alpha=args.ridge_alpha,
            feature_names=target_feature_names(target, 10),
        )
        probe.save(output / f"probe_target_{target}.npz")
        probes[target] = probe
        write_run_log(output, f"target={target}: fitted Probe on {len(labels_parts)} reference records")
    write_csv(output / "probe_training_records.csv", probe_rows)

    random_orders = {
        (seed, target): np.random.default_rng(args.split_seed + 10000 + seed * 100 + target).permutation(test_indices).astype(int).tolist()
        for seed in target_seeds for target in targets
    }
    model_specs = [("clean", args.clean_group), ("backdoor", args.backdoor_group)]
    pool_rows: list[dict] = []
    selector_rows: list[dict] = []
    deployment_rows: list[dict] = []
    paired_rows: list[dict] = []
    sanity_rows: list[dict] = []

    for target in targets:
        probe = probes[target]
        for evaluation_kind, checkpoint_group in model_specs:
            for seed in target_seeds:
                model = load_model(checkpoint_path(model_root, checkpoint_group, seed), args.backdoorbench_root, device)[0]
                logits, _ = logits_for_indices(model, test_dataset, test_indices, batch_size=args.batch_size, device=device)
                predictions = logits.argmax(dim=1).tolist()
                features = target_conditioned_logits_features(logits, target).numpy()
                probe_scores = probe.predict(features)
                other = logits.clone()
                other[:, target] = float("-inf")
                margin_scores = (other.max(dim=1).values - logits[:, target]).numpy()
                eligible = {index for index, prediction in zip(test_indices, predictions) if int(prediction) != int(target)}
                probe_by_index = {index: float(score) for index, score in zip(test_indices, probe_scores)}
                margin_by_index = {index: float(score) for index, score in zip(test_indices, margin_scores)}
                selected = {
                    "probe": select_top(sorted(eligible), probe_by_index, args.top_k),
                    "target_margin": select_top(sorted(eligible), margin_by_index, args.top_k),
                    "random": select_random_eligible(test_indices, random_orders[(seed, target)], eligible, args.top_k),
                }
                membership: dict[int, list[str]] = defaultdict(list)
                for selector, indices in selected.items():
                    for rank, sample_index in enumerate(indices, start=1):
                        membership[sample_index].append(selector)
                        position = test_indices.index(sample_index)
                        selector_rows.append({
                            "analysis_part": "deployment",
                            "selection_group": evaluation_kind,
                            "selection_seed": seed,
                            "target": target,
                            "selector": selector,
                            "rank": rank,
                            "sample_index": sample_index,
                            "initial_prediction": predictions[position],
                            "probe_score": probe_by_index[sample_index],
                            "target_margin": margin_by_index[sample_index],
                            "eligible": True,
                        })
                for position, sample_index in enumerate(test_indices):
                    pool_rows.append({
                        "analysis_part": "deployment",
                        "selection_group": evaluation_kind,
                        "selection_seed": seed,
                        "target": target,
                        "sample_index": sample_index,
                        "initial_prediction": predictions[position],
                        "eligible": sample_index in eligible,
                        "probe_score": float(probe_scores[position]),
                        "target_margin": float(margin_scores[position]),
                    })
                union = sorted(membership)
                raw_rows, _ = run_targeted_grid(
                    model,
                    test_dataset,
                    union,
                    target=target,
                    eps_pixels=(0.0,) + test_eps,
                    steps=args.test_steps,
                    random_start=True,
                    restarts=args.test_restarts,
                    batch_size=args.batch_size,
                    device=device,
                    phase="deployment_attack",
                    model_group=evaluation_kind,
                    model_seed=seed,
                )
                deployment_rows.extend(attach_selector(
                    raw_rows,
                    membership,
                    analysis_part="deployment",
                    target=int(target),
                    selection_seed=int(seed),
                    evaluation_group=evaluation_kind,
                    evaluation_seed=int(seed),
                ))
                if evaluation_kind == "clean":
                    sanity_rows.extend(attach_selector(
                        raw_rows,
                        membership,
                        analysis_part="clean_selector_sanity",
                        target=int(target),
                        selection_seed=int(seed),
                        evaluation_group="clean",
                        evaluation_seed=int(seed),
                    ))
                if evaluation_kind == "clean":
                    paired_membership = {index: ["clean_probe_shared"] for index in selected["probe"]}
                    paired_raw_clean, _ = run_targeted_grid(
                        model, test_dataset, selected["probe"], target=target,
                        eps_pixels=(0.0,) + test_eps, steps=args.test_steps,
                        random_start=True, restarts=args.test_restarts,
                        batch_size=args.batch_size, device=device,
                        phase="paired_clean_probe", model_group="clean", model_seed=seed,
                    )
                    paired_rows.extend(attach_selector(
                        paired_raw_clean, paired_membership,
                        analysis_part="paired_clean_probe", target=int(target),
                        selection_seed=int(seed), evaluation_group="clean", evaluation_seed=int(seed),
                    ))
                    badnet = load_model(checkpoint_path(model_root, args.backdoor_group, seed), args.backdoorbench_root, device)[0]
                    paired_raw_badnet, _ = run_targeted_grid(
                        badnet, test_dataset, selected["probe"], target=target,
                        eps_pixels=(0.0,) + test_eps, steps=args.test_steps,
                        random_start=True, restarts=args.test_restarts,
                        batch_size=args.batch_size, device=device,
                        phase="paired_clean_probe", model_group="backdoor", model_seed=seed,
                    )
                    paired_rows.extend(attach_selector(
                        paired_raw_badnet, paired_membership,
                        analysis_part="paired_clean_probe", target=int(target),
                        selection_seed=int(seed), evaluation_group="backdoor", evaluation_seed=seed,
                    ))
        write_run_log(output, f"target={target}: completed deployment and paired evaluations")

    all_rows = deployment_rows + paired_rows
    metrics = summarize_asr(all_rows)
    gap_rows = []
    lookup = {
        (row["analysis_part"], row["target"], row["selection_seed"], row["selector"], row["evaluation_group"], row["evaluation_seed"], row["epsilon_pixels"]): row
        for row in metrics
    }
    for part in ("deployment", "paired_clean_probe"):
        selectors = sorted({row["selector"] for row in metrics if row["analysis_part"] == part})
        for target in targets:
            for seed in target_seeds:
                for selector in selectors:
                    for epsilon in (0.0,) + test_eps:
                        clean = lookup.get((part, target, seed, selector, "clean", seed, float(epsilon)))
                        backdoor = lookup.get((part, target, seed, selector, "backdoor", seed, float(epsilon)))
                        if clean and backdoor:
                            delta = float(backdoor["asr"]) - float(clean["asr"])
                            gap_rows.append({
                                "analysis_part": part,
                                "target": target,
                                "selection_seed": seed,
                                "selector": selector,
                                "epsilon_pixels": float(epsilon),
                                "clean_asr": clean["asr"],
                                "backdoor_asr": backdoor["asr"],
                                "delta_asr": delta,
                            })
    small_gap_rows = []
    for target in targets:
        for seed in target_seeds:
            for selector in ("probe", "target_margin", "random"):
                values = [
                    row["delta_asr"] for row in gap_rows
                    if row["analysis_part"] == "deployment" and row["target"] == target
                    and row["selection_seed"] == seed and row["selector"] == selector
                    and row["epsilon_pixels"] in LOW_EPS_PIXELS
                ]
                if len(values) == len(LOW_EPS_PIXELS):
                    small_gap_rows.append({
                        "target": target,
                        "selection_seed": seed,
                        "selector": selector,
                        "small_budget_mean_delta_asr": float(np.mean(values)),
                    })

    quality_rows = load_quality_report(args.quality_report)
    if not quality_rows:
        quality_rows = [{"group": group, "seed": seed, "status": "not_provided"} for _, group in model_specs for seed in target_seeds]
    write_csv(output / "target_pool_scores.csv", pool_rows)
    write_csv(output / "selector_sets.csv", selector_rows)
    write_csv(output / "deployment_attack_records.csv", deployment_rows)
    write_csv(output / "paired_diagnostic_records.csv", paired_rows)
    write_csv(output / "clean_sanity_records.csv", sanity_rows)
    write_csv(output / "group_metrics.csv", metrics)
    write_csv(output / "gap_metrics.csv", gap_rows)
    write_csv(output / "small_budget_gap_metrics.csv", small_gap_rows)
    write_csv(output / "model_quality.csv", quality_rows)

    summary = {
        **config,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "probe_models": {str(target): str(output / f"probe_target_{target}.npz") for target in targets},
        "group_metrics": metrics,
        "gap_metrics": gap_rows,
        "small_budget_gap_metrics": small_gap_rows,
        "model_quality": quality_rows,
        "eligibility_rule": "initial_prediction != target",
        "censoring_label": "max_robustness_level",
        "circularity_controls": {
            "probe_fit_models": reference_seeds,
            "probe_fit_dataset": "CIFAR100_train",
            "deployment_selector_uses_target_model_pgd": False,
            "pgd_reference_selector_used": False,
            "random_selection_uses_fixed_order": True,
        },
    }
    write_json(output / "summary.json", summary)
    write_run_log(output, f"Wrong-target experiment complete: {output.resolve()}")


if __name__ == "__main__":
    main()
