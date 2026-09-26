"""Pixel-space residual decoupling analysis for the fixed Probe Top-100."""

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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from mdluap.data import cifar10_dataset
from mdluap.models import load_modelzoo_classifier
from mdluap.official_triggers import build_trigger_adapters
from mdluap.targeted_pgd import targeted_pgd_endpoint
from pilot_common import batch_images, seed_everything, timestamp_run_dir, write_csv, write_json


TARGET = 0
EPSILONS = (1.0, 1.5)
REFERENCE_ALIAS = "clean0"
CLEAN_CONTROLS = ("clean1", "clean2")
GROUPS = (
    ("badnet", "badnet0", "badnet"),
    ("blended", "blended0", "blended"),
    ("wanet", "wanet0", "wanet"),
    ("inputaware", "inputaware0", "inputaware"),
    ("ssba", "ssba0", "ssba"),
    ("adaptive_blend", "adaptive_blend01", "adaptive_blend"),
)
ALIASES = (REFERENCE_ALIAS, *CLEAN_CONTROLS, *(alias for _group, alias, _trigger in GROUPS))
EPS_NUM = 1e-12
ZERO_TOL = 1e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-zoo-root", required=True)
    parser.add_argument("--model-zoo-source-root", default=None)
    parser.add_argument("--selection-file", required=True)
    parser.add_argument("--cache-root", default="results", help="search existing result archives for compatible endpoints")
    parser.add_argument("--trigger-artifact-root", required=True)
    parser.add_argument("--backdoorbench-root", required=True)
    parser.add_argument("--output-root", default="results/stage1d_adversarial_residual_decoupling")
    parser.add_argument("--target", type=int, default=TARGET)
    parser.add_argument("--epsilon-pixels", default="1,1.5")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--random-seed", type=int, default=20260926)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--blended-alpha", type=float, default=0.2)
    parser.add_argument("--wanet-s", type=float, default=0.5)
    parser.add_argument("--wanet-grid-rescale", type=float, default=1.0)
    parser.add_argument("--badnet-trigger-path", default=None)
    parser.add_argument("--blended-trigger-path", default=None)
    parser.add_argument("--wanet-state-path", default=None)
    parser.add_argument("--inputaware-state-path", default=None)
    parser.add_argument("--adaptive-blend-trigger-path", default=None)
    parser.add_argument("--ssba-encoder-path", default=None)
    parser.add_argument("--ssba-config-path", default=None)
    return parser.parse_args()


def parse_epsilon_list(value: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if values != EPSILONS:
        raise ValueError(f"epsilon-pixels is fixed to {EPSILONS}, got {values}")
    return values


def sha256_file(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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


def read_selection(path: Path, dataset) -> tuple[list[int], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 100 or not rows or "sample_index" not in rows[0]:
        raise ValueError(f"selection must contain exactly 100 rows with sample_index: {path}")
    if "target" in rows[0] and any(int(row["target"]) != TARGET for row in rows):
        raise ValueError("selection file contains a target other than 0")
    if "rank" in rows[0]:
        rows.sort(key=lambda row: int(row["rank"]))
    indices = [int(row["sample_index"]) for row in rows]
    if len(indices) != len(set(indices)) or min(indices) < 0 or max(indices) >= len(dataset):
        raise ValueError("selection contains duplicate or out-of-range CIFAR-10 test indices")
    if "true_label" in rows[0]:
        for row in rows:
            if int(row["true_label"]) != int(dataset[int(row["sample_index"])][1]):
                raise ValueError(f"selection label mismatch at sample {row['sample_index']}")
    return indices, rows


def model_zoo_provenance(root: Path, source_root: Path | None) -> dict[str, Any]:
    os.environ["MODEL_ZOO_ROOT"] = str(root)
    from modelzoo import get_model_info, list_models

    registry = next((root / name for name in ("registry.yaml", "registry.yml") if (root / name).is_file()), None)
    if registry is None:
        matches = sorted(root.rglob("registry.yaml")) if root.is_dir() else []
        registry = matches[0] if matches else None
    try:
        package_version = importlib.metadata.version("backdoor-model-zoo")
    except importlib.metadata.PackageNotFoundError:
        package_version = None
    listed = list_models()
    infos: dict[str, Any] = {}
    info_errors: dict[str, str] = {}
    for alias in ALIASES:
        try:
            infos[alias] = get_model_info(alias)
        except Exception as exc:  # one missing alias must not hide the others
            info_errors[alias] = f"{type(exc).__name__}: {exc}"
    return {
        "package_version": package_version,
        "source_root": str(source_root) if source_root else None,
        "source_git_commit": git_commit(source_root),
        "registry_path": str(registry.resolve()) if registry else None,
        "registry_sha256": sha256_file(registry),
        "list_models": listed,
        "infos": infos,
        "info_errors": info_errors,
    }


def parse_bool(value: Any) -> bool:
    return value if isinstance(value, bool) else str(value).strip().lower() in {"1", "true", "yes"}


def load_endpoint_cache(
    cache_root: Path,
    indices: list[int],
    original: np.ndarray,
    *,
    alias: str,
    epsilon: float,
    steps: int,
    restarts: int,
    target: int,
) -> tuple[dict[int, dict[str, Any]] | None, str | None]:
    """Reuse only layerwise archives with an exact cohort/config match."""

    if not cache_root.is_dir():
        return None, None
    try:
        import yaml
    except ImportError:
        return None, None

    for archive_path in sorted(cache_root.rglob("endpoint_arrays.npz")):
        config_path = archive_path.parent / "config.resolved.yaml"
        records_path = archive_path.parent / "attack_records.csv"
        if not config_path.is_file() or not records_path.is_file():
            continue
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            if int(config.get("target", -1)) != target:
                continue
            if int(config.get("analysis_steps", -1)) != steps or int(config.get("analysis_restarts", -1)) != restarts:
                continue
            if not math.isclose(float(config.get("analysis_epsilon_pixels", -1)), epsilon, abs_tol=1e-9):
                continue
            if not math.isclose(float(config.get("analysis_epsilon_pixels", -1)) / 10.0, epsilon / 10.0, abs_tol=1e-9):
                continue
            backdoor_alias = str(config.get("backdoor_alias", ""))
            models = config.get("models", [])
            if alias not in models and alias != backdoor_alias:
                continue
            with np.load(archive_path, allow_pickle=False) as arrays:
                if not {"sample_indices", "original"}.issubset(arrays.files):
                    continue
                cached_indices = np.asarray(arrays["sample_indices"], dtype=np.int64)
                cached_original = np.asarray(arrays["original"], dtype=np.float32)
                if not np.array_equal(cached_indices, np.asarray(indices, dtype=np.int64)):
                    continue
                if cached_original.shape != original.shape or not np.allclose(cached_original, original, atol=1e-7, rtol=0):
                    continue
                key = "clean0_adv" if alias == "clean0" else "backdoor_adv"
                if key not in arrays.files or (alias != "clean0" and alias != backdoor_alias):
                    continue
                endpoints = np.asarray(arrays[key], dtype=np.float32)
            if endpoints.shape != original.shape:
                continue
            with records_path.open(newline="", encoding="utf-8") as handle:
                csv_rows = list(csv.DictReader(handle))
            by_index = {
                int(row["sample_index"]): row
                for row in csv_rows
                if row.get("model_alias") == alias
                and int(row.get("target", -1)) == target
                and math.isclose(float(row.get("epsilon_pixels", -1)), epsilon, abs_tol=1e-9)
            }
            if set(by_index) != set(indices):
                continue
            result: dict[int, dict[str, Any]] = {}
            for position, sample_index in enumerate(indices):
                row = by_index[sample_index]
                prediction = int(row["original_prediction"])
                eligible = prediction != target
                result[sample_index] = {
                    "endpoint": endpoints[position] if eligible else np.full_like(original[position], np.nan),
                    "original_prediction": prediction,
                    "eligible": eligible,
                    "success": parse_bool(row.get("success", False)) if eligible else None,
                    "endpoint_prediction": int(row.get("final_prediction", target if parse_bool(row.get("success", False)) else -1)),
                    "actual_linf": float(row.get("actual_linf", "nan")) if eligible else float("nan"),
                    "actual_l2": float(row.get("actual_l2", "nan")) if eligible else float("nan"),
                    "targeted_loss": float(row.get("targeted_loss", "nan")) if eligible else float("nan"),
                    "cache_source": str(archive_path.resolve()),
                }
            return result, str(archive_path.resolve())
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
    return None, None


def run_pgd_for_alias(
    model,
    dataset,
    indices: list[int],
    predictions: dict[int, int],
    *,
    epsilon: float,
    steps: int,
    restarts: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[int, dict[str, Any]]:
    seed_everything(seed)
    output: dict[int, dict[str, Any]] = {}
    eligible_indices = [index for index in indices if predictions[index] != TARGET]
    for batch_indices, images, _labels in batch_images(dataset, eligible_indices, batch_size=batch_size, device=device):
        attacked = targeted_pgd_endpoint(
            model,
            images,
            target=TARGET,
            epsilon=epsilon / 255.0,
            steps=steps,
            alpha=epsilon / 2550.0,
            random_start=True,
            restarts=restarts,
        )
        endpoints = attacked.endpoint.detach().cpu().numpy().astype(np.float32)
        for offset, sample_index in enumerate(batch_indices):
            delta = endpoints[offset] - images[offset].detach().cpu().numpy()
            output[int(sample_index)] = {
                "endpoint": endpoints[offset],
                "original_prediction": int(predictions[int(sample_index)]),
                "eligible": True,
                "success": bool(attacked.success[offset].item()),
                "endpoint_prediction": int(attacked.endpoint_prediction[offset].item()),
                "actual_linf": float(attacked.endpoint_linf[offset].item()),
                "actual_l2": float(np.linalg.norm(delta.reshape(-1))),
                "targeted_loss": float(attacked.target_loss[offset].item()),
                "cache_source": None,
            }
    for sample_index in indices:
        if sample_index not in output:
            output[sample_index] = {
                "endpoint": None,
                "original_prediction": int(predictions[sample_index]),
                "eligible": False,
                "success": None,
                "endpoint_prediction": int(predictions[sample_index]),
                "actual_linf": float("nan"),
                "actual_l2": float("nan"),
                "targeted_loss": float("nan"),
                "cache_source": None,
            }
    return output


def projection_residual(delta: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project each row off its paired Clean0 adversarial direction."""

    if delta.ndim != 2 or reference.ndim != 2 or delta.shape != reference.shape:
        raise ValueError(f"projection expects equal [N,D] arrays, got {delta.shape} and {reference.shape}")
    finite = np.isfinite(delta).all(axis=1) & np.isfinite(reference).all(axis=1)
    delta_safe = np.where(np.isfinite(delta), delta, 0.0)
    ref_safe = np.where(np.isfinite(reference), reference, 0.0)
    delta_norm = np.linalg.norm(delta_safe, axis=1)
    ref_sq = np.einsum("nd,nd->n", ref_safe, ref_safe)
    valid = finite & (delta_norm > ZERO_TOL) & (ref_sq > ZERO_TOL**2)
    alpha = np.full(len(delta), np.nan, dtype=np.float64)
    alpha[valid] = np.einsum("nd,nd->n", delta_safe[valid], ref_safe[valid]) / (ref_sq[valid] + EPS_NUM)
    residual = np.full_like(delta_safe, np.nan, dtype=np.float64)
    residual[valid] = delta_safe[valid] - alpha[valid, None] * ref_safe[valid]
    projection_ratio = np.full(len(delta), np.nan, dtype=np.float64)
    projection_ratio[valid] = np.linalg.norm(alpha[valid, None] * ref_safe[valid], axis=1) / (delta_norm[valid] + EPS_NUM)
    # The reference compared with itself has mathematically zero residual.
    residual_norm = np.linalg.norm(np.nan_to_num(residual, nan=0.0), axis=1)
    residual_valid = valid & (residual_norm > ZERO_TOL)
    orth_cosine = np.full(len(delta), np.nan, dtype=np.float64)
    orthogonal_dot = np.full(len(delta), np.nan, dtype=np.float64)
    nonzero = valid & (residual_norm > ZERO_TOL)
    if np.any(nonzero):
        dot = np.einsum("nd,nd->n", residual[nonzero], ref_safe[nonzero])
        orthogonal_dot[nonzero] = np.abs(dot)
        orth_cosine[nonzero] = np.abs(dot) / (
            residual_norm[nonzero] * np.sqrt(ref_sq[nonzero]) + EPS_NUM
        )
    return residual, alpha, projection_ratio, np.stack((orth_cosine, orthogonal_dot, residual_valid.astype(float)), axis=1)


def vector_valid(vectors: np.ndarray) -> np.ndarray:
    flat = vectors.reshape(len(vectors), -1)
    return np.isfinite(flat).all(axis=1) & (np.linalg.norm(np.nan_to_num(flat, nan=0.0), axis=1) > ZERO_TOL)


def pairwise_cosine_stats(vectors: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    flat = vectors.reshape(len(vectors), -1).astype(np.float64)
    valid = vector_valid(flat)
    if mask is not None:
        valid &= mask.astype(bool)
    chosen = flat[valid]
    if len(chosen) < 2:
        return {"mean": None, "std": None, "median": None, "valid_pair_count": 0, "valid_vector_count": int(len(chosen))}
    normalized = chosen / np.linalg.norm(chosen, axis=1, keepdims=True)
    pair_values = (normalized @ normalized.T)[np.triu_indices(len(chosen), k=1)]
    return {
        "mean": float(np.mean(pair_values)),
        "std": float(np.std(pair_values)),
        "median": float(np.median(pair_values)),
        "valid_pair_count": int(len(pair_values)),
        "valid_vector_count": int(len(chosen)),
    }


def row_cosines(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError(f"cosine inputs must have equal [N,D] shapes, got {left.shape}, {right.shape}")
    left_norm = np.linalg.norm(np.nan_to_num(left, nan=0.0), axis=1)
    right_norm = np.linalg.norm(np.nan_to_num(right, nan=0.0), axis=1)
    valid = np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1) & (left_norm > ZERO_TOL) & (right_norm > ZERO_TOL)
    values = np.full(len(left), np.nan, dtype=np.float64)
    values[valid] = np.einsum("nd,nd->n", left[valid], right[valid]) / (left_norm[valid] * right_norm[valid])
    return values


def scalar_stats(values: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(values)
    chosen = values[valid]
    if not len(chosen):
        return {"mean": None, "std": None, "median": None, "sample_count": 0}
    return {"mean": float(chosen.mean()), "std": float(chosen.std()), "median": float(np.median(chosen)), "sample_count": int(len(chosen))}


def metric_row(
    *, group: str, alias: str, epsilon: float, deltas: dict[str, np.ndarray],
    successes: dict[str, np.ndarray], eligibility: dict[str, np.ndarray],
    trigger_delta: np.ndarray | None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    reference = deltas[REFERENCE_ALIAS]
    current = deltas[alias]
    residual, alpha, projection_ratio, orth = projection_residual(current, reference)
    raw_valid = vector_valid(current)
    residual_valid = vector_valid(residual)
    ref_success = successes[REFERENCE_ALIAS] & eligibility[REFERENCE_ALIAS]
    model_success = successes[alias] & eligibility[alias]
    common = ref_success & model_success
    raw_conc_all = pairwise_cosine_stats(current, raw_valid)
    residual_conc_all = pairwise_cosine_stats(residual, residual_valid)
    residual_conc_common = pairwise_cosine_stats(residual, residual_valid & common)
    raw_alignment = row_cosines(current, trigger_delta) if trigger_delta is not None else np.full(len(current), np.nan)
    residual_alignment = row_cosines(residual, trigger_delta) if trigger_delta is not None else np.full(len(current), np.nan)
    raw_alignment_common = raw_alignment.copy()
    raw_alignment_common[~common] = np.nan
    residual_alignment_common = residual_alignment.copy()
    residual_alignment_common[~common] = np.nan
    raw_alignment_stats = scalar_stats(raw_alignment)
    residual_alignment_stats = scalar_stats(residual_alignment)
    raw_alignment_common_stats = scalar_stats(raw_alignment_common)
    residual_alignment_common_stats = scalar_stats(residual_alignment_common)
    valid_projection = np.isfinite(projection_ratio)
    orth_cos = orth[:, 0]
    orth_dot = orth[:, 1]
    orth_values = {
        "raw_valid": raw_valid,
        "residual_valid": residual_valid,
        "common_success": common,
        "alpha": alpha,
        "projection_ratio": projection_ratio,
        "orthogonality_cosine": orth_cos,
        "orthogonality_dot": orth_dot,
        "raw_alignment": raw_alignment,
        "residual_alignment": residual_alignment,
    }
    row = {
        "target": TARGET,
        "epsilon_pixels": epsilon,
        "reference_model": REFERENCE_ALIAS,
        "model": alias,
        "backdoor_group": group,
        "n_total": len(current),
        "n_success_ref": int(ref_success.sum()),
        "n_success_model": int(model_success.sum()),
        "n_common_success": int(common.sum()),
        "raw_concentration_mean": raw_conc_all["mean"],
        "raw_concentration_std": raw_conc_all["std"],
        "raw_concentration_median": raw_conc_all["median"],
        "raw_concentration_valid_pair_count": raw_conc_all["valid_pair_count"],
        "raw_valid_sample_count": raw_conc_all["valid_vector_count"],
        "residual_concentration_all_mean": residual_conc_all["mean"],
        "residual_concentration_all_std": residual_conc_all["std"],
        "residual_concentration_all_median": residual_conc_all["median"],
        "residual_concentration_all_valid_pair_count": residual_conc_all["valid_pair_count"],
        "residual_concentration_all_valid_sample_count": residual_conc_all["valid_vector_count"],
        "residual_concentration_common_success_mean": residual_conc_common["mean"],
        "residual_concentration_common_success_std": residual_conc_common["std"],
        "residual_concentration_common_success_median": residual_conc_common["median"],
        "residual_concentration_common_success_valid_pair_count": residual_conc_common["valid_pair_count"],
        "residual_concentration_common_success_valid_sample_count": residual_conc_common["valid_vector_count"],
        "raw_trigger_alignment_mean": raw_alignment_stats["mean"],
        "raw_trigger_alignment_std": raw_alignment_stats["std"],
        "raw_trigger_alignment_median": raw_alignment_stats["median"],
        "raw_trigger_alignment_sample_count": raw_alignment_stats["sample_count"],
        "residual_trigger_alignment_mean": residual_alignment_stats["mean"],
        "residual_trigger_alignment_std": residual_alignment_stats["std"],
        "residual_trigger_alignment_median": residual_alignment_stats["median"],
        "residual_trigger_alignment_sample_count": residual_alignment_stats["sample_count"],
        "raw_trigger_alignment_common_success_mean": raw_alignment_common_stats["mean"],
        "raw_trigger_alignment_common_success_std": raw_alignment_common_stats["std"],
        "raw_trigger_alignment_common_success_median": raw_alignment_common_stats["median"],
        "raw_trigger_alignment_common_success_sample_count": raw_alignment_common_stats["sample_count"],
        "residual_trigger_alignment_common_success_mean": residual_alignment_common_stats["mean"],
        "residual_trigger_alignment_common_success_std": residual_alignment_common_stats["std"],
        "residual_trigger_alignment_common_success_median": residual_alignment_common_stats["median"],
        "residual_trigger_alignment_common_success_sample_count": residual_alignment_common_stats["sample_count"],
        "mean_residual_norm": float(np.nanmean(np.linalg.norm(np.nan_to_num(residual, nan=0.0), axis=1)[np.isfinite(alpha)])) if np.isfinite(alpha).any() else None,
        "mean_projection_ratio": float(np.nanmean(projection_ratio[valid_projection])) if valid_projection.any() else None,
        "projection_sample_count": int(valid_projection.sum()),
        "orthogonality_check": float(np.nanmean(orth_cos)) if np.isfinite(orth_cos).any() else None,
        "orthogonality_abs_dot_mean": float(np.nanmean(orth_dot)) if np.isfinite(orth_dot).any() else None,
        "orthogonality_sample_count": int(np.isfinite(orth_cos).sum()),
        "zero_norm_raw_count": int((~raw_valid).sum()),
        "zero_norm_residual_count": int((~residual_valid & np.isfinite(alpha)).sum()),
        "trigger_alignment_available": trigger_delta is not None,
    }
    return row, orth_values


def main() -> None:
    args = parse_args()
    if args.target != TARGET or args.steps <= 0 or args.restarts <= 0 or args.batch_size <= 0:
        raise ValueError("target is fixed to 0; PGD steps, restarts, and batch size must be positive")
    epsilons = parse_epsilon_list(args.epsilon_pixels)
    seed_everything(args.random_seed)
    device = torch.device(args.device)
    data_root = Path(args.data_root).expanduser().resolve()
    zoo_root = Path(args.model_zoo_root).expanduser().resolve()
    zoo_source = Path(args.model_zoo_source_root).expanduser().resolve() if args.model_zoo_source_root else None
    selection_path = Path(args.selection_file).expanduser().resolve()
    cache_root = Path(args.cache_root).expanduser().resolve()
    trigger_root = Path(args.trigger_artifact_root).expanduser().resolve()
    bdb_root = Path(args.backdoorbench_root).expanduser().resolve()
    for required_path in (selection_path,):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)

    dataset = cifar10_dataset(data_root, train=False)
    indices, selection_rows = read_selection(selection_path, dataset)
    original = np.stack([dataset[index][0].numpy().astype(np.float32) for index in indices])
    labels = np.asarray([int(dataset[index][1]) for index in indices], dtype=np.int64)
    if original.shape != (100, 3, 32, 32) or not np.isfinite(original).all():
        raise ValueError(f"expected finite CIFAR-10 Top-100 pixels [100,3,32,32], got {original.shape}")
    if float(original.min()) < 0.0 or float(original.max()) > 1.0:
        raise ValueError("CIFAR-10 inputs must be raw pixels in [0,1]")

    provenance = model_zoo_provenance(zoo_root, zoo_source)
    trigger_paths: dict[str, Path | None] = {
        "badnet": Path(args.badnet_trigger_path).expanduser().resolve() if args.badnet_trigger_path else None,
        "blended": Path(args.blended_trigger_path).expanduser().resolve() if args.blended_trigger_path else None,
        "wanet_identity": None,
        "wanet_noise": None,
        "inputaware": Path(args.inputaware_state_path).expanduser().resolve() if args.inputaware_state_path else None,
        "adaptive_blend": Path(args.adaptive_blend_trigger_path).expanduser().resolve() if args.adaptive_blend_trigger_path else None,
        "ssba_encoder": Path(args.ssba_encoder_path).expanduser().resolve() if args.ssba_encoder_path else None,
        "ssba_config": Path(args.ssba_config_path).expanduser().resolve() if args.ssba_config_path else None,
    }
    if args.wanet_state_path:
        wanet_state = Path(args.wanet_state_path).expanduser().resolve()
        if wanet_state.name == "state_identity_grid.pt":
            trigger_paths["wanet_identity"] = wanet_state
        elif wanet_state.name == "state_noise_grid.pt":
            trigger_paths["wanet_noise"] = wanet_state
    adapters = build_trigger_adapters(
        model_root=trigger_root,
        backdoorbench_root=bdb_root,
        explicit=trigger_paths,
        device=device,
        blended_alpha=args.blended_alpha,
        wanet_s=args.wanet_s,
        wanet_grid_rescale=args.wanet_grid_rescale,
    )
    group_trigger_images: dict[str, np.ndarray] = {}
    group_trigger_status: dict[str, dict[str, Any]] = {}
    for group, _alias, trigger_type in GROUPS:
        adapter = adapters[trigger_type]
        status = {"trigger_type": trigger_type, "source": adapter.status.source, "available": adapter.status.available, "reason": adapter.status.reason}
        if adapter.status.available:
            try:
                chunks: list[np.ndarray] = []
                for batch_indices, images, _batch_labels in batch_images(dataset, indices, batch_size=args.batch_size, device=device):
                    transformed = adapter.apply(images, sample_indices=batch_indices, split="cifar10_test")
                    if transformed.shape != images.shape:
                        raise ValueError(f"trigger changed tensor shape: {tuple(images.shape)} -> {tuple(transformed.shape)}")
                    chunks.append(transformed.detach().cpu().numpy().astype(np.float32))
                transformed = np.concatenate(chunks, axis=0)
                if transformed.shape != original.shape or not np.isfinite(transformed).all():
                    raise ValueError("trigger output shape/non-finite validation failed")
                group_trigger_images[group] = transformed
            except Exception as exc:
                status["available"] = False
                status["reason"] = f"{type(exc).__name__}: {exc}"
        group_trigger_status[group] = status

    output = timestamp_run_dir(args.output_root, "residual_decoupling")
    log_path = output / "run.log"

    def log(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    config = vars(args).copy()
    config.update({
        "protocol": "stage1d-pixel-adversarial-residual-decoupling-v1",
        "dataset": "CIFAR-10 test; one existing shared Probe Top-100 selection",
        "selected_count": len(indices),
        "target": TARGET,
        "models": list(ALIASES),
        "epsilon_pixels": list(epsilons),
        "pgd": {"steps": args.steps, "restarts": args.restarts, "random_start": True, "alpha": "epsilon/10"},
        "input_space": "raw float32 CIFAR-10 pixels in [0,1]; normalization applied once by Model Zoo classifier wrapper",
        "projection": "paired pixel residual projected off each sample's clean0 PGD residual; eps_num=1e-12",
        "cohort_policy": "the same selected 100 CIFAR-10 indices for every alias and epsilon; report all-valid and clean0/current common-success subsets",
        "trigger_status": group_trigger_status,
        "trigger_adapters": {group: status.get("source") for group, status in group_trigger_status.items()},
        "model_zoo_provenance": provenance,
        "ssba_note": "Uses the existing official SSBA encoder/provenance adapter; its documented tiny quantization reproduction discrepancy is recorded, without replacing the attack trigger.",
    })
    import yaml
    (output / "config.resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    write_csv(output / "cohort.csv", [
        {"rank": rank, "sample_index": index, "true_label": int(labels[position]), **selection_rows[position]}
        for position, (rank, index) in enumerate(zip(range(1, 101), indices))
    ])

    cache_hits: dict[str, dict[float, str | None]] = {alias: {} for alias in ALIASES}
    attack_by_alias_eps: dict[str, dict[float, dict[int, dict[str, Any]]]] = {alias: {} for alias in ALIASES}
    prediction_by_alias: dict[str, dict[int, int]] = {}
    trigger_prediction_by_group_alias: dict[str, dict[str, dict[int, int]]] = {group: {} for group, _alias, _trig in GROUPS}
    model_quality: list[dict[str, Any]] = []
    cached_endpoint_arrays: dict[str, np.ndarray] = {"sample_indices": np.asarray(indices, dtype=np.int64), "original": original}
    cache_summary: list[dict[str, Any]] = []

    log(f"Residual decoupling analysis started: {output}")
    for alias_index, alias in enumerate(ALIASES):
        info = provenance["infos"].get(alias)
        if info is None:
            reason = provenance["info_errors"].get(alias, "get_model_info returned no metadata")
            model_quality.append({"model_alias": alias, "status": "unavailable", "reason": reason, "metadata_json": None})
            log(f"SKIP {alias}: {reason}")
            continue
        try:
            model, loaded_info = load_modelzoo_classifier(alias, model_zoo_root=str(zoo_root), device=device)
            if tuple(model(torch.from_numpy(original[:1]).to(device)).shape) != (1, 10):
                raise ValueError("Model Zoo output shape is not [1,10]")
            predictions: dict[int, int] = {}
            trigger_predictions_for_alias: dict[str, dict[int, int]] = {group: {} for group, _bd, _trig in GROUPS}
            with torch.no_grad():
                for batch_indices, images, _batch_labels in batch_images(dataset, indices, batch_size=args.batch_size, device=device):
                    logits = model(images)
                    if tuple(logits.shape) != (len(batch_indices), 10):
                        raise ValueError(f"{alias} output must be [N,10], got {tuple(logits.shape)}")
                    predictions.update({int(i): int(p) for i, p in zip(batch_indices, logits.argmax(1).cpu().tolist())})
                for group, _bd_alias, _trigger_type in GROUPS:
                    if group not in group_trigger_images:
                        continue
                    transformed = group_trigger_images[group]
                    group_preds: dict[int, int] = {}
                    for start in range(0, len(indices), args.batch_size):
                        stop = min(start + args.batch_size, len(indices))
                        batch = torch.from_numpy(transformed[start:stop]).to(device)
                        logits = model(batch)
                        if tuple(logits.shape) != (stop - start, 10):
                            raise ValueError(f"{alias} trigger output must be [N,10], got {tuple(logits.shape)}")
                        pred = logits.argmax(1).cpu().tolist()
                        group_preds.update({int(i): int(p) for i, p in zip(indices[start:stop], pred)})
                    trigger_predictions_for_alias[group] = group_preds
            for eps_index, epsilon in enumerate(epsilons):
                cache, cache_source = load_endpoint_cache(
                    cache_root, indices, original, alias=alias, epsilon=epsilon,
                    steps=args.steps, restarts=args.restarts, target=TARGET,
                )
                if cache is not None and all(cache[index]["original_prediction"] == predictions[index] for index in indices):
                    attacks = cache
                    cache_hits[alias][epsilon] = cache_source
                    cache_summary.append({"model_alias": alias, "epsilon_pixels": epsilon, "reused": True, "source": cache_source})
                else:
                    attacks = run_pgd_for_alias(
                        model, dataset, indices, predictions,
                        epsilon=epsilon, steps=args.steps, restarts=args.restarts,
                        batch_size=args.batch_size, device=device,
                        seed=args.random_seed + alias_index * 1000 + eps_index,
                    )
                    cache_hits[alias][epsilon] = None
                    cache_summary.append({"model_alias": alias, "epsilon_pixels": epsilon, "reused": False, "source": None})
                attack_by_alias_eps[alias][epsilon] = attacks
                key = f"{alias}__eps{epsilon:g}"
                cached_endpoint_arrays[f"{key}__endpoint"] = np.stack([
                    attacks[index]["endpoint"] if attacks[index]["endpoint"] is not None else np.full_like(original[position], np.nan)
                    for position, index in enumerate(indices)
                ])
                cached_endpoint_arrays[f"{key}__success"] = np.asarray([
                    -1 if attacks[index]["success"] is None else int(attacks[index]["success"]) for index in indices
                ], dtype=np.int8)
            prediction_by_alias[alias] = predictions
            for group in trigger_predictions_for_alias:
                trigger_prediction_by_group_alias[group][alias] = trigger_predictions_for_alias[group]
            model_quality.append({
                "model_alias": alias,
                "status": "loaded",
                "reason": None,
                "model_type": loaded_info.get("model_type"),
                "attack": loaded_info.get("attack"),
                "target_class": loaded_info.get("target_class"),
                "classifier_seed": loaded_info.get("classifier_seed"),
                "clean_acc": loaded_info.get("clean_acc"),
                "native_asr": loaded_info.get("asr"),
                "metadata_json": json.dumps(info, ensure_ascii=False, default=str),
            })
            log(f"{alias}: loaded; PGD endpoints ready for eps={','.join(map(str, epsilons))}")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            model_quality.append({"model_alias": alias, "status": "unavailable", "reason": reason, "metadata_json": json.dumps(info, ensure_ascii=False, default=str)})
            log(f"SKIP {alias}: {reason}")
            if alias == REFERENCE_ALIAS:
                raise RuntimeError(f"reference model clean0 is required for projection: {reason}") from exc

    model_quality_path = output / "model_quality.csv"
    write_csv(model_quality_path, model_quality)
    if REFERENCE_ALIAS not in prediction_by_alias:
        raise RuntimeError("clean0 is unavailable; no reference directions can be formed")

    attack_records: list[dict[str, Any]] = []
    for alias in ALIASES:
        if alias not in prediction_by_alias:
            continue
        for epsilon in epsilons:
            for position, sample_index in enumerate(indices):
                attack = attack_by_alias_eps[alias][epsilon][sample_index]
                attack_records.append({
                    "model_alias": alias,
                    "target": TARGET,
                    "epsilon_pixels": epsilon,
                    "sample_index": sample_index,
                    "true_label": int(labels[position]),
                    "original_prediction": attack["original_prediction"],
                    "eligible": attack["eligible"],
                    "pgd_status": "ineligible_original_target" if not attack["eligible"] else ("success" if attack["success"] else "failure"),
                    "success": attack["success"],
                    "endpoint_prediction": attack["endpoint_prediction"],
                    "actual_linf": attack["actual_linf"],
                    "actual_linf_pixels": attack["actual_linf"] * 255 if np.isfinite(attack["actual_linf"]) else None,
                    "actual_l2": attack["actual_l2"],
                    "targeted_loss": attack["targeted_loss"],
                    "endpoint_source": attack.get("cache_source"),
                })
    write_csv(output / "attack_records.csv", attack_records)
    write_csv(output / "trigger_records.csv", [
        {
            "backdoor_group": group,
            "trigger_type": trigger_type,
            "model_alias": alias,
            "sample_index": sample_index,
            "trigger_prediction": trigger_prediction_by_group_alias.get(group, {}).get(alias, {}).get(sample_index),
            "trigger_success_target0": trigger_prediction_by_group_alias.get(group, {}).get(alias, {}).get(sample_index) == TARGET,
            "trigger_source": group_trigger_status[group].get("source"),
            "alignment_available": group in group_trigger_images and alias in trigger_prediction_by_group_alias.get(group, {}),
            "alignment_unavailable_reason": group_trigger_status[group].get("reason"),
        }
        for group, _bd_alias, trigger_type in GROUPS
        for alias in ALIASES
        for sample_index in indices
    ])

    group_metrics: list[dict[str, Any]] = []
    per_sample: list[dict[str, Any]] = []
    concise: list[dict[str, Any]] = []
    for group, bd_alias, _trigger_type in GROUPS:
        trigger_array = group_trigger_images.get(group)
        trigger_delta = (trigger_array - original).reshape(len(indices), -1).astype(np.float64) if trigger_array is not None else None
        aliases_for_group = [alias for alias in (REFERENCE_ALIAS, *CLEAN_CONTROLS, bd_alias) if alias in prediction_by_alias]
        for epsilon in epsilons:
            deltas: dict[str, np.ndarray] = {}
            success: dict[str, np.ndarray] = {}
            eligible: dict[str, np.ndarray] = {}
            for alias in aliases_for_group:
                attacks = attack_by_alias_eps[alias][epsilon]
                values = []
                flags = []
                eligibility_mask = []
                for sample_index in indices:
                    attack = attacks[sample_index]
                    endpoint = attack["endpoint"]
                    row_index = indices.index(sample_index)
                    values.append(np.zeros_like(original[0]).reshape(-1) if endpoint is None else (endpoint - original[row_index]).reshape(-1))
                    flags.append(bool(attack["success"]) if attack["success"] is not None else False)
                    eligibility_mask.append(bool(attack["eligible"]))
                deltas[alias] = np.stack(values).astype(np.float64)
                # Ineligible/no endpoint is represented as NaN, never as a zero perturbation.
                for row_index, is_eligible in enumerate(eligibility_mask):
                    if not is_eligible:
                        deltas[alias][row_index] = np.nan
                success[alias] = np.asarray(flags, dtype=bool)
                eligible[alias] = np.asarray(eligibility_mask, dtype=bool)

            ref_success = success[REFERENCE_ALIAS]
            ref_eligible = eligible[REFERENCE_ALIAS]
            ref_delta = deltas[REFERENCE_ALIAS]
            for alias in aliases_for_group:
                rows_for_alias = deltas.copy()
                rows_for_alias[alias] = deltas[alias]
                row, sample_masks = metric_row(
                    group=group, alias=alias, epsilon=epsilon,
                    deltas=rows_for_alias, successes=success, eligibility=eligible,
                    trigger_delta=trigger_delta,
                )
                row["reference_model_success"] = int((ref_success & ref_eligible).sum())
                row["model_success"] = int((success[alias] & eligible[alias]).sum())
                row["trigger_source"] = group_trigger_status[group].get("source")
                row["trigger_alignment_unavailable_reason"] = group_trigger_status[group].get("reason")
                group_metrics.append(row)
                raw_delta = deltas[alias]
                residual, alpha, ratio, orth = projection_residual(raw_delta, ref_delta)
                trig_cos_raw = sample_masks["raw_alignment"]
                trig_cos_res = sample_masks["residual_alignment"]
                for position, sample_index in enumerate(indices):
                    attack = attack_by_alias_eps[alias][epsilon][sample_index]
                    ref_attack = attack_by_alias_eps[REFERENCE_ALIAS][epsilon][sample_index]
                    per_sample.append({
                        "backdoor_group": group,
                        "model_alias": alias,
                        "target": TARGET,
                        "epsilon_pixels": epsilon,
                        "sample_index": sample_index,
                        "true_label": int(labels[position]),
                        "original_prediction": attack["original_prediction"],
                        "eligible": attack["eligible"],
                        "pgd_success": attack["success"],
                        "reference_original_prediction": ref_attack["original_prediction"],
                        "reference_eligible": ref_attack["eligible"],
                        "reference_pgd_success": ref_attack["success"],
                        "common_success": bool(ref_attack["success"] and attack["success"] and ref_attack["eligible"] and attack["eligible"]),
                        "raw_delta_norm": float(np.linalg.norm(np.nan_to_num(raw_delta[position], nan=0.0))) if np.isfinite(raw_delta[position]).all() else None,
                        "projection_coefficient": float(alpha[position]) if np.isfinite(alpha[position]) else None,
                        "projection_ratio": float(ratio[position]) if np.isfinite(ratio[position]) else None,
                        "residual_norm": float(np.linalg.norm(np.nan_to_num(residual[position], nan=0.0))) if np.isfinite(residual[position]).all() else None,
                        "residual_valid": bool(sample_masks["residual_valid"][position]),
                        "orthogonality_abs_cosine": float(orth[position, 0]) if np.isfinite(orth[position, 0]) else None,
                        "orthogonality_abs_dot": float(orth[position, 1]) if np.isfinite(orth[position, 1]) else None,
                        "raw_trigger_alignment": float(trig_cos_raw[position]) if np.isfinite(trig_cos_raw[position]) else None,
                        "residual_trigger_alignment": float(trig_cos_res[position]) if np.isfinite(trig_cos_res[position]) else None,
                        "trigger_prediction": trigger_prediction_by_group_alias.get(group, {}).get(alias, {}).get(sample_index),
                        "endpoint_source": attack.get("cache_source"),
                    })

            by_alias = {row["model"]: row for row in group_metrics if row["backdoor_group"] == group and row["epsilon_pixels"] == epsilon}
            clean_res = [by_alias.get(alias, {}).get("residual_concentration_common_success_mean") for alias in CLEAN_CONTROLS]
            clean_res = [float(value) for value in clean_res if value is not None]
            clean_res_alignment = [by_alias.get(alias, {}).get("residual_trigger_alignment_common_success_mean") for alias in CLEAN_CONTROLS]
            clean_res_alignment = [float(value) for value in clean_res_alignment if value is not None]
            bd = by_alias.get(bd_alias, {})
            clean_raw = [by_alias.get(alias, {}).get("raw_concentration_mean") for alias in CLEAN_CONTROLS]
            clean_raw = [float(value) for value in clean_raw if value is not None]
            clean_raw_alignment = [by_alias.get(alias, {}).get("raw_trigger_alignment_common_success_mean") for alias in CLEAN_CONTROLS]
            clean_raw_alignment = [float(value) for value in clean_raw_alignment if value is not None]
            residual_conc_delta = (bd.get("residual_concentration_common_success_mean") - float(np.mean(clean_res))) if bd.get("residual_concentration_common_success_mean") is not None and clean_res else None
            residual_align_delta = (bd.get("residual_trigger_alignment_common_success_mean") - float(np.mean(clean_res_alignment))) if bd.get("residual_trigger_alignment_common_success_mean") is not None and clean_res_alignment else None
            concise.append({
                "backdoor_group": group,
                "backdoor_alias": bd_alias,
                "epsilon_pixels": epsilon,
                "raw_concentration_backdoor": bd.get("raw_concentration_mean"),
                "raw_concentration_clean_baseline": float(np.mean(clean_raw)) if clean_raw else None,
                "raw_concentration_backdoor_minus_clean": (bd.get("raw_concentration_mean") - float(np.mean(clean_raw))) if bd.get("raw_concentration_mean") is not None and clean_raw else None,
                "residual_concentration_backdoor_common_success": bd.get("residual_concentration_common_success_mean"),
                "residual_concentration_clean_baseline_common_success": float(np.mean(clean_res)) if clean_res else None,
                "residual_concentration_backdoor_minus_clean": residual_conc_delta,
                "raw_trigger_alignment_backdoor": bd.get("raw_trigger_alignment_common_success_mean"),
                "raw_trigger_alignment_clean_baseline": float(np.mean(clean_raw_alignment)) if clean_raw_alignment else None,
                "residual_trigger_alignment_backdoor": bd.get("residual_trigger_alignment_common_success_mean"),
                "residual_trigger_alignment_clean_baseline": float(np.mean(clean_res_alignment)) if clean_res_alignment else None,
                "residual_alignment_gain_backdoor": (bd.get("residual_trigger_alignment_common_success_mean") - bd.get("raw_trigger_alignment_common_success_mean")) if bd.get("residual_trigger_alignment_common_success_mean") is not None and bd.get("raw_trigger_alignment_common_success_mean") is not None else None,
                "residual_alignment_backdoor_minus_clean": residual_align_delta,
                "n_common_success": bd.get("n_common_success"),
                "trigger_alignment_available": group in group_trigger_images,
                "status": "complete" if bd else "backdoor_model_unavailable",
            })

    write_csv(output / "group_metrics.csv", group_metrics)
    write_csv(output / "per_sample_metrics.csv", per_sample)
    write_csv(output / "summary_comparisons.csv", concise)
    for group, trigger_images in group_trigger_images.items():
        cached_endpoint_arrays[f"trigger__{group}"] = trigger_images
    np.savez_compressed(output / "endpoint_arrays.npz", **cached_endpoint_arrays)

    bd_rows = [row for row in concise if row["status"] == "complete"]
    q2 = [row for row in bd_rows if row["residual_concentration_backdoor_minus_clean"] is not None and row["residual_concentration_backdoor_minus_clean"] > 0]
    q3 = [row for row in bd_rows if row["residual_alignment_gain_backdoor"] is not None and row["residual_alignment_gain_backdoor"] > 0]
    q4: dict[str, dict[str, Any]] = {}
    for group, alias, _trigger in GROUPS:
        rows = [row for row in bd_rows if row["backdoor_group"] == group]
        q4[group] = {
            "alias": alias,
            "epsilons_with_residual_concentration_above_clean": [row["epsilon_pixels"] for row in rows if row["residual_concentration_backdoor_minus_clean"] is not None and row["residual_concentration_backdoor_minus_clean"] > 0],
            "epsilons_with_residual_alignment_gain": [row["epsilon_pixels"] for row in rows if row["residual_alignment_gain_backdoor"] is not None and row["residual_alignment_gain_backdoor"] > 0],
            "alignment_available": all(row["trigger_alignment_available"] for row in rows) if rows else False,
        }
    summary = {
        "status": "complete" if bd_rows else "no_backdoor_comparisons_completed",
        "protocol": config["protocol"],
        "output_directory": str(output.resolve()),
        "selected_count": len(indices),
        "selected_indices": indices,
        "epsilon_pixels": list(epsilons),
        "pgd": config["pgd"],
        "cache_reuse": cache_summary,
        "cache_reused_count": sum(bool(item["reused"]) for item in cache_summary),
        "cache_miss_count": sum(not item["reused"] for item in cache_summary),
        "model_zoo_provenance": provenance,
        "trigger_status": group_trigger_status,
        "model_quality": model_quality,
        "summary_comparisons": concise,
        "Q1_raw_direction": "Compare raw_concentration_backdoor_minus_clean in summary_comparisons; no equivalence threshold was prespecified.",
        "Q2_residual_concentration": {"positive_backdoor_epsilon_count": len(q2), "total_completed_backdoor_epsilon_comparisons": len(bd_rows), "comparisons": [{"group": row["backdoor_group"], "epsilon_pixels": row["epsilon_pixels"], "delta": row["residual_concentration_backdoor_minus_clean"], "n_common_success": row["n_common_success"]} for row in q2]},
        "Q3_trigger_alignment_gain": {"positive_backdoor_epsilon_count": len(q3), "comparisons": [{"group": row["backdoor_group"], "epsilon_pixels": row["epsilon_pixels"], "gain": row["residual_alignment_gain_backdoor"]} for row in q3]},
        "Q4_attack_types": q4,
        "interpretation_boundary": "single fixed Top-100 cohort; descriptive pixel-space mechanism check, no claim of statistical significance or literal causal path",
        "ssba_note": config["ssba_note"],
    }
    write_json(output / "summary.json", summary)
    log(f"Compatible endpoint caches reused: {summary['cache_reused_count']}/{len(cache_summary)}")
    for row in concise:
        log(
            f"{row['backdoor_group']} eps={row['epsilon_pixels']}: "
            f"Cdelta BD/Clean={row['raw_concentration_backdoor']}/{row['raw_concentration_clean_baseline']} "
            f"Cr(common) BD/Clean={row['residual_concentration_backdoor_common_success']}/{row['residual_concentration_clean_baseline_common_success']} "
            f"align raw/resid={row['raw_trigger_alignment_backdoor']}/{row['residual_trigger_alignment_backdoor']} "
            f"n_common={row['n_common_success']}"
        )
    log(f"Residual decoupling analysis complete: {output}")


if __name__ == "__main__":
    main()
