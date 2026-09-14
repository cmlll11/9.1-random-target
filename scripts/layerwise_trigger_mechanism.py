"""Layerwise Clean0-vs-BadNet trigger-path mechanism analysis.

The input cohort and PGD endpoints are reused from the completed pixel-space
mechanism run.  This script only performs forward passes through Clean0 and
BadNet0, collecting feature-change vectors at pixel, conv1, layer1--4, and
avgpool.  No Probe, PGD, trigger re-selection, or additional attack is run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
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

from mdluap.models import load_modelzoo_classifier
from pilot_common import seed_everything, timestamp_run_dir, write_csv, write_json


LAYERS = ("pixel", "conv1", "layer1", "layer2", "layer3", "layer4", "avgpool")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-zoo-root", required=True)
    parser.add_argument("--model-zoo-source-root", default=None)
    parser.add_argument("--source-endpoint-arrays", required=True)
    parser.add_argument("--source-run-dir", default=None)
    parser.add_argument("--output-root", default="results/stage1d_layerwise_trigger_mechanism")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--random-seed", type=int, default=20260914)
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


def model_zoo_provenance(root: Path, source_root: Path | None) -> dict[str, Any]:
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
        "infos": {alias: get_model_info(alias) for alias in ("clean0", "badnet0")},
    }


class LayerCapture:
    """Forward-hook capture for the required PreActResNet18 nodes."""

    def __init__(self, classifier: torch.nn.Module):
        backbone = getattr(classifier, "model", classifier)
        self.handles = []
        self.values: dict[str, torch.Tensor] = {}
        for name in LAYERS[1:]:
            module = getattr(backbone, name, None)
            if module is None:
                raise AttributeError(f"model does not expose required layer {name}")
            self.handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            self.values[name] = output.detach()

        return hook

    @torch.no_grad()
    def forward(self, model: torch.nn.Module, images: torch.Tensor) -> dict[str, torch.Tensor]:
        self.values = {}
        _ = model(images)
        missing = set(LAYERS[1:]) - set(self.values)
        if missing:
            raise RuntimeError(f"forward hooks did not capture: {sorted(missing)}")
        return {name: self.values[name].detach() for name in LAYERS[1:]}

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def flatten_features(values: torch.Tensor) -> np.ndarray:
    return values.detach().cpu().numpy().reshape(values.shape[0], -1).astype(np.float32)


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left.astype(np.float64)
    right = right.astype(np.float64)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    result = np.full(len(left), np.nan, dtype=np.float64)
    valid = denominator > 1e-12
    result[valid] = (left[valid] * right[valid]).sum(axis=1) / denominator[valid]
    return result


def concentration(vectors: np.ndarray) -> float | None:
    if len(vectors) < 2:
        return None
    vectors = vectors.reshape(len(vectors), -1).astype(np.float64)
    norms = np.linalg.norm(vectors, axis=1)
    valid = norms > 1e-12
    if int(valid.sum()) < 2:
        return None
    normalized = vectors[valid] / norms[valid, None]
    values = (normalized @ normalized.T)[np.triu_indices(int(valid.sum()), k=1)]
    return float(values.mean()) if len(values) else None


def validate_endpoint_arrays(arrays: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate the exact archive contract produced by the pixel experiment."""

    required = (
        "badnet_sample_indices",
        "badnet_original",
        "badnet_clean_adv",
        "badnet_backdoor_adv",
        "badnet_trigger",
    )
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError(f"source endpoint archive is missing: {missing}")

    sample_indices = np.asarray(arrays["badnet_sample_indices"], dtype=np.int64)
    if sample_indices.ndim != 1 or len(sample_indices) != 100:
        raise ValueError(f"expected exactly 100 reused cohort indices, got shape {sample_indices.shape}")
    if len(np.unique(sample_indices)) != len(sample_indices):
        raise ValueError("reused cohort contains duplicate sample indices")

    image_arrays = tuple(np.asarray(arrays[key], dtype=np.float32) for key in required[1:])
    shapes = {array.shape for array in image_arrays}
    if len(shapes) != 1:
        raise ValueError(f"source endpoint arrays have inconsistent shapes: {sorted(shapes, key=str)}")
    shape = image_arrays[0].shape
    if len(shape) != 4 or shape[0] != 100 or shape[1:] != (3, 32, 32):
        raise ValueError(f"expected four [100,3,32,32] image arrays, got {shape}")
    for name, array in zip(required[1:], image_arrays):
        if not np.isfinite(array).all():
            raise ValueError(f"source array contains non-finite values: {name}")
        if array.min() < -1e-5 or array.max() > 1.00001:
            raise ValueError(f"source array is outside [0,1]: {name} min={array.min()} max={array.max()}")
    return (sample_indices, *image_arrays)


def model_feature_residuals(
    model: torch.nn.Module,
    original: np.ndarray,
    adversarial: np.ndarray,
    triggered: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    capture = LayerCapture(model)
    adv_residuals: dict[str, list[np.ndarray]] = {name: [] for name in LAYERS}
    trigger_residuals: dict[str, list[np.ndarray]] = {name: [] for name in LAYERS}
    try:
        for start in range(0, len(original), batch_size):
            stop = min(start + batch_size, len(original))
            x = torch.from_numpy(original[start:stop]).to(device)
            adv = torch.from_numpy(adversarial[start:stop]).to(device)
            trig = torch.from_numpy(triggered[start:stop]).to(device)
            with torch.no_grad():
                captured = capture.forward(model, torch.cat((x, adv, trig), dim=0))
            count = stop - start
            for name in LAYERS[1:]:
                values = flatten_features(captured[name])
                base, adv_values, trig_values = values[:count], values[count:2 * count], values[2 * count:]
                adv_residuals[name].append(adv_values - base)
                trigger_residuals[name].append(trig_values - base)
            adv_residuals["pixel"].append((adversarial[start:stop] - original[start:stop]).reshape(count, -1))
            trigger_residuals["pixel"].append((triggered[start:stop] - original[start:stop]).reshape(count, -1))
    finally:
        capture.close()
    return (
        {name: np.concatenate(values, axis=0) for name, values in adv_residuals.items()},
        {name: np.concatenate(values, axis=0) for name, values in trigger_residuals.items()},
    )


def model_zoo_rows(provenance: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for alias, info in provenance["infos"].items():
        rows.append({
            "model_alias": alias,
            "model_type": info.get("model_type"),
            "attack": info.get("attack"),
            "classifier_seed": info.get("classifier_seed"),
            "clean_acc": info.get("clean_acc"),
            "native_asr": info.get("asr"),
        })
    return rows


def plot_curves(output: Path, metrics: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    layers = [row["layer"] for row in metrics]

    def values(field: str) -> list[float]:
        return [float(row[field]) if row[field] is not None else float("nan") for row in metrics]

    for filename, title, clean_field, bad_field, ylabel in (
        ("pgd_trigger_alignment_vs_depth.png", "PGD-trigger alignment vs depth", "clean_alignment_mean", "badnet_alignment_mean", "alignment"),
        ("trigger_concentration_vs_depth.png", "Trigger concentration vs depth", "clean_trigger_concentration", "badnet_trigger_concentration", "pairwise cosine"),
        ("pgd_concentration_vs_depth.png", "PGD residual concentration vs depth", "clean_adv_concentration", "badnet_adv_concentration", "pairwise cosine"),
    ):
        fig, axis = plt.subplots(figsize=(9, 5))
        axis.plot(layers, values(clean_field), marker="o", label="Clean0")
        axis.plot(layers, values(bad_field), marker="o", label="BadNet0")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xlabel("network depth")
        axis.tick_params(axis="x", rotation=25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(output / "figures" / filename, dpi=160)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    seed_everything(args.random_seed)
    device = torch.device(args.device)
    source_arrays = Path(args.source_endpoint_arrays).expanduser().resolve()
    if not source_arrays.is_file():
        raise FileNotFoundError(source_arrays)
    with np.load(source_arrays, allow_pickle=False) as arrays:
        sample_indices, original, clean_adv, badnet_adv, triggered = validate_endpoint_arrays(arrays)

    model_zoo_root = Path(args.model_zoo_root).expanduser().resolve()
    source_root = Path(args.model_zoo_source_root).expanduser().resolve() if args.model_zoo_source_root else None
    source_run_dir = Path(args.source_run_dir).expanduser().resolve() if args.source_run_dir else None
    if source_run_dir is not None and not source_run_dir.is_dir():
        raise FileNotFoundError(f"source run directory does not exist: {source_run_dir}")
    provenance = model_zoo_provenance(model_zoo_root, source_root)
    output = timestamp_run_dir(args.output_root, "layerwise_trigger_mechanism")
    config = vars(args).copy()
    config.update({
        "protocol": "stage1d-layerwise-clean0-badnet0-reused-pixel-cohort-v1",
        "models": ["clean0", "badnet0"],
        "layers": list(LAYERS),
        "cohort_count": len(sample_indices),
        "source_endpoint_arrays_sha256": sha256_file(source_arrays),
        "source_run_dir": str(source_run_dir) if source_run_dir else None,
        "source_endpoint_keys": [
            "badnet_sample_indices",
            "badnet_original",
            "badnet_clean_adv",
            "badnet_backdoor_adv",
            "badnet_trigger",
        ],
        "source_endpoint_shape": list(original.shape),
        "pgd_reused_without_rerun": True,
        "feature_definition": "h_l(x_adv)-h_l(x) and h_l(T(x))-h_l(x); convolution maps flattened without pooling",
        "model_zoo_provenance": provenance,
    })
    write_yaml(output / "config.resolved.yaml", config)
    write_csv(output / "cohort.csv", [{"rank": rank, "sample_index": int(index)} for rank, index in enumerate(sample_indices, start=1)])
    write_csv(output / "model_quality.csv", model_zoo_rows(provenance))
    log_path = output / "run.log"

    def log(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    log(f"Layerwise trigger mechanism analysis started: {output}")
    residuals: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for alias, adversarial in (("clean0", clean_adv), ("badnet0", badnet_adv)):
        log(f"Loading Model Zoo alias: {alias}")
        model, _info = load_modelzoo_classifier(alias, model_zoo_root=str(model_zoo_root), device=device)
        adv_values, trigger_values = model_feature_residuals(
            model, original, adversarial, triggered, batch_size=args.batch_size, device=device
        )
        residuals[alias] = {"adv": adv_values, "trigger": trigger_values}
        log(f"Captured {len(LAYERS)} residual levels for {alias}")
        del model

    metric_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for layer in LAYERS:
        clean_adv_values = residuals["clean0"]["adv"][layer]
        clean_trigger_values = residuals["clean0"]["trigger"][layer]
        bad_adv_values = residuals["badnet0"]["adv"][layer]
        bad_trigger_values = residuals["badnet0"]["trigger"][layer]
        clean_alignment = cosine_rows(clean_adv_values, clean_trigger_values)
        bad_alignment = cosine_rows(bad_adv_values, bad_trigger_values)
        clean_adv_l2 = np.linalg.norm(clean_adv_values, axis=1)
        bad_adv_l2 = np.linalg.norm(bad_adv_values, axis=1)
        clean_trigger_l2 = np.linalg.norm(clean_trigger_values, axis=1)
        bad_trigger_l2 = np.linalg.norm(bad_trigger_values, axis=1)
        row = {
            "layer": layer,
            "sample_count": len(sample_indices),
            "clean_alignment_mean": float(np.nanmean(clean_alignment)),
            "clean_alignment_median": float(np.nanmedian(clean_alignment)),
            "badnet_alignment_mean": float(np.nanmean(bad_alignment)),
            "badnet_alignment_median": float(np.nanmedian(bad_alignment)),
            "alignment_gap_badnet_minus_clean": float(np.nanmean(bad_alignment) - np.nanmean(clean_alignment)),
            "clean_adv_concentration": concentration(clean_adv_values),
            "badnet_adv_concentration": concentration(bad_adv_values),
            "adv_concentration_gap_badnet_minus_clean": (concentration(bad_adv_values) - concentration(clean_adv_values)) if concentration(clean_adv_values) is not None and concentration(bad_adv_values) is not None else None,
            "clean_trigger_concentration": concentration(clean_trigger_values),
            "badnet_trigger_concentration": concentration(bad_trigger_values),
            "trigger_concentration_gap_badnet_minus_clean": (concentration(bad_trigger_values) - concentration(clean_trigger_values)) if concentration(clean_trigger_values) is not None and concentration(bad_trigger_values) is not None else None,
            "clean_adv_l2_mean": float(clean_adv_l2.mean()),
            "badnet_adv_l2_mean": float(bad_adv_l2.mean()),
            "clean_trigger_l2_mean": float(clean_trigger_l2.mean()),
            "badnet_trigger_l2_mean": float(bad_trigger_l2.mean()),
            "clean_adv_zero_norm_count": int((clean_adv_l2 <= 1e-12).sum()),
            "badnet_adv_zero_norm_count": int((bad_adv_l2 <= 1e-12).sum()),
            "clean_trigger_zero_norm_count": int((clean_trigger_l2 <= 1e-12).sum()),
            "badnet_trigger_zero_norm_count": int((bad_trigger_l2 <= 1e-12).sum()),
            "clean_alignment_valid_count": int(np.isfinite(clean_alignment).sum()),
            "badnet_alignment_valid_count": int(np.isfinite(bad_alignment).sum()),
        }
        metric_rows.append(row)
        for position, sample_index in enumerate(sample_indices):
            sample_rows.extend((
                {
                    "model_alias": "clean0",
                    "layer": layer,
                    "sample_index": int(sample_index),
                    "adv_alignment": float(clean_alignment[position]),
                    "adv_delta_l2": float(clean_adv_l2[position]),
                    "trigger_delta_l2": float(clean_trigger_l2[position]),
                    "adv_zero_norm": bool(clean_adv_l2[position] <= 1e-12),
                    "trigger_zero_norm": bool(clean_trigger_l2[position] <= 1e-12),
                },
                {
                    "model_alias": "badnet0",
                    "layer": layer,
                    "sample_index": int(sample_index),
                    "adv_alignment": float(bad_alignment[position]),
                    "adv_delta_l2": float(bad_adv_l2[position]),
                    "trigger_delta_l2": float(bad_trigger_l2[position]),
                    "adv_zero_norm": bool(bad_adv_l2[position] <= 1e-12),
                    "trigger_zero_norm": bool(bad_trigger_l2[position] <= 1e-12),
                },
            ))

    write_csv(output / "layer_metrics.csv", metric_rows)
    write_csv(output / "per_sample_layer_records.csv", sample_rows)
    plot_curves(output, metric_rows)
    summary = {
        "protocol": config["protocol"],
        "output_directory": str(output),
        "source_endpoint_arrays": str(source_arrays),
        "source_endpoint_arrays_sha256": sha256_file(source_arrays),
        "source_run_dir": str(source_run_dir) if source_run_dir else None,
        "cohort_count": len(sample_indices),
        "sample_indices": [int(index) for index in sample_indices],
        "models": ["clean0", "badnet0"],
        "layers": list(LAYERS),
        "metrics": metric_rows,
        "model_zoo_provenance": provenance,
        "interpretation_boundary": "reports representation-direction alignment and concentration; does not establish a literal causal path",
    }
    write_json(output / "summary.json", summary)
    log(f"Layerwise trigger mechanism analysis complete: {output}")


if __name__ == "__main__":
    main()
