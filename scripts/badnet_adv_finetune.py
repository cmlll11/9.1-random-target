"""Causal BadNet0 fine-tuning experiment on Probe-selected robust samples."""

from __future__ import annotations

import argparse
import csv
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
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mdluap.data import cifar10_dataset
from mdluap.models import CIFAR10_MEAN, CIFAR10_STD, NormalizedClassifier
from mdluap.official_triggers import build_trigger_adapters
from mdluap.targeted_pgd import targeted_pgd_endpoint
from pilot_common import batch_images, seed_everything, timestamp_run_dir, write_csv, write_json


TARGET = 0
TRAIN_COHORT_SIZE = 200
MINIMUM_COHORT_SIZE = TRAIN_COHORT_SIZE
FIXED_EPSILON_PIXELS = 1.0
VALIDITY_MAX_CLEAN_ACCURACY_DROP = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-zoo-root", required=True)
    parser.add_argument("--model-zoo-source-root", default=None)
    parser.add_argument("--selection-file", required=True)
    parser.add_argument("--backdoorbench-root", required=True)
    parser.add_argument("--badnet-trigger-path", required=True)
    parser.add_argument("--output-root", default="results/stage1d_badnet_adv_finetune")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--anchor-weight", type=float, default=0.1)
    parser.add_argument("--fine-tune-seed", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=20260916)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(path: Path | None) -> str | None:
    if path is None or not (path / ".git").exists():
        return None
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    import yaml

    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def load_selection(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"rank", "sample_index", "true_label"}
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise ValueError(f"selection file is missing columns: {sorted(missing)}")
    rows.sort(key=lambda row: int(row["rank"]))
    if len(rows) < 400:
        raise ValueError(f"selection file contains only {len(rows)} rows; at least 400 are required")
    indices = [int(row["sample_index"]) for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError("selection file contains duplicate sample indices")
    return rows


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
        "registry_sha256": sha256_file(registry) if registry else None,
        "list_models": list_models(),
        "badnet0_info": get_model_info("badnet0"),
    }


def load_trainable_badnet(model_zoo_root: Path, device: torch.device):
    """Load BadNet0 through Model Zoo and add only the raw-image normalizer."""

    os.environ["MODEL_ZOO_ROOT"] = str(model_zoo_root)
    from modelzoo import load_model

    base = load_model("badnet0", device=device, root=str(model_zoo_root))
    model = NormalizedClassifier(base).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    backbone = model.model
    layer4 = getattr(backbone, "layer4", None)
    if layer4 is None:
        raise AttributeError("badnet0 must expose layer4 for layer4+head fine-tuning")
    head_name = None
    head = None
    for candidate in ("linear", "fc", "classifier"):
        value = getattr(backbone, candidate, None)
        if isinstance(value, nn.Module) and any(True for _ in value.parameters()):
            head_name, head = candidate, value
            break
    if head is None:
        linear_modules = [(name, module) for name, module in backbone.named_modules() if isinstance(module, nn.Linear)]
        if not linear_modules:
            raise AttributeError("badnet0 has no classifier head")
        head_name, head = linear_modules[-1]

    for parameter in layer4.parameters():
        parameter.requires_grad_(True)
    for parameter in head.parameters():
        parameter.requires_grad_(True)

    # BatchNorm affine parameters and running statistics are not part of the
    # intended intervention.  Keeping them frozen prevents a tiny cohort from
    # changing the representation globally through running-stat updates.
    for module in layer4.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable_names:
        raise RuntimeError("no trainable parameters selected")
    return model, {"trainable_scope": ["model.layer4", f"model.{head_name}"], "trainable_parameters": trainable_names}


def set_fine_tune_mode(model: nn.Module) -> None:
    """Train only layer4/head while keeping every BatchNorm in eval mode."""

    model.eval()
    model.model.layer4.train()
    for candidate in ("linear", "fc", "classifier"):
        value = getattr(model.model, candidate, None)
        if isinstance(value, nn.Module):
            value.train()
    for module in model.model.layer4.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def load_raw_batch(dataset, indices: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.stack([dataset[index][0] for index in indices]).to(device)
    labels = torch.tensor([int(dataset[index][1]) for index in indices], dtype=torch.long, device=device)
    return images, labels


def attack_candidates(model, dataset, rows: list[dict[str, Any]], *, epsilon_pixels: float, args: argparse.Namespace, device: torch.device, seed: int) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    seed_everything(seed)
    candidate_rows = [row for row in rows if int(row["true_label"]) != TARGET]
    indices = [int(row["sample_index"]) for row in candidate_rows]
    output_rows: list[dict[str, Any]] = []
    attacks: dict[int, dict[str, Any]] = {}
    epsilon = epsilon_pixels / 255.0
    for batch_indices, images, labels in batch_images(dataset, indices, batch_size=args.batch_size, device=device):
        with torch.no_grad():
            original_prediction = model(images).argmax(dim=1)
        eligible_mask = original_prediction.ne(TARGET)
        eligible_indices = [index for index, eligible in zip(batch_indices, eligible_mask.cpu().tolist()) if eligible]
        if eligible_indices:
            positions = [batch_indices.index(index) for index in eligible_indices]
            eligible_images = images[positions]
            result = targeted_pgd_endpoint(
                model,
                eligible_images,
                target=TARGET,
                epsilon=epsilon,
                steps=args.steps,
                alpha=epsilon / 10.0,
                random_start=True,
                restarts=args.restarts,
            )
            for position, index in enumerate(eligible_indices):
                endpoint = result.endpoint[position].detach().cpu().numpy().astype(np.float32)
                original = eligible_images[position].detach().cpu().numpy().astype(np.float32)
                record = {
                    "endpoint": endpoint,
                    "success": bool(result.success[position].item()),
                    "original_prediction": int(original_prediction[positions[position]].item()),
                    "endpoint_prediction": int(result.endpoint_prediction[position].item()),
                    "actual_linf": float(result.endpoint_linf[position].item()),
                    "actual_linf_pixels": float(result.endpoint_linf[position].item() * 255.0),
                    "actual_l2": float(torch.linalg.vector_norm(result.endpoint[position] - eligible_images[position]).item()),
                    "targeted_loss": float(result.target_loss[position].item()),
                }
                attacks[index] = record
        for index, label, prediction in zip(batch_indices, labels.cpu().tolist(), original_prediction.cpu().tolist()):
            attack = attacks.get(int(index))
            output_rows.append({
                "epsilon_pixels": epsilon_pixels,
                "sample_index": int(index),
                "true_label": int(label),
                "original_prediction": int(prediction),
                "eligible": int(prediction) != TARGET,
                "pgd_status": "ineligible_true_label_0" if int(label) == TARGET else ("ineligible_original_target" if int(prediction) == TARGET else ("success" if attack and attack["success"] else "failure")),
                "success": None if int(label) == TARGET or int(prediction) == TARGET else bool(attack and attack["success"]),
                "endpoint_prediction": None if attack is None else attack["endpoint_prediction"],
                "actual_linf": None if attack is None else attack["actual_linf"],
                "actual_linf_pixels": None if attack is None else attack["actual_linf_pixels"],
                "actual_l2": None if attack is None else attack["actual_l2"],
                "targeted_loss": None if attack is None else attack["targeted_loss"],
            })
    return output_rows, attacks


def select_successful_cohort(rows: list[dict[str, Any]], attacks: dict[int, dict[str, Any]], *, count: int | None = None) -> list[dict[str, Any]]:
    """Select successful non-target samples in the original Probe-rank order."""

    qualified: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: int(item["rank"])):
        index = int(row["sample_index"])
        attack = attacks.get(index)
        if int(row["true_label"]) == TARGET or not attack or not attack["success"]:
            continue
        qualified.append({**row, **attack})
        if count is not None and len(qualified) == count:
            break
    return qualified


def train_model(model: nn.Module, original: torch.Tensor, adversarial: torch.Tensor, labels: torch.Tensor, anchor_logits: torch.Tensor, *, arm: str, args: argparse.Namespace, seed: int) -> list[dict[str, Any]]:
    seed_everything(seed)
    set_fine_tune_mode(model)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    dataset = TensorDataset(original.detach().cpu(), adversarial.detach().cpu(), labels.detach().cpu(), anchor_logits.detach().cpu())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed), drop_last=False)
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        set_fine_tune_mode(model)
        total_loss = 0.0
        total_count = 0
        for clean_batch, adv_batch, label_batch, anchor_batch in loader:
            clean_batch = clean_batch.to(next(model.parameters()).device)
            adv_batch = adv_batch.to(clean_batch.device)
            label_batch = label_batch.to(clean_batch.device)
            anchor_batch = anchor_batch.to(clean_batch.device)
            optimizer.zero_grad(set_to_none=True)
            clean_logits = model(clean_batch)
            clean_loss = F.cross_entropy(clean_logits, label_batch)
            anchor_loss = F.mse_loss(clean_logits, anchor_batch)
            if arm == "adv_ft":
                adv_loss = F.cross_entropy(model(adv_batch), label_batch)
                loss = 0.5 * (clean_loss + adv_loss)
            else:
                adv_loss = torch.zeros((), device=clean_batch.device)
                loss = clean_loss
            loss = loss + args.anchor_weight * anchor_loss
            loss.backward()
            optimizer.step()
            count = int(label_batch.numel())
            total_loss += float(loss.item()) * count
            total_count += count
        history.append({"arm": arm, "epoch": epoch, "loss": total_loss / max(total_count, 1)})
    return history


@torch.no_grad()
def evaluate_clean_accuracy(model, dataset, indices: list[int], *, batch_size: int, device: torch.device) -> float:
    correct = 0
    total = 0
    for _batch_indices, images, labels in batch_images(dataset, indices, batch_size=batch_size, device=device):
        correct += int(model(images).argmax(dim=1).eq(labels).sum().item())
        total += len(labels)
    return correct / total if total else float("nan")


@torch.no_grad()
def evaluate_trigger(model, adapter, dataset, indices: list[int], *, batch_size: int, device: torch.device) -> tuple[int, int, float]:
    success = 0
    eligible = 0
    for batch_indices, images, labels in batch_images(dataset, indices, batch_size=batch_size, device=device):
        mask = labels.ne(TARGET)
        if not bool(mask.any()):
            continue
        triggered = adapter.apply(images, sample_indices=batch_indices, split="cifar10_test")
        predictions = model(triggered).argmax(dim=1)
        success += int(predictions[mask].eq(TARGET).sum().item())
        eligible += int(mask.sum().item())
    return success, eligible, success / eligible if eligible else float("nan")


def evaluate_fixed_endpoints(model, dataset, selected_rows: list[dict[str, Any]], *, device: torch.device, baseline_eligible: set[int], batch_size: int) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Evaluate the exact pre-fine-tuning endpoints; do not rerun PGD."""
    rows: list[dict[str, Any]] = []
    eligible_fixed = 0
    success_fixed = 0
    standard_eligible = 0
    standard_success = 0
    indices = [int(row["sample_index"]) for row in selected_rows]
    row_by_index = {int(row["sample_index"]): row for row in selected_rows}
    for batch_indices, images, labels in batch_images(dataset, indices, batch_size=batch_size, device=device):
        endpoints = torch.from_numpy(np.stack([row_by_index[int(index)]["endpoint"] for index in batch_indices])).to(device)
        with torch.no_grad():
            original_prediction = model(images).argmax(dim=1)
            endpoint_logits = model(endpoints)
            endpoint_prediction = endpoint_logits.argmax(dim=1)
            endpoint_target_loss = F.cross_entropy(endpoint_logits, torch.full_like(labels, TARGET), reduction="none")
        for position, index in enumerate(batch_indices):
            prediction = int(original_prediction[position].item())
            success = bool(endpoint_prediction[position].eq(TARGET).item())
            baseline_ok = int(index) in baseline_eligible
            if baseline_ok:
                eligible_fixed += 1
                success_fixed += int(success)
            if prediction != TARGET:
                standard_eligible += 1
                standard_success += int(success)
            rows.append({
                "model_arm": None,
                "sample_index": int(index),
                "true_label": int(labels[position].item()),
                "original_prediction": prediction,
                "baseline_eligible": baseline_ok,
                "eligible": prediction != TARGET,
                "success": success,
                "endpoint_prediction": int(endpoint_prediction[position].item()),
                "actual_linf": float((endpoints[position] - images[position]).abs().amax().item()),
                "actual_linf_pixels": float((endpoints[position] - images[position]).abs().amax().item() * 255.0),
                "actual_l2": float(torch.linalg.vector_norm(endpoints[position] - images[position]).item()),
                "targeted_loss": float(endpoint_target_loss[position].item()),
                "evaluation_type": "fixed_pre_finetuning_endpoint",
                "endpoint_reused": True,
            })
    return rows, {
        "fixed_baseline_eligible": eligible_fixed,
        "fixed_baseline_success": success_fixed,
        "fixed_baseline_asr": success_fixed / eligible_fixed if eligible_fixed else float("nan"),
        "fixed_endpoint_per_model_eligible": standard_eligible,
        "fixed_endpoint_per_model_success": standard_success,
        "fixed_endpoint_per_model_asr": standard_success / standard_eligible if standard_eligible else float("nan"),
    }


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.steps <= 0 or args.restarts <= 0:
        raise ValueError("epochs, steps, and restarts must be positive")
    seed_everything(args.random_seed)
    device = torch.device(args.device)
    data_root = Path(args.data_root).expanduser().resolve()
    model_zoo_root = Path(args.model_zoo_root).expanduser().resolve()
    source_root = Path(args.model_zoo_source_root).expanduser().resolve() if args.model_zoo_source_root else None
    selection_path = Path(args.selection_file).expanduser().resolve()
    trigger_path = Path(args.badnet_trigger_path).expanduser().resolve()
    if not selection_path.is_file():
        raise FileNotFoundError(selection_path)
    if not trigger_path.is_file():
        raise FileNotFoundError(trigger_path)
    source_rows = load_selection(selection_path)
    dataset = cifar10_dataset(data_root, train=False)
    if max(int(row["sample_index"]) for row in source_rows) >= len(dataset):
        raise ValueError("selection contains an index outside CIFAR-10 test split")
    output = timestamp_run_dir(args.output_root, "badnet_adv_finetune")
    log_path = output / "run.log"

    def log(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    provenance = model_zoo_provenance(model_zoo_root, source_root)
    config = vars(args).copy()
    config.update({
        "protocol": "stage1d-badnet0-robust-sample-adv-finetune-v2-fixed-eps1-train200",
        "target": TARGET,
        "selection_file": str(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "cohort_policy": {"minimum_successful": MINIMUM_COHORT_SIZE, "fine_tune": TRAIN_COHORT_SIZE, "heldout": "all remaining successful samples", "order": "Probe rank after filtering true_label != 0 and BadNet PGD success"},
        "epsilon_pixels": FIXED_EPSILON_PIXELS,
        "pgd": {"steps": args.steps, "restarts": args.restarts, "random_start": True, "alpha": "epsilon/10"},
        "fine_tuning": {"arms": ["baseline", "clean_ft", "adv_ft"], "scope": "layer4 + classifier head, BatchNorm frozen", "optimizer": "AdamW", "losses": {"clean_ft": "CE(clean,y)+anchor_weight*MSE(clean_logits,baseline_logits)", "adv_ft": "0.5*CE(clean,y)+0.5*CE(adv,y)+anchor_weight*MSE(clean_logits,baseline_logits)"}, "epochs": args.epochs, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "anchor_weight": args.anchor_weight, "seed": args.fine_tune_seed},
        "evaluation": {"primary": "fixed pre-fine-tuning endpoints", "rerun_pgd": False},
        "validity": {"max_clean_accuracy_drop": VALIDITY_MAX_CLEAN_ACCURACY_DROP, "requires_clean_accuracy_drop_within_threshold": True},
        "model_zoo_provenance": provenance,
        "badnet_trigger": {"source": str(trigger_path), "sha256": sha256_file(trigger_path)},
    })
    write_yaml(output / "config.resolved.yaml", config)
    write_csv(output / "source_selection.csv", source_rows)
    log(f"BadNet0 adversarial fine-tuning experiment started: {output}")

    probe_model, scope = load_trainable_badnet(model_zoo_root, device)
    log("Generating pre-fine-tuning BadNet target-0 PGD endpoints")
    attempts: list[dict[str, Any]] = []
    final_epsilon = None
    final_attacks: dict[int, dict[str, Any]] = {}
    final_attack_rows: list[dict[str, Any]] = []
    attack_rows, attacks = attack_candidates(probe_model, dataset, source_rows, epsilon_pixels=FIXED_EPSILON_PIXELS, args=args, device=device, seed=args.random_seed)
    attempts.extend(attack_rows)
    qualified = select_successful_cohort(source_rows, attacks)
    log(f"epsilon={FIXED_EPSILON_PIXELS}/255 qualified={len(qualified)}/{MINIMUM_COHORT_SIZE}")
    if len(qualified) >= MINIMUM_COHORT_SIZE:
        final_epsilon = FIXED_EPSILON_PIXELS
        final_attacks = attacks
        final_attack_rows = attack_rows
    del probe_model
    if final_epsilon is None:
        write_csv(output / "pgd_generation_records.csv", attempts)
        write_json(output / "summary.json", {"status": "insufficient_cohort", "minimum_fine_tune_size": MINIMUM_COHORT_SIZE, "epsilon_pixels": FIXED_EPSILON_PIXELS, "attempts": attempts, "model_zoo_provenance": provenance})
        log("ERROR: insufficient successful PGD cohort; no fine-tuning was run")
        raise SystemExit(2)

    selected_rows = [{**row, "epsilon_pixels": final_epsilon} for row in select_successful_cohort(source_rows, final_attacks)]
    train_rows = selected_rows[:TRAIN_COHORT_SIZE]
    heldout_rows = selected_rows[TRAIN_COHORT_SIZE:]
    if len(train_rows) != TRAIN_COHORT_SIZE or not heldout_rows:
        raise RuntimeError("selected cohort must contain 200 fine-tuning samples and at least one held-out sample")
    train_indices = [int(row["sample_index"]) for row in train_rows]
    heldout_indices = [int(row["sample_index"]) for row in heldout_rows]
    if set(train_indices) & set(heldout_indices):
        raise RuntimeError("fine-tuning and held-out cohorts overlap")
    write_csv(output / "pgd_generation_records.csv", attempts)
    cohort_rows = []
    for split, rows in (("fine_tune", train_rows), ("heldout", heldout_rows)):
        for rank, row in enumerate(rows, start=1):
            cohort_rows.append({"cohort": split, "cohort_rank": rank, "probe_rank": row["rank"], "sample_index": row["sample_index"], "true_label": row["true_label"], "original_prediction": row["original_prediction"], "epsilon_pixels": final_epsilon, "pgd_success": row["success"], "actual_linf_pixels": row["actual_linf_pixels"], "actual_l2": row["actual_l2"]})
    write_csv(output / "cohort_records.csv", cohort_rows)

    all_indices = [int(row["sample_index"]) for row in selected_rows]
    original, labels = load_raw_batch(dataset, all_indices, device)
    adversarial = torch.from_numpy(np.stack([row["endpoint"] for row in selected_rows])).to(device)
    np.savez_compressed(output / "endpoint_arrays.npz", sample_indices=np.asarray(all_indices, dtype=np.int64), true_labels=labels.cpu().numpy(), original=original.cpu().numpy(), badnet_pgd_endpoint=adversarial.cpu().numpy(), train_positions=np.arange(TRAIN_COHORT_SIZE), heldout_positions=np.arange(TRAIN_COHORT_SIZE, len(selected_rows)), epsilon_pixels=np.asarray([final_epsilon], dtype=np.float32))

    adapter = build_trigger_adapters(model_root=model_zoo_root, backdoorbench_root=Path(args.backdoorbench_root).expanduser().resolve(), explicit={"badnet": trigger_path}, device=device, blended_alpha=0.2, wanet_s=0.5, wanet_grid_rescale=1.0)["badnet"]
    if not adapter.status.available:
        raise RuntimeError(adapter.status.reason or "BadNet trigger unavailable")
    baseline_eligible = {int(row["sample_index"]) for row in heldout_rows if int(row["original_prediction"]) != TARGET}
    eval_indices = list(range(len(dataset)))
    eval_indices_without_ft = [index for index in eval_indices if index not in set(train_indices)]
    arm_metrics: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_original, train_labels = original[:TRAIN_COHORT_SIZE], labels[:TRAIN_COHORT_SIZE]
    train_adversarial = adversarial[:TRAIN_COHORT_SIZE]
    anchor_model, _ = load_trainable_badnet(model_zoo_root, device)
    anchor_model.eval()
    with torch.no_grad():
        anchor_logits = anchor_model(train_original).detach()
    del anchor_model
    for arm in ("baseline", "clean_ft", "adv_ft"):
        log(f"Evaluating {arm}")
        model, scope = load_trainable_badnet(model_zoo_root, device)
        if arm in {"clean_ft", "adv_ft"}:
            history_rows.extend(train_model(model, train_original, train_adversarial, train_labels, anchor_logits, arm=arm, args=args, seed=args.fine_tune_seed))
            state_dict = {name: value.detach().cpu() for name, value in model.state_dict().items()}
            torch.save({"state_dict": state_dict, "base_alias": "badnet0", "fine_tune_arm": arm, "trainable_scope": scope}, checkpoint_dir / f"{arm}_state_dict.pt")
        model.eval()
        clean_accuracy_full = evaluate_clean_accuracy(model, dataset, eval_indices, batch_size=args.eval_batch_size, device=device)
        clean_accuracy_heldout = evaluate_clean_accuracy(model, dataset, eval_indices_without_ft, batch_size=args.eval_batch_size, device=device)
        train_accuracy = evaluate_clean_accuracy(model, dataset, train_indices, batch_size=args.eval_batch_size, device=device)
        trigger_success_full, trigger_eligible_full, trigger_asr_full = evaluate_trigger(model, adapter, dataset, eval_indices, batch_size=args.eval_batch_size, device=device)
        trigger_success_heldout, trigger_eligible_heldout, trigger_asr_heldout = evaluate_trigger(model, adapter, dataset, eval_indices_without_ft, batch_size=args.eval_batch_size, device=device)
        pgd_rows, pgd_metrics = evaluate_fixed_endpoints(
            model,
            dataset,
            heldout_rows,
            device=device,
            baseline_eligible=baseline_eligible,
            batch_size=args.eval_batch_size,
        )
        for row in pgd_rows:
            row["model_arm"] = arm
            evaluation_rows.append(row)
        arm_metrics.append({"model_arm": arm, "clean_accuracy_full": clean_accuracy_full, "clean_accuracy_heldout": clean_accuracy_heldout, "fine_tune_cohort_accuracy": train_accuracy, "native_trigger_success_full": trigger_success_full, "native_trigger_eligible_full": trigger_eligible_full, "native_trigger_asr_full": trigger_asr_full, "native_trigger_success_heldout": trigger_success_heldout, "native_trigger_eligible_heldout": trigger_eligible_heldout, "native_trigger_asr_heldout": trigger_asr_heldout, **pgd_metrics})
        del model

    write_csv(output / "fine_tune_history.csv", history_rows)
    write_csv(output / "evaluation_records.csv", evaluation_rows)
    baseline_metrics = next(row for row in arm_metrics if row["model_arm"] == "baseline")
    for row in arm_metrics:
        drop = baseline_metrics["clean_accuracy_heldout"] - row["clean_accuracy_heldout"]
        row["clean_accuracy_heldout_drop_vs_baseline"] = drop
        row["validity_clean_accuracy_within_2pp"] = drop <= VALIDITY_MAX_CLEAN_ACCURACY_DROP
    write_csv(output / "group_metrics.csv", arm_metrics)
    write_csv(output / "model_quality.csv", [{"alias": "badnet0", "clean_acc": provenance["badnet0_info"].get("clean_acc"), "native_asr": provenance["badnet0_info"].get("asr"), "model_id": provenance["badnet0_info"].get("model_id")}])
    validity_rows = []
    for row in arm_metrics:
        drop = baseline_metrics["clean_accuracy_heldout"] - row["clean_accuracy_heldout"]
        validity_rows.append({"model_arm": row["model_arm"], "clean_accuracy_heldout_drop": drop, "within_threshold": drop <= VALIDITY_MAX_CLEAN_ACCURACY_DROP})
    summary = {"status": "complete", "protocol": config["protocol"], "output_directory": str(output), "final_epsilon_pixels": final_epsilon, "selected_count": len(selected_rows), "fine_tune_count": len(train_rows), "heldout_count": len(heldout_rows), "attempts": [{"epsilon_pixels": FIXED_EPSILON_PIXELS, "qualified_count": len(selected_rows)}], "trainable_scope": scope, "model_zoo_provenance": provenance, "metrics": arm_metrics, "validity": validity_rows, "interpretation_boundary": "single-seed exploratory functional intervention; results are invalid for mechanism claims if clean accuracy drops more than 2 percentage points"}
    write_json(output / "summary.json", summary)
    log(f"BadNet0 adversarial fine-tuning experiment complete: {output}")


if __name__ == "__main__":
    main()
