"""Compute the Stage 1D-WT quality gate for official checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mdluap.data import cifar10_dataset
from mdluap.models import load_attack_result_model, load_backdoor_toolbox_resnet18


def accuracy(model, dataset, device, *, target: int | None = None) -> float:
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
    correct = total = 0
    with torch.inference_mode():
        for batch in loader:
            images = batch[0].to(device)
            predictions = model(images).argmax(dim=1)
            if target is None:
                labels = batch[1].to(device)
                correct += int(predictions.eq(labels).sum())
            else:
                correct += int(predictions.eq(int(target)).sum())
            total += int(predictions.numel())
    return correct / max(total, 1)


def load_bd_test(path: Path, root: Path):
    """Load the official poisoned test dataset saved by BackdoorBench."""

    old = os.getcwd()
    os.chdir(str(root.resolve()))
    if str(root.resolve()) not in sys.path:
        sys.path.insert(0, str(root.resolve()))
    from utils.save_load_attack import load_attack_result
    try:
        payload = load_attack_result(str(path))
        return payload["bd_test"]
    finally:
        os.chdir(old)


def result_path(root: Path, group: str, seed: int) -> Path:
    aliases = {"inputaware": ("inputaware", "input_aware", "input-aware")}
    for name in aliases.get(group, (group,)):
        path = root / name / f"seed{seed}" / "attack_result.pt"
        if path.is_file():
            return path
    return root / group / f"seed{seed}" / "attack_result.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--backdoorbench-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--adaptive-blend-root", default=None)
    parser.add_argument("--adaptive-quality-json", default=None)
    parser.add_argument("--clean-group", default="clean_select_shared")
    parser.add_argument("--clean-seeds", default="0,1,2,3")
    parser.add_argument("--backdoor-groups", default="badnet,blended,wanet,ssba,inputaware,adaptive_blend")
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_root, model_root, bdb_root = Path(args.data_root), Path(args.model_root), Path(args.backdoorbench_root)
    test_data = cifar10_dataset(data_root, train=False)
    rows = []
    for seed_text in args.clean_seeds.split(","):
        seed = int(seed_text)
        path = result_path(model_root, args.clean_group, seed)
        model, metadata = load_attack_result_model(str(path), backdoorbench_root=str(bdb_root), device=device)
        acc = accuracy(model, test_data, device)
        rows.append({"group": args.clean_group, "seed": seed, "model_path": str(path.resolve()), "model_name": metadata["model_name"], "clean_accuracy": acc, "status": "qualified" if acc >= 0.90 else "gate_failed"})

    adaptive_quality = {}
    if args.adaptive_quality_json:
        adaptive_quality = json.loads(Path(args.adaptive_quality_json).read_text(encoding="utf-8"))
    for group in [item.strip() for item in args.backdoor_groups.split(",") if item.strip()]:
        seed = 0
        if group == "adaptive_blend":
            values = adaptive_quality.get(group, adaptive_quality.get("adaptive_blend_seed0", {}))
            if values:
                acc = float(values.get("clean_accuracy", values.get("backdoor_clean_accuracy", 0)))
                bd_asr = float(values.get("backdoor_asr", 0))
                clean_trigger = float(values.get("clean_trigger_asr", 1))
                passed = acc >= 0.90 and bd_asr >= 0.90 and clean_trigger <= 0.10
                rows.append({"group": group, "seed": seed, **values, "status": "qualified" if passed else "gate_failed"})
            else:
                rows.append({"group": group, "seed": seed, "status": "gate_failed", "reason": "Adaptive-Blend quality metrics were not supplied"})
            continue
        path = result_path(model_root, group, seed)
        model, metadata = load_attack_result_model(str(path), backdoorbench_root=str(bdb_root), device=device)
        clean_acc = accuracy(model, test_data, device)
        bd_test = load_bd_test(path, bdb_root)
        bd_asr = accuracy(model, bd_test, device, target=0)
        clean_model, _ = load_attack_result_model(str(result_path(model_root, args.clean_group, 0)), backdoorbench_root=str(bdb_root), device=device)
        clean_trigger_asr = accuracy(clean_model, bd_test, device, target=0)
        passed = clean_acc >= 0.90 and bd_asr >= 0.90 and clean_trigger_asr <= 0.10
        rows.append({"group": group, "seed": seed, "model_path": str(path.resolve()), "model_name": metadata["model_name"], "clean_accuracy": clean_acc, "backdoor_clean_accuracy": clean_acc, "backdoor_asr": bd_asr, "clean_trigger_asr": clean_trigger_asr, "status": "qualified" if passed else "gate_failed"})
    payload = {"protocol": "stage1d-wt-official-quality-gate-v1", "thresholds": {"clean_accuracy_min": 0.90, "backdoor_clean_accuracy_min": 0.90, "native_asr_min": 0.90, "clean_trigger_asr_max": 0.10}, "rows": rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "rows": rows}, indent=2))
    if any(row.get("status") != "qualified" for row in rows if row["group"] != args.clean_group):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
