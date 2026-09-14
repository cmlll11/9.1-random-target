"""Preflight Model Zoo and official-trigger assets for Stage 1D layerwise runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mdluap.models import load_modelzoo_classifier
from mdluap.official_triggers import build_trigger_adapters
from layerwise_trigger_mechanism import model_zoo_provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-zoo-root", required=True)
    parser.add_argument("--model-zoo-source-root", default=None)
    parser.add_argument("--backdoorbench-root", required=True)
    parser.add_argument("--trigger-artifact-root", required=True)
    parser.add_argument("--backdoor-alias", required=True, choices=("wanet0", "ssba0"))
    parser.add_argument("--wanet-state-path", default=None)
    parser.add_argument("--ssba-encoder-path", default=None)
    parser.add_argument("--ssba-config-path", default=None)
    parser.add_argument("--min-clean-acc", type=float, default=0.80)
    parser.add_argument("--min-asr", type=float, default=0.80)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.model_zoo_root).expanduser().resolve()
    source_root = Path(args.model_zoo_source_root).expanduser().resolve() if args.model_zoo_source_root else None
    bdb_root = Path(args.backdoorbench_root).expanduser().resolve()
    artifact_root = Path(args.trigger_artifact_root).expanduser().resolve()
    device = torch.device(args.device)
    aliases = ("clean0", "clean1", "clean2", "clean3", args.backdoor_alias)
    provenance = model_zoo_provenance(root, source_root, aliases)
    info = provenance["infos"][args.backdoor_alias]
    checks: dict[str, Any] = {}
    failures: list[str] = []

    clean_acc = info.get("clean_acc")
    native_asr = info.get("asr")
    checks["quality_metadata"] = {
        "clean_acc": clean_acc,
        "native_asr": native_asr,
        "min_clean_acc": args.min_clean_acc,
        "min_native_asr": args.min_asr,
        "passed": clean_acc is not None and native_asr is not None and float(clean_acc) >= args.min_clean_acc and float(native_asr) >= args.min_asr,
    }
    if not checks["quality_metadata"]["passed"]:
        failures.append("Model Zoo quality metadata is below the requested ACC/ASR gate")

    models: dict[str, Any] = {}
    for alias in ("clean0", args.backdoor_alias):
        model, _ = load_modelzoo_classifier(alias, model_zoo_root=str(root), device=device)
        with torch.no_grad():
            logits = model(torch.zeros(2, 3, 32, 32, device=device))
        passed = tuple(logits.shape) == (2, 10) and bool(torch.isfinite(logits).all())
        models[alias] = {"output_shape": list(logits.shape), "finite_logits": bool(torch.isfinite(logits).all()), "passed": passed}
        if not passed:
            failures.append(f"{alias} failed Model Zoo load/output check")
        del model
    checks["model_load_and_output"] = models

    trigger_type = {"wanet0": "wanet", "ssba0": "ssba"}[args.backdoor_alias]
    explicit = {
        "badnet": None,
        "blended": None,
        "wanet_identity": None,
        "wanet_noise": None,
        "ssba_encoder": Path(args.ssba_encoder_path).expanduser().resolve() if args.ssba_encoder_path else None,
        "ssba_config": Path(args.ssba_config_path).expanduser().resolve() if args.ssba_config_path else None,
        "inputaware": None,
        "adaptive_blend": None,
    }
    if args.wanet_state_path:
        state = Path(args.wanet_state_path).expanduser().resolve()
        if state.name == "state_identity_grid.pt":
            explicit["wanet_identity"] = state
        elif state.name == "state_noise_grid.pt":
            explicit["wanet_noise"] = state
    adapter_map = build_trigger_adapters(model_root=artifact_root, backdoorbench_root=bdb_root, explicit=explicit, device=device, blended_alpha=0.2, wanet_s=0.5, wanet_grid_rescale=1.0)
    adapter = adapter_map[trigger_type]
    trigger_check = {"source": adapter.status.source, "available": adapter.status.available, "reason": adapter.status.reason}
    if adapter.status.available:
        try:
            output = adapter.apply(torch.zeros(2, 3, 32, 32, device=device), sample_indices=[0, 1], split="cifar10_test")
            trigger_check.update({"output_shape": list(output.shape), "range": [float(output.min()), float(output.max())], "passed": tuple(output.shape) == (2, 3, 32, 32) and bool(torch.isfinite(output).all())})
        except Exception as exc:
            trigger_check.update({"passed": False, "reason": str(exc)})
    else:
        trigger_check["passed"] = False
    if not trigger_check["passed"]:
        failures.append(f"Official {trigger_type} trigger could not be loaded/applied")
    checks["trigger"] = trigger_check

    result = {
        "status": "pass" if not failures else "fail",
        "backdoor_alias": args.backdoor_alias,
        "trigger_type": trigger_type,
        "checks": checks,
        "failures": failures,
        "model_zoo_provenance": provenance,
        "interpretation": "Quality is verified from registered Model Zoo metadata plus model/output and official-trigger asset checks; this is not a fresh full-test ASR evaluation.",
    }
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
