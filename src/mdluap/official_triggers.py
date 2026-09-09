"""Strict test-time trigger adapters for Stage 1D-WT.

The mechanism experiment must compare each model with the trigger that was
used by that attack.  This module therefore fails closed: a missing or
incompatible trigger state produces an unavailable adapter instead of a
synthetic replacement trigger.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import importlib.util
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn


@dataclass
class TriggerStatus:
    """Availability and provenance of an attack's test-time trigger."""

    trigger_type: str
    source: str | None
    available: bool
    reason: str | None = None


class TriggerAdapter:
    """Interface for applying a raw-pixel trigger to a batch."""

    def __init__(self, status: TriggerStatus):
        self.status = status

    def apply(self, images: torch.Tensor, *, sample_indices: list[int], split: str) -> torch.Tensor:
        raise NotImplementedError


class UnavailableTrigger(TriggerAdapter):
    """Adapter used when the official trigger cannot be reproduced."""

    def apply(self, images: torch.Tensor, *, sample_indices: list[int], split: str) -> torch.Tensor:
        raise RuntimeError(self.status.reason or "trigger unavailable")


class FixedImageTrigger(TriggerAdapter):
    """BadNet patch or Blended fixed-image trigger."""

    def __init__(self, status: TriggerStatus, image: torch.Tensor, *, mode: str, alpha: float):
        super().__init__(status)
        self.image = image
        self.mode = mode
        self.alpha = float(alpha)

    def apply(self, images: torch.Tensor, *, sample_indices: list[int], split: str) -> torch.Tensor:
        trigger = self.image.to(images.device, images.dtype).unsqueeze(0)
        if self.mode == "badnet":
            return torch.where(trigger > 0, trigger, images)
        return ((1.0 - self.alpha) * images + self.alpha * trigger).clamp(0.0, 1.0)


class WaNetTrigger(TriggerAdapter):
    """Official WaNet identity/noise-grid transformation."""

    def __init__(self, status: TriggerStatus, identity: torch.Tensor, noise: torch.Tensor, *, s: float, grid_rescale: float):
        super().__init__(status)
        self.identity = identity
        self.noise = noise
        self.s = float(s)
        self.grid_rescale = float(grid_rescale)

    def apply(self, images: torch.Tensor, *, sample_indices: list[int], split: str) -> torch.Tensor:
        grid = (self.identity + self.s * self.noise / images.shape[-2]) * self.grid_rescale
        grid = grid.clamp(-1.0, 1.0).expand(images.shape[0], -1, -1, -1)
        return F.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


class SSBAArrayTrigger(TriggerAdapter):
    """Official SSBA sample-specific replacement images."""

    def __init__(self, status: TriggerStatus, replacements: torch.Tensor):
        super().__init__(status)
        self.replacements = replacements

    def apply(self, images: torch.Tensor, *, sample_indices: list[int], split: str) -> torch.Tensor:
        if split != "cifar100_test":
            raise ValueError("Stage 1D-WT SSBA expects a CIFAR-100 test replacement array")
        indices = torch.as_tensor(sample_indices, dtype=torch.long)
        if indices.numel() == 0 or int(indices.max()) >= len(self.replacements):
            raise IndexError("SSBA replacement array does not cover the requested dataset indices")
        return self.replacements[indices].to(images.device, images.dtype)


class InputAwareTrigger(TriggerAdapter):
    """Official BackdoorBench Input-Aware generator/mask composition."""

    def __init__(self, status: TriggerStatus, generator: nn.Module, mask: nn.Module, threshold: nn.Module):
        super().__init__(status)
        self.generator = generator.eval()
        self.mask = mask.eval()
        self.threshold = threshold.eval()
        self.mean = torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1)
        self.std = torch.tensor((0.247, 0.243, 0.261)).view(1, 3, 1, 1)

    @torch.no_grad()
    def apply(self, images: torch.Tensor, *, sample_indices: list[int], split: str) -> torch.Tensor:
        mean = self.mean.to(images.device, images.dtype)
        std = self.std.to(images.device, images.dtype)
        normalized = (images - mean) / std
        pattern = (self.generator(normalized) - mean) / std
        mask = self.threshold(self.mask(normalized))
        triggered_normalized = normalized + (pattern - normalized) * mask
        return (triggered_normalized * std + mean).clamp(0.0, 1.0)


def _image(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()


def _array(path: Path) -> torch.Tensor:
    values = np.asarray(np.load(path))
    tensor = torch.from_numpy(values).float()
    if tensor.ndim != 4 or tuple(tensor.shape[1:]) != (3, 32, 32):
        raise ValueError(f"expected [N,3,32,32] SSBA array, got {tuple(tensor.shape)}")
    if float(tensor.max()) > 1.0:
        tensor = tensor / 255.0
    return tensor.clamp(0.0, 1.0).contiguous()


def _load_state(path: Path, device: torch.device):
    return torch.load(path, map_location=device, weights_only=False)


def _find(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        direct = root / name
        if direct.is_file():
            return direct
    for name in names:
        matches = sorted(root.rglob(name)) if root.exists() else []
        if matches:
            return matches[0]
    return None


def _load_inputaware(path: Path, root: Path, device: torch.device) -> TriggerAdapter:
    root_string = str(root.resolve())
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    from attack.inputaware import InputAwareGenerator, Threshold

    artifact: Any = _load_state(path, device)
    args = SimpleNamespace(dataset="cifar10", input_channel=3)
    generator = InputAwareGenerator(args).to(device)
    mask = InputAwareGenerator(args, out_channels=1).to(device)
    generator_state = artifact.get("netG", artifact.get("generator"))
    mask_state = artifact.get("netM", artifact.get("mask"))
    if generator_state is None or mask_state is None:
        raise ValueError("official Input-Aware netCGM.pt must contain netG and netM")
    generator.load_state_dict({key.removeprefix("module."): value for key, value in generator_state.items()})
    mask.load_state_dict({key.removeprefix("module."): value for key, value in mask_state.items()})
    return InputAwareTrigger(TriggerStatus("inputaware", str(path), True), generator, mask, Threshold().to(device))


def build_trigger_adapters(
    *, model_root: Path, backdoorbench_root: Path, explicit: dict[str, Path | None],
    device: torch.device, blended_alpha: float, wanet_s: float, wanet_grid_rescale: float,
) -> dict[str, TriggerAdapter]:
    """Build only officially sourced adapters for all requested attack families."""

    adapters: dict[str, TriggerAdapter] = {}
    badnet = explicit.get("badnet") or backdoorbench_root / "resource/badnet/trigger_image.png"
    blended = explicit.get("blended") or backdoorbench_root / "resource/blended/hello_kitty.jpeg"
    if badnet.is_file():
        adapters["badnet"] = FixedImageTrigger(TriggerStatus("badnet", str(badnet), True), _image(badnet), mode="badnet", alpha=0.0)
    else:
        adapters["badnet"] = UnavailableTrigger(TriggerStatus("badnet", str(badnet), False, "official BadNet trigger missing"))
    if blended.is_file():
        adapters["blended"] = FixedImageTrigger(TriggerStatus("blended", str(blended), True), _image(blended), mode="blended", alpha=blended_alpha)
    else:
        adapters["blended"] = UnavailableTrigger(TriggerStatus("blended", str(blended), False, "official Blended trigger missing"))

    wanet_dir = model_root / "wanet" / "seed0"
    identity = explicit.get("wanet_identity") or wanet_dir / "state_identity_grid.pt"
    noise = explicit.get("wanet_noise") or wanet_dir / "state_noise_grid.pt"
    state_path = wanet_dir / "state_dict.pt"
    if identity.is_file() and noise.is_file():
        try:
            adapters["wanet"] = WaNetTrigger(TriggerStatus("wanet", f"{identity};{noise}", True), _load_state(identity, device).float(), _load_state(noise, device).float(), s=wanet_s, grid_rescale=wanet_grid_rescale)
        except Exception as exc:
            adapters["wanet"] = UnavailableTrigger(TriggerStatus("wanet", str(identity), False, str(exc)))
    elif state_path.is_file():
        try:
            state = _load_state(state_path, device)
            adapters["wanet"] = WaNetTrigger(TriggerStatus("wanet", str(state_path), True), state["identity_grid"].float(), state["noise_grid"].float(), s=wanet_s, grid_rescale=wanet_grid_rescale)
        except Exception as exc:
            adapters["wanet"] = UnavailableTrigger(TriggerStatus("wanet", str(state_path), False, str(exc)))
    else:
        adapters["wanet"] = UnavailableTrigger(TriggerStatus("wanet", str(identity), False, "official WaNet grid state missing"))

    ssba = explicit.get("ssba")
    if ssba and ssba.is_file():
        try:
            adapters["ssba"] = SSBAArrayTrigger(TriggerStatus("ssba", str(ssba), True), _array(ssba))
        except Exception as exc:
            adapters["ssba"] = UnavailableTrigger(TriggerStatus("ssba", str(ssba), False, str(exc)))
    else:
        adapters["ssba"] = UnavailableTrigger(TriggerStatus("ssba", str(ssba) if ssba else None, False, "no official CIFAR-100 SSBA test-time array supplied"))

    inputaware = explicit.get("inputaware") or model_root / "inputaware" / "seed0" / "netCGM.pt"
    if inputaware.is_file():
        try:
            adapters["inputaware"] = _load_inputaware(inputaware, backdoorbench_root, device)
        except Exception as exc:
            adapters["inputaware"] = UnavailableTrigger(TriggerStatus("inputaware", str(inputaware), False, str(exc)))
    else:
        adapters["inputaware"] = UnavailableTrigger(TriggerStatus("inputaware", str(inputaware), False, "official Input-Aware netCGM.pt missing"))

    adaptive = explicit.get("adaptive_blend")
    if adaptive and adaptive.is_file():
        adapters["adaptive_blend"] = FixedImageTrigger(TriggerStatus("adaptive_blend", str(adaptive), True), _image(adaptive), mode="adaptive_blend", alpha=blended_alpha)
    else:
        adapters["adaptive_blend"] = UnavailableTrigger(TriggerStatus("adaptive_blend", str(adaptive) if adaptive else None, False, "official Adaptive-Blend trigger missing"))
    return adapters
