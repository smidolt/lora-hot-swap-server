"""Forward-hook LoRA adapters with named slots.

The base nn.Linear is never replaced. Adapters live in a hook, with
pre-allocated A/B tensors per module. To switch LoRA, copy new weights
into the existing slot in-place; shapes don't change, the compiled graph
stays valid.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG = logging.getLogger(__name__)


class LoRAAdapter:
    """A, B tensors and scaling for one adapter slot.

    A has shape [rank, in_features], B has shape [out_features, rank].
    """

    def __init__(self, A: torch.Tensor, B: torch.Tensor, alpha: float, rank: int, strength: float = 1.0):
        self.A = A
        self.B = B
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank)
        self.strength = float(strength)

    def to(self, device, dtype) -> "LoRAAdapter":
        self.A = self.A.to(device=device, dtype=dtype)
        self.B = self.B.to(device=device, dtype=dtype)
        return self

    @torch.no_grad()
    def delta_weight(self) -> torch.Tensor:
        return (self.B @ self.A) * (self.scaling * self.strength)


class LoRAHookManager:
    """Per-module forward hooks with multiple named adapter slots."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.module_adapters: Dict[str, Dict[str, LoRAAdapter]] = {}
        self.hooks: Dict[str, torch.utils.hooks.RemovableHandle] = {}

    def _ensure_hook(self, module_name: str, module: nn.Linear) -> None:
        if module_name in self.hooks:
            return

        def _hook(_mod, inputs, output):
            x = inputs[0]
            adapters = self.module_adapters.get(module_name, {})
            if not adapters:
                return output
            add = None
            for ad in adapters.values():
                if ad.strength == 0.0:
                    continue
                Ax = F.linear(x, ad.A)
                BAx = F.linear(Ax, ad.B)
                contrib = BAx * (ad.scaling * ad.strength)
                add = contrib if add is None else add + contrib
            return output + add if add is not None else output

        self.hooks[module_name] = module.register_forward_hook(_hook, with_kwargs=False)

    def _remove_hook(self, module_name: str) -> None:
        h = self.hooks.pop(module_name, None)
        if h is not None:
            h.remove()

    def add_adapter(self, module_name: str, module: nn.Linear, adapter_name: str, adapter: LoRAAdapter) -> None:
        adapter.to(module.weight.device, module.weight.dtype)
        self.module_adapters.setdefault(module_name, {})[adapter_name] = adapter
        self._ensure_hook(module_name, module)

    def remove_adapter(self, adapter_name: str) -> None:
        empty = []
        for module_name, ads in self.module_adapters.items():
            if adapter_name in ads:
                del ads[adapter_name]
            if not ads:
                empty.append(module_name)
        for module_name in empty:
            self._remove_hook(module_name)
            del self.module_adapters[module_name]

    def clear(self) -> None:
        for h in list(self.hooks.values()):
            h.remove()
        self.hooks.clear()
        self.module_adapters.clear()

    def set_strength(self, adapter_name: str, strength: float) -> None:
        for ads in self.module_adapters.values():
            if adapter_name in ads:
                ads[adapter_name].strength = float(strength)

    def active_adapters(self) -> Dict[str, Dict[str, float]]:
        return {m: {n: a.strength for n, a in ads.items()} for m, ads in self.module_adapters.items()}


def _get_module_by_name(root: nn.Module, name: str) -> Optional[nn.Module]:
    mod = root
    if not name:
        return None
    for part in name.split("."):
        if hasattr(mod, part):
            mod = getattr(mod, part)
            continue
        try:
            mod = mod[int(part)]
        except Exception:
            return None
    return mod


def _parse_state_to_adapters(
    state_dict: Dict[str, torch.Tensor],
    alpha_default: Optional[float] = None,
    strength: float = 1.0,
) -> Dict[str, LoRAAdapter]:
    A_map, B_map = {}, {}
    for k, v in state_dict.items():
        if k.endswith(".lora.lora_A"):
            A_map[k.rsplit(".lora.lora_A", 1)[0]] = v
        elif k.endswith(".lora.lora_B"):
            B_map[k.rsplit(".lora.lora_B", 1)[0]] = v
    out = {}
    for module_name in set(A_map) & set(B_map):
        A = A_map[module_name]
        B = B_map[module_name]
        rank = A.shape[0]
        alpha = alpha_default if alpha_default is not None else rank
        out[module_name] = LoRAAdapter(A, B, alpha=alpha, rank=rank, strength=strength)
    return out


def attach_lora_runtime(
    model: nn.Module,
    lora_state: Dict[str, torch.Tensor],
    adapter_name: str = "default",
    alpha: Optional[float] = None,
    strength: float = 1.0,
    include_keywords: Optional[list] = None,
) -> int:
    """Attach a LoRA via forward hooks. Returns the number of modules touched."""
    include_keywords = include_keywords or []
    adapters = _parse_state_to_adapters(lora_state, alpha_default=alpha, strength=strength)

    mgr = getattr(model, "_lora_hooks_manager", None)
    if mgr is None:
        mgr = LoRAHookManager(model)
        model._lora_hooks_manager = mgr

    target_names = (
        {m for m in adapters if any(kw in m for kw in include_keywords)}
        if include_keywords
        else set(adapters)
    )

    ok, missing = 0, 0
    for mname in sorted(target_names):
        mod = _get_module_by_name(model, mname)
        if isinstance(mod, nn.Linear):
            mgr.add_adapter(mname, mod, adapter_name, adapters[mname])
            ok += 1
        else:
            missing += 1
    LOG.info("attach_lora_runtime[%s]: attached=%d missing=%d", adapter_name, ok, missing)
    return ok


def detach_lora_runtime(model: nn.Module, adapter_name: Optional[str] = None) -> None:
    mgr = getattr(model, "_lora_hooks_manager", None)
    if mgr is None:
        return
    if adapter_name is None:
        mgr.clear()
    else:
        mgr.remove_adapter(adapter_name)


def set_lora_strength(model: nn.Module, adapter_name: str, strength: float) -> None:
    mgr = getattr(model, "_lora_hooks_manager", None)
    if mgr is None:
        return
    mgr.set_strength(adapter_name, strength)


def list_active_loras(model: nn.Module) -> Dict[str, Dict[str, float]]:
    mgr = getattr(model, "_lora_hooks_manager", None)
    return mgr.active_adapters() if mgr is not None else {}


@torch.no_grad()
def swap_lora(
    model: nn.Module,
    lora_state: Dict[str, torch.Tensor],
    adapter_name: str = "default",
    alpha: Optional[float] = None,
    strength: Optional[float] = None,
) -> int:
    """Copy new LoRA weights into existing slots, keeping shapes fixed.

    Requires attach_lora_runtime(adapter_name=...) to have been called once
    to allocate the slots. Modules absent from lora_state are zeroed.

    Returns the number of slots updated.
    """
    mgr = getattr(model, "_lora_hooks_manager", None)
    if mgr is None:
        raise RuntimeError(
            "no _lora_hooks_manager on model; call attach_lora_runtime once before swap_lora"
        )
    new = _parse_state_to_adapters(lora_state, alpha_default=alpha, strength=1.0)
    updated = 0
    for module_name, slots in mgr.module_adapters.items():
        if adapter_name not in slots:
            continue
        dst = slots[adapter_name]
        src = new.get(module_name)
        if src is None:
            dst.A.zero_()
            dst.B.zero_()
            continue
        dst.A.copy_(src.A.to(device=dst.A.device, dtype=dst.A.dtype))
        dst.B.copy_(src.B.to(device=dst.B.device, dtype=dst.B.dtype))
        dst.rank = src.rank
        dst.alpha = src.alpha
        dst.scaling = src.scaling
        if strength is not None:
            dst.strength = float(strength)
        updated += 1
    return updated
