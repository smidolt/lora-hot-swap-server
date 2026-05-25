"""Merge LoRA deltas into base weights in-place."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

from .adapters import _get_module_by_name, _parse_state_to_adapters

LOG = logging.getLogger(__name__)


@torch.no_grad()
def bake_lora_into_weights(
    model: nn.Module,
    lora_state: Dict[str, torch.Tensor],
    alpha: Optional[float] = None,
    strength: float = 1.0,
) -> int:
    """Add (strength * B @ A * alpha/rank) to each base nn.Linear in-place.

    Returns the number of modules merged.
    """
    adapters = _parse_state_to_adapters(lora_state, alpha_default=alpha, strength=strength)
    count = 0
    for module_name, ad in adapters.items():
        mod = _get_module_by_name(model, module_name)
        if not isinstance(mod, nn.Linear):
            LOG.warning("bake_lora: skipping %s (not nn.Linear)", module_name)
            continue
        ad.to(mod.weight.device, mod.weight.dtype)
        mod.weight.data.add_(ad.delta_weight())
        count += 1
    LOG.info("bake_lora: merged into %d modules (strength=%.3f)", count, strength)
    return count
