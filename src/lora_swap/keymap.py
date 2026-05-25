"""Normalize LoRA state-dict keys to the diffusers convention.

Trained files come in several flavors:
  - kohya / WAN / VideoX-Fun: ``lora_unet__layers_0_attention_to_k.lora_down.weight``
  - diffusers / PEFT:         ``layers.0.attention.to_k.lora.lora_A``

Both map to the diffusers form, which is what ``adapters._parse_state_to_adapters``
expects.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


def remap_kohya_keys(
    lora_state: Dict[str, torch.Tensor],
    transformer: nn.Module,
) -> Dict[str, torch.Tensor]:
    """Map ``lora_unet__<flat>.lora_down.weight`` -> ``<dotted>.lora.lora_A``.

    The reverse lookup is built from the live model's named_modules, so the
    same function works across model families without hand-written maps.
    """
    module_lookup = {}
    for name, mod in transformer.named_modules():
        if hasattr(mod, "weight"):
            module_lookup[name.replace(".", "_")] = name

    remapped: Dict[str, torch.Tensor] = {}
    for key, tensor in lora_state.items():
        k = key
        if k.startswith("lora_unet__"):
            k = k[len("lora_unet__"):]
        if ".lora_down.weight" in k:
            flat = k.replace(".lora_down.weight", "")
            suffix = ".lora.lora_A"
        elif ".lora_up.weight" in k:
            flat = k.replace(".lora_up.weight", "")
            suffix = ".lora.lora_B"
        else:
            continue
        if flat in module_lookup:
            remapped[module_lookup[flat] + suffix] = tensor
    return remapped


def remap_diffusers_keys(lora_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Pass-through for files already in diffusers form; drops non-LoRA keys."""
    return {
        k: v for k, v in lora_state.items()
        if k.endswith(".lora.lora_A") or k.endswith(".lora.lora_B")
    }
