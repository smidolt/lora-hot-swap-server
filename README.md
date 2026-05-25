# lora-swap

Runtime LoRA management for diffusion transformers. Hook-based adapters
with pre-allocated A/B slots, so switching a LoRA is a `tensor.copy_()`
into existing storage and the compiled graph stays valid.

## Install

```
pip install -e .
pip install -e .[server]   # optional, for the FastAPI example
```

Requires `torch>=2.5` and `safetensors`.

## Use

```python
from lora_swap import attach_lora_runtime, swap_lora, LoRARegistry
from lora_swap.keymap import remap_kohya_keys

# once, at startup: allocate the slot
attach_lora_runtime(pipe.transformer, {}, adapter_name="runtime", strength=0.0)

# build the registry
registry = LoRARegistry(vram_budget_bytes=8 * 1024**3)
for path in lora_dir.glob("*.safetensors"):
    registry.register(name=path.stem, path=path)

# per request
state = registry.get("my_lora")
state = remap_kohya_keys(state, pipe.transformer)   # if needed
swap_lora(pipe.transformer, state, adapter_name="runtime", strength=0.9)
image = pipe(prompt=..., ...).images[0]
```

`swap_lora` copies into the slots allocated by `attach_lora_runtime`.
Modules absent from the new state dict are zeroed so the previous LoRA
doesn't leak.

For the frozen path, merge into the base weights before serving:

```python
from lora_swap import bake_lora_into_weights
bake_lora_into_weights(pipe.transformer, lora_state, strength=1.0)
```

## Layout

```
src/lora_swap/
    adapters.py    hook manager, attach_lora_runtime, swap_lora, set_lora_strength
    baking.py      bake_lora_into_weights
    keymap.py      remap_kohya_keys, remap_diffusers_keys
    registry.py    LoRARegistry: content hashes, LRU, byte budget, prefetch
examples/
    fastapi_server.py
```

## Notes

The hook never replaces `nn.Linear`; it adds `(B @ A) * scale * strength`
to the output. Replacing the module (the `LinearWithLoRA` pattern) changes
the module identity and invalidates a `torch.compile`'d graph. The hook
path keeps the graph and only mutates tensor values.

`strength == 0.0` short-circuits the hook, so an idle slot costs one
branch per linear and nothing else.

The registry holds an `OrderedDict` of resident state dicts. `get()` is
blocking; `prefetch()` schedules a background load and is a no-op if the
file is already resident or already in flight. Eviction is by oldest
access until the byte budget fits.

## License

Apache-2.0.
# lora-hot-swap-server
