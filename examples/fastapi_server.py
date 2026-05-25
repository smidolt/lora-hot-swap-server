"""Minimal FastAPI server: one model, runtime LoRA swap per request.

    uvicorn examples.fastapi_server:app --host 0.0.0.0 --port 8000

    curl -X POST http://localhost:8000/generate \\
         -H 'content-type: application/json' \\
         -d '{"prompt": "studio portrait", "lora": "my_lora", "width": 768, "height": 1024}' \\
         --output out.webp
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from lora_swap import LoRARegistry, attach_lora_runtime, swap_lora
from lora_swap.keymap import remap_diffusers_keys, remap_kohya_keys

LOG = logging.getLogger("lora-server")
logging.basicConfig(level=logging.INFO)

MODEL_ID = os.environ.get("MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
LORA_DIR = Path(os.environ.get("LORA_DIR", "./loras"))
VRAM_BUDGET_GB = float(os.environ.get("VRAM_BUDGET_GB", "8"))

app = FastAPI()

pipe = None
registry: LoRARegistry


class GenRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    lora: Optional[str] = None
    strength: float = 1.0
    steps: int = 8
    guidance_scale: float = 0.0
    width: int = 768
    height: int = 1024
    seed: int = 42


@app.on_event("startup")
def _startup() -> None:
    global pipe, registry  # noqa: PLW0603
    from diffusers import DiffusionPipeline

    LOG.info("loading pipeline %s", MODEL_ID)
    pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    # Allocate the runtime slot once; later swap_lora() calls copy into it.
    attach_lora_runtime(pipe.transformer, {}, adapter_name="runtime", strength=0.0)

    registry = LoRARegistry(vram_budget_bytes=int(VRAM_BUDGET_GB * 1024 ** 3))
    for path in sorted(LORA_DIR.glob("*.safetensors")):
        registry.register(name=path.stem, path=path)
    LOG.info("registry: %s", registry.stats())


@app.post("/generate")
def generate(req: GenRequest):
    if pipe is None:
        raise HTTPException(503, "pipeline not ready")

    if req.lora:
        try:
            state = registry.get(req.lora)
        except KeyError as e:
            raise HTTPException(404, str(e))
        if any("lora_unet_" in k for k in state):
            state = remap_kohya_keys(state, pipe.transformer)
        else:
            state = remap_diffusers_keys(state)
        swap_lora(pipe.transformer, state, adapter_name="runtime", strength=req.strength)
    else:
        swap_lora(pipe.transformer, {}, adapter_name="runtime", strength=0.0)

    image = pipe(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        width=req.width,
        height=req.height,
        num_inference_steps=req.steps,
        guidance_scale=req.guidance_scale,
        generator=torch.Generator("cuda").manual_seed(req.seed),
    ).images[0]

    buf = io.BytesIO()
    image.save(buf, format="WEBP", quality=92)
    return Response(content=buf.getvalue(), media_type="image/webp")


@app.get("/stats")
def stats():
    return registry.stats()
