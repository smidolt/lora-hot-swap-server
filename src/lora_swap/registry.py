"""Content-addressed LoRA registry with VRAM-aware LRU and async prefetch."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
from safetensors.torch import load_file

LOG = logging.getLogger(__name__)


@dataclass
class LoRARef:
    name: str
    path: Path
    content_hash: str
    state_dict: Optional[Dict[str, torch.Tensor]] = field(default=None, repr=False)
    last_used_at: float = 0.0


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _state_bytes(state: Dict[str, torch.Tensor]) -> int:
    return sum(t.numel() * t.element_size() for t in state.values())


class LoRARegistry:
    """Registry that keeps a working set of LoRAs in memory under a byte budget.

    - Lookup is by name; content hash is computed at register() and stored on
      the ref so duplicates can be detected by the caller.
    - get() loads on demand and evicts LRU to stay within budget.
    - prefetch() schedules a load in a worker thread; subsequent get() is a
      no-op if the prefetch finished.
    """

    def __init__(
        self,
        vram_budget_bytes: int,
        prefetch_workers: int = 2,
        loader: Optional[Callable[[Path], Dict[str, torch.Tensor]]] = None,
    ) -> None:
        self.vram_budget_bytes = int(vram_budget_bytes)
        self._loader = loader or (lambda p: load_file(str(p), device="cpu"))
        self._lock = threading.Lock()
        self._by_name: Dict[str, LoRARef] = {}
        self._resident: "OrderedDict[str, LoRARef]" = OrderedDict()
        self._resident_bytes = 0
        self._pending: Dict[str, Future] = {}
        self._pool = (
            ThreadPoolExecutor(max_workers=prefetch_workers, thread_name_prefix="lora-prefetch")
            if prefetch_workers > 0 else None
        )

    def register(self, name: str, path: Path) -> LoRARef:
        path = Path(path)
        with self._lock:
            existing = self._by_name.get(name)
            if existing is not None and existing.path == path:
                return existing
        content_hash = _sha256_file(path)
        ref = LoRARef(name=name, path=path, content_hash=content_hash)
        with self._lock:
            self._by_name[name] = ref
        LOG.info("registered %s sha=%s..%s", name, content_hash[:6], content_hash[-4:])
        return ref

    def _evict_until_fits(self, incoming_bytes: int) -> None:
        # Caller holds self._lock.
        while (self._resident_bytes + incoming_bytes) > self.vram_budget_bytes and self._resident:
            victim_name, victim_ref = self._resident.popitem(last=False)
            if victim_ref.state_dict is not None:
                self._resident_bytes -= _state_bytes(victim_ref.state_dict)
                victim_ref.state_dict = None
            LOG.info("evicted %s (resident=%d/%d)", victim_name,
                     self._resident_bytes, self.vram_budget_bytes)

    def _load_into_resident(self, ref: LoRARef) -> None:
        state = self._loader(ref.path)
        size = _state_bytes(state)
        with self._lock:
            self._evict_until_fits(size)
            ref.state_dict = state
            ref.last_used_at = time.time()
            self._resident[ref.name] = ref
            self._resident.move_to_end(ref.name, last=True)
            self._resident_bytes += size

    def get(self, name: str) -> Dict[str, torch.Tensor]:
        """Blocking. Loads from disk and evicts if needed."""
        with self._lock:
            ref = self._by_name.get(name)
            if ref is None:
                raise KeyError(f"LoRA {name!r} not registered")
            pending = self._pending.get(name)
            if ref.state_dict is not None:
                ref.last_used_at = time.time()
                self._resident.move_to_end(ref.name, last=True)
                return ref.state_dict
        if pending is not None:
            pending.result()
            with self._lock:
                return self._by_name[name].state_dict  # type: ignore[return-value]
        self._load_into_resident(ref)
        with self._lock:
            return ref.state_dict  # type: ignore[return-value]

    def prefetch(self, name: str) -> Optional[Future]:
        """Schedule a background load. Idempotent. Returns the running Future."""
        if self._pool is None:
            return None
        with self._lock:
            ref = self._by_name.get(name)
            if ref is None:
                return None
            if ref.state_dict is not None:
                return None
            existing = self._pending.get(name)
            if existing is not None and not existing.done():
                return existing

        def _do() -> None:
            try:
                self._load_into_resident(ref)
            finally:
                with self._lock:
                    self._pending.pop(name, None)

        fut = self._pool.submit(_do)
        with self._lock:
            self._pending[name] = fut
        return fut

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "registered": len(self._by_name),
                "resident": len(self._resident),
                "resident_bytes": self._resident_bytes,
                "budget_bytes": self.vram_budget_bytes,
                "pending": len(self._pending),
            }

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
