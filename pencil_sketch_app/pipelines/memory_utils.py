from __future__ import annotations

from dataclasses import replace


class RecoverableGenerationError(RuntimeError):
    pass


OOM_TEXT_MARKERS = (
    "out of memory",
    "cuda out of memory",
    "mps backend out of memory",
    "not enough memory",
    "alloc",
)


def is_memory_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in OOM_TEXT_MARKERS)


def apply_runtime_memory_optimizations(pipe, device: str) -> None:
    methods = [
        "enable_attention_slicing",
        "enable_vae_slicing",
        "enable_vae_tiling",
        "enable_sequential_cpu_offload",
        "enable_model_cpu_offload",
    ]

    for method_name in methods:
        method = getattr(pipe, method_name, None)
        if callable(method):
            try:
                if method_name in {"enable_model_cpu_offload", "enable_sequential_cpu_offload"}:
                    if device == "cuda":
                        method()
                else:
                    method()
            except Exception:
                pass


LOW_MEMORY_PROFILES = [
    {
        "label": "обычный",
        "steps": None,
        "width": None,
        "height": None,
        "guidance_scale": None,
        "conditioning_scale": None,
    },
    {
        "label": "fallback-1",
        "steps": 24,
        "max_side": 896,
        "guidance_scale": 5.5,
        "conditioning_scale": 0.72,
    },
    {
        "label": "fallback-2",
        "steps": 18,
        "max_side": 768,
        "guidance_scale": 5.0,
        "conditioning_scale": 0.68,
    },
    {
        "label": "fallback-3",
        "steps": 14,
        "max_side": 640,
        "guidance_scale": 4.8,
        "conditioning_scale": 0.64,
    },
]


def pick_profile(low_memory_mode: bool, attempt_index: int) -> dict:
    if not low_memory_mode:
        return LOW_MEMORY_PROFILES[min(attempt_index, len(LOW_MEMORY_PROFILES) - 1)]
    safe_index = min(max(1, attempt_index + 1), len(LOW_MEMORY_PROFILES) - 1)
    return LOW_MEMORY_PROFILES[safe_index]
