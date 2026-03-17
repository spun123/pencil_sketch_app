from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np

from pencil_sketch_app.config.settings import (
    IPADAPTER_ADAPTER_FILE,
    IPADAPTER_BASE_MODEL_DIR,
    IPADAPTER_CONTROLNET_DIR,
    IPADAPTER_ENCODER_DIR,
    LOCAL_AI_CPU_MAX_SIDE,
    LOCAL_AI_DEFAULT_GUIDANCE,
    LOCAL_AI_DEFAULT_HEIGHT,
    LOCAL_AI_DEFAULT_STEPS,
    LOCAL_AI_DEFAULT_WIDTH,
    LOCAL_AI_LOW_MEMORY_DEFAULT,
    LOCAL_AI_SEED_RANDOM,
)
from pencil_sketch_app.pipelines.local_ai_preprocess import (
    make_control_image,
    pil_to_bgr,
    prepare_reference_image,
)
from pencil_sketch_app.pipelines.memory_utils import (
    RecoverableGenerationError,
    apply_runtime_memory_optimizations,
    is_memory_error,
    pick_profile,
)
from pencil_sketch_app.pipelines.prompt_utils import build_local_ai_prompts

try:
    import torch
except ImportError:
    torch = None

try:
    from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
except ImportError:
    ControlNetModel = None
    StableDiffusionXLControlNetPipeline = None


@dataclass
class IPAdapterConfig:
    base_model_dir: Path = IPADAPTER_BASE_MODEL_DIR
    controlnet_dir: Path = IPADAPTER_CONTROLNET_DIR
    image_encoder_dir: Path = IPADAPTER_ENCODER_DIR
    adapter_file: Path = IPADAPTER_ADAPTER_FILE
    steps: int = LOCAL_AI_DEFAULT_STEPS
    guidance_scale: float = LOCAL_AI_DEFAULT_GUIDANCE
    width: int = LOCAL_AI_DEFAULT_WIDTH
    height: int = LOCAL_AI_DEFAULT_HEIGHT
    controlnet_conditioning_scale: float = 0.78
    adapter_scale_min: float = 0.35
    adapter_scale_max: float = 0.9
    seed: int = LOCAL_AI_SEED_RANDOM
    low_memory_mode: bool = LOCAL_AI_LOW_MEMORY_DEFAULT


def _resolve_device_and_dtype() -> tuple[str, object]:
    if torch is None:
        raise RuntimeError("Для режима IP-Adapter нужен PyTorch.")

    if torch.cuda.is_available():
        return "cuda", torch.float16

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float16

    return "cpu", torch.float32


def _require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"Не найден {label}: {path}")


def validate_ipadapter_environment(config: IPAdapterConfig) -> None:
    if torch is None:
        raise RuntimeError(
            "Для режима IP-Adapter нужен PyTorch. Установите зависимости из requirements.txt"
        )

    if StableDiffusionXLControlNetPipeline is None or ControlNetModel is None:
        raise RuntimeError(
            "Для режима IP-Adapter нужен пакет diffusers с поддержкой SDXL + ControlNet."
        )

    _require_path(config.base_model_dir, "базовый SDXL-модуль")
    _require_path(config.controlnet_dir, "ControlNet-модуль")
    _require_path(config.image_encoder_dir, "image_encoder")
    _require_path(config.adapter_file, "файл IP-Adapter")


def _compute_adapter_scale(similarity_strength: int, config: IPAdapterConfig) -> float:
    strength = max(10, min(100, int(similarity_strength)))
    ratio = (strength - 10) / 90.0
    return config.adapter_scale_min + ratio * (config.adapter_scale_max - config.adapter_scale_min)


class IPAdapterGenerator:
    def __init__(self, config: IPAdapterConfig | None = None) -> None:
        self.config = config or IPAdapterConfig()
        self._pipeline = None
        self._device = None
        self._dtype = None

    def _build_pipeline(self):
        validate_ipadapter_environment(self.config)
        device, dtype = _resolve_device_and_dtype()

        controlnet = ControlNetModel.from_pretrained(
            str(self.config.controlnet_dir),
            torch_dtype=dtype,
            use_safetensors=True,
        )
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            str(self.config.base_model_dir),
            controlnet=controlnet,
            torch_dtype=dtype,
            use_safetensors=True,
            add_watermarker=False,
        )
        pipe.load_ip_adapter(
            str(self.config.image_encoder_dir),
            subfolder="",
            weight_name=str(self.config.adapter_file.name),
        )
        apply_runtime_memory_optimizations(pipe, device)

        if device in {"cuda", "mps"}:
            pipe = pipe.to(device)
        else:
            pipe = pipe.to("cpu")

        self._pipeline = pipe
        self._device = device
        self._dtype = dtype
        return pipe

    def get_pipeline(self):
        if self._pipeline is None:
            return self._build_pipeline()
        return self._pipeline

    def _run_single_pass(self, image_bgr: np.ndarray, prompt: str, similarity_strength: int, attempt_index: int = 0) -> np.ndarray:
        pipe = self.get_pipeline()
        positive_prompt, negative_prompt = build_local_ai_prompts(prompt)

        profile = pick_profile(self.config.low_memory_mode, attempt_index)
        control_target = profile.get("max_side") or self.config.width
        if self._device == "cpu":
            control_target = min(control_target, LOCAL_AI_CPU_MAX_SIDE)

        reference_image = prepare_reference_image(image_bgr, target_long_side=control_target)
        control_image = make_control_image(image_bgr, target_long_side=control_target)
        width, height = control_image.size
        width = max(512, int(round(width / 64) * 64))
        height = max(512, int(round(height / 64) * 64))

        if profile.get("max_side"):
            width = min(width, int(profile["max_side"]))
            height = min(height, int(profile["max_side"])) if height > width else height
            width = max(512, int(round(width / 64) * 64))
            height = max(512, int(round(height / 64) * 64))

        steps = profile.get("steps") or self.config.steps
        guidance_scale = profile.get("guidance_scale") or self.config.guidance_scale
        conditioning_scale = profile.get("conditioning_scale") or self.config.controlnet_conditioning_scale

        adapter_scale = _compute_adapter_scale(similarity_strength, self.config)
        if attempt_index > 0:
            adapter_scale = max(self.config.adapter_scale_min, adapter_scale - 0.06 * attempt_index)
        pipe.set_ip_adapter_scale(adapter_scale)

        seed = self.config.seed
        if seed == LOCAL_AI_SEED_RANDOM:
            seed = random.randint(1, 2_147_483_647)

        generator = None
        if torch is not None:
            generator = torch.Generator(device=self._device if self._device != "mps" else "cpu").manual_seed(seed)

        try:
            result = pipe(
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                image=control_image,
                ip_adapter_image=reference_image,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=conditioning_scale,
                width=width,
                height=height,
                generator=generator,
            ).images[0]
        except Exception as exc:
            if is_memory_error(exc):
                raise RecoverableGenerationError(str(exc)) from exc
            raise

        return pil_to_bgr(result)

    def generate(self, image_bgr: np.ndarray, prompt: str, similarity_strength: int = 75) -> np.ndarray:
        last_error: Exception | None = None
        for attempt_index in range(4):
            try:
                return self._run_single_pass(image_bgr, prompt, similarity_strength, attempt_index)
            except RecoverableGenerationError as exc:
                last_error = exc
                if torch is not None and torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                continue

        raise RuntimeError(
            "IP-Adapter не смог завершить генерацию даже в режиме экономии памяти. "
            "Попробуйте уменьшить размер изображения или использовать более мощное устройство."
        ) from last_error


_GENERATOR: IPAdapterGenerator | None = None


def get_ipadapter_generator() -> IPAdapterGenerator:
    global _GENERATOR
    if _GENERATOR is None:
        _GENERATOR = IPAdapterGenerator()
    return _GENERATOR


def generate_with_ipadapter(
    image_bgr: np.ndarray,
    prompt: str,
    similarity_strength: int = 75,
    low_memory_mode: bool = LOCAL_AI_LOW_MEMORY_DEFAULT,
) -> np.ndarray:
    generator = get_ipadapter_generator()
    generator.config.low_memory_mode = low_memory_mode
    return generator.generate(
        image_bgr=image_bgr,
        prompt=prompt,
        similarity_strength=similarity_strength,
    )
