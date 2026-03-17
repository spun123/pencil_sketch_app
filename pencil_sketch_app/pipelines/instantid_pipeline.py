from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib.util
import random

import cv2
import numpy as np

from pencil_sketch_app.config.settings import (
    INSIGHTFACE_MODEL_NAME,
    INSIGHTFACE_ROOT_DIR,
    INSTANTID_ADAPTER_FILE,
    INSTANTID_BASE_MODEL_DIR,
    INSTANTID_CONTROLNET_DIR,
    INSTANTID_PIPELINE_FILE,
    LOCAL_AI_CPU_MAX_SIDE,
    LOCAL_AI_DEFAULT_HEIGHT,
    LOCAL_AI_DEFAULT_STEPS,
    LOCAL_AI_DEFAULT_WIDTH,
    LOCAL_AI_LOW_MEMORY_DEFAULT,
    LOCAL_AI_SEED_RANDOM,
)
from pencil_sketch_app.pipelines.local_ai_preprocess import (
    make_face_keypoint_image,
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
    from diffusers.models import ControlNetModel
except ImportError:
    ControlNetModel = None

try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None


@dataclass
class InstantIDConfig:
    base_model_dir: Path = INSTANTID_BASE_MODEL_DIR
    controlnet_dir: Path = INSTANTID_CONTROLNET_DIR
    adapter_file: Path = INSTANTID_ADAPTER_FILE
    pipeline_file: Path = INSTANTID_PIPELINE_FILE
    insightface_root_dir: Path = INSIGHTFACE_ROOT_DIR
    insightface_model_name: str = INSIGHTFACE_MODEL_NAME
    steps: int = LOCAL_AI_DEFAULT_STEPS
    guidance_scale: float = 4.2
    width: int = LOCAL_AI_DEFAULT_WIDTH
    height: int = LOCAL_AI_DEFAULT_HEIGHT
    controlnet_conditioning_scale: float = 0.78
    adapter_scale_min: float = 0.45
    adapter_scale_max: float = 0.85
    seed: int = LOCAL_AI_SEED_RANDOM
    low_memory_mode: bool = LOCAL_AI_LOW_MEMORY_DEFAULT


def _require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"Не найден {label}: {path}")


def _resolve_device_and_dtype() -> tuple[str, object]:
    if torch is None:
        raise RuntimeError("Для режима InstantID нужен PyTorch.")

    if torch.cuda.is_available():
        return "cuda", torch.float16

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float16

    return "cpu", torch.float32


def validate_instantid_environment(config: InstantIDConfig) -> None:
    if torch is None:
        raise RuntimeError("Для режима InstantID нужен PyTorch. Установите зависимости из requirements.txt")
    if ControlNetModel is None:
        raise RuntimeError("Для режима InstantID нужен пакет diffusers с поддержкой ControlNet.")
    if FaceAnalysis is None:
        raise RuntimeError("Для режима InstantID нужен insightface. Установите зависимости из requirements.txt")

    _require_path(config.base_model_dir, "базовый SDXL-модуль")
    _require_path(config.controlnet_dir, "InstantID ControlNet")
    _require_path(config.adapter_file, "InstantID adapter")
    _require_path(config.pipeline_file, "файл pipeline_stable_diffusion_xl_instantid.py")
    _require_path(config.insightface_root_dir, "папка insightface")
    _require_path(config.insightface_root_dir / "models" / config.insightface_model_name, "модели antelopev2")


def _compute_adapter_scale(similarity_strength: int, config: InstantIDConfig) -> float:
    strength = max(10, min(100, int(similarity_strength)))
    ratio = (strength - 10) / 90.0
    return config.adapter_scale_min + ratio * (config.adapter_scale_max - config.adapter_scale_min)


def _load_instantid_pipeline_class(pipeline_file: Path):
    spec = importlib.util.spec_from_file_location("instantid_pipeline_module", str(pipeline_file))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Не удалось загрузить pipeline-файл InstantID: {pipeline_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pipeline_cls = getattr(module, "StableDiffusionXLInstantIDPipeline", None)
    if pipeline_cls is None:
        raise RuntimeError("В pipeline-файле InstantID не найден класс StableDiffusionXLInstantIDPipeline.")
    return pipeline_cls


def _pick_primary_face(face_info_list: list[dict]) -> dict:
    if not face_info_list:
        raise RuntimeError("Лицо не найдено. Для режима InstantID нужно фото, где лицо хорошо видно.")
    return max(face_info_list, key=lambda item: float((item["bbox"][2] - item["bbox"][0]) * (item["bbox"][3] - item["bbox"][1])))


class InstantIDGenerator:
    def __init__(self, config: InstantIDConfig | None = None) -> None:
        self.config = config or InstantIDConfig()
        self._pipeline = None
        self._face_analyzer = None
        self._device = None
        self._dtype = None

    def _build_face_analyzer(self):
        validate_instantid_environment(self.config)
        device, _ = _resolve_device_and_dtype()
        providers = ["CPUExecutionProvider"]
        ctx_id = -1
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = 0

        analyzer = FaceAnalysis(
            name=self.config.insightface_model_name,
            root=str(self.config.insightface_root_dir),
            providers=providers,
        )
        analyzer.prepare(ctx_id=ctx_id, det_size=(640, 640))
        self._face_analyzer = analyzer
        return analyzer

    def get_face_analyzer(self):
        if self._face_analyzer is None:
            return self._build_face_analyzer()
        return self._face_analyzer

    def _build_pipeline(self):
        validate_instantid_environment(self.config)
        device, dtype = _resolve_device_and_dtype()
        pipeline_cls = _load_instantid_pipeline_class(self.config.pipeline_file)

        controlnet = ControlNetModel.from_pretrained(
            str(self.config.controlnet_dir),
            torch_dtype=dtype,
            use_safetensors=True,
        )

        pipe = pipeline_cls.from_pretrained(
            str(self.config.base_model_dir),
            controlnet=controlnet,
            torch_dtype=dtype,
            use_safetensors=True,
            add_watermarker=False,
        )
        pipe.load_ip_adapter_instantid(str(self.config.adapter_file))
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
        analyzer = self.get_face_analyzer()
        positive_prompt, negative_prompt = build_local_ai_prompts(prompt)

        profile = pick_profile(self.config.low_memory_mode, attempt_index)
        target_long_side = profile.get("max_side") or self.config.width
        if self._device == "cpu":
            target_long_side = min(target_long_side, LOCAL_AI_CPU_MAX_SIDE)

        resized_bgr = np.array(prepare_reference_image(image_bgr, target_long_side=target_long_side).convert("RGB"))
        resized_bgr = cv2.cvtColor(resized_bgr, cv2.COLOR_RGB2BGR)
        face_info_list = analyzer.get(resized_bgr)
        primary_face = _pick_primary_face(face_info_list)

        face_embedding = primary_face.get("embedding")
        face_keypoints = primary_face.get("kps")
        if face_embedding is None or face_keypoints is None:
            raise RuntimeError("InstantID не смог получить ключевые точки или embedding лица.")

        control_image = make_face_keypoint_image(resized_bgr, face_keypoints, target_long_side=target_long_side)
        adapter_scale = _compute_adapter_scale(similarity_strength, self.config)
        if attempt_index > 0:
            adapter_scale = max(self.config.adapter_scale_min, adapter_scale - 0.05 * attempt_index)
        pipe.set_ip_adapter_scale(adapter_scale)

        width, height = control_image.size
        width = max(512, int(round(width / 64) * 64))
        height = max(512, int(round(height / 64) * 64))

        steps = profile.get("steps") or self.config.steps
        guidance_scale = profile.get("guidance_scale") or self.config.guidance_scale
        conditioning_scale = profile.get("conditioning_scale") or self.config.controlnet_conditioning_scale

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
                image_embeds=face_embedding,
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
            "InstantID не смог завершить генерацию даже в режиме экономии памяти. "
            "Попробуйте уменьшить размер изображения или использовать более мощное устройство."
        ) from last_error


_GENERATOR: InstantIDGenerator | None = None


def get_instantid_generator() -> InstantIDGenerator:
    global _GENERATOR
    if _GENERATOR is None:
        _GENERATOR = InstantIDGenerator()
    return _GENERATOR


def generate_with_instantid(
    image_bgr: np.ndarray,
    prompt: str,
    similarity_strength: int = 75,
    low_memory_mode: bool = LOCAL_AI_LOW_MEMORY_DEFAULT,
) -> np.ndarray:
    generator = get_instantid_generator()
    generator.config.low_memory_mode = low_memory_mode
    return generator.generate(
        image_bgr=image_bgr,
        prompt=prompt,
        similarity_strength=similarity_strength,
    )
