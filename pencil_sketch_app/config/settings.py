from pathlib import Path
import os
import sys

APP_NAME = "Преобразование фото в карандашный рисунок"
OUTPUT_DIR_NAME = "results"
WINDOW_MIN_W = 1280
WINDOW_MIN_H = 840
PREVIEW_MAX_W = 520
PREVIEW_MAX_H = 640
DEFAULT_OPENAI_MODEL = "gpt-image-1"
DEFAULT_EDIT_PROMPT = (
    "Преобразуй загруженную фотографию в чистый профессиональный рисунок простым карандашом. "
    "Сделай белый чистый фон без лишних объектов. "
    "Используй тонкие аккуратные графитовые линии серого цвета вместо чёрных. "
    "Минимизируй штриховку и грязные текстуры. "
    "Сохрани узнаваемость лица, пропорции, волосы, выражение лица и основные детали одежды. "
    "Результат должен выглядеть как качественная линейная иллюстрация, нарисованная художником от руки."
)

DEFAULT_LOCAL_AI_PROMPT = (
    "professional graphite pencil sketch, clean white paper background, thin gray graphite lines, "
    "minimal shading, clean hand-drawn illustration, preserve composition and recognizable details"
)
DEFAULT_NEGATIVE_PROMPT = (
    "photo, photorealistic, color, watercolor, oil painting, messy shading, dirty paper, "
    "black ink, extra fingers, deformed face, text, watermark, logo, low quality, blurry"
)

MODE_LOCAL = "Локальный улучшенный"
MODE_OPENAI = "Онлайн через OpenAI"
MODE_INSTANTID = "Локально: InstantID"
MODE_IPADAPTER = "Локально: IP-Adapter"
MODE_AUTO_AI = "Локально: Авто (InstantID/IP-Adapter)"

LOCAL_MODES = {MODE_LOCAL, MODE_INSTANTID, MODE_IPADAPTER, MODE_AUTO_AI}
AI_REFERENCE_MODES = {MODE_INSTANTID, MODE_IPADAPTER, MODE_AUTO_AI}


def get_app_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


BASE_DIR = get_app_base_dir()
OUTPUT_DIR = BASE_DIR / OUTPUT_DIR_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = BASE_DIR / "models"
INSTANTID_MODELS_DIR = MODELS_DIR / "instantid"
IPADAPTER_MODELS_DIR = MODELS_DIR / "ipadapter"

# Shared local AI paths
LOCAL_AI_BASE_MODEL_DIR = MODELS_DIR / "sdxl_base"
LOCAL_AI_CONTROLNET_DIR = MODELS_DIR / "controlnet"

# IP-Adapter layout
IPADAPTER_BASE_MODEL_DIR = LOCAL_AI_BASE_MODEL_DIR
IPADAPTER_CONTROLNET_DIR = LOCAL_AI_CONTROLNET_DIR
IPADAPTER_ENCODER_DIR = IPADAPTER_MODELS_DIR / "image_encoder"
IPADAPTER_ADAPTER_FILE = IPADAPTER_MODELS_DIR / "ip-adapter_sdxl.bin"

# InstantID layout
INSTANTID_BASE_MODEL_DIR = LOCAL_AI_BASE_MODEL_DIR
INSTANTID_CONTROLNET_DIR = INSTANTID_MODELS_DIR / "ControlNetModel"
INSTANTID_ADAPTER_FILE = INSTANTID_MODELS_DIR / "ip-adapter.bin"
INSTANTID_PIPELINE_FILE = INSTANTID_MODELS_DIR / "pipeline_stable_diffusion_xl_instantid.py"
INSIGHTFACE_ROOT_DIR = MODELS_DIR / "insightface"
INSIGHTFACE_MODEL_NAME = "antelopev2"

# Local AI defaults
LOCAL_AI_DEFAULT_STEPS = 28
LOCAL_AI_DEFAULT_GUIDANCE = 6.0
LOCAL_AI_DEFAULT_WIDTH = 1024
LOCAL_AI_DEFAULT_HEIGHT = 1024
LOCAL_AI_SEED_RANDOM = -1

OPENAI_API_KEY_ENV = os.environ.get("OPENAI_API_KEY", "")
LOCAL_AI_LOW_MEMORY_DEFAULT = True
LOCAL_AI_CPU_MAX_SIDE = 640
LOCAL_AI_FALLBACK_MIN_SIDE = 640
APP_VERSION = "0.5-phase5"
