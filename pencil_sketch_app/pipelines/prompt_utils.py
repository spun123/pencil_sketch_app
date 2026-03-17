from __future__ import annotations

from pencil_sketch_app.config.settings import DEFAULT_LOCAL_AI_PROMPT, DEFAULT_NEGATIVE_PROMPT


def build_local_ai_prompts(user_prompt: str | None) -> tuple[str, str]:
    base_positive = DEFAULT_LOCAL_AI_PROMPT
    extra = (user_prompt or "").strip()
    if extra:
        positive = f"{base_positive}, {extra}"
    else:
        positive = base_positive
    return positive, DEFAULT_NEGATIVE_PROMPT
