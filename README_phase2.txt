Этап 4 проекта

Что уже работает:
- Локальный улучшенный режим line-art
- Онлайн режим через OpenAI
- Локальный IP-Adapter через отдельный модуль
- Локальный InstantID через отдельный модуль
- Автовыбор между InstantID и IP-Adapter

Важно:
- Для режимов IP-Adapter и InstantID нужны локальные модели и зависимости.
- Если какого-то файла не хватает, программа показывает понятное сообщение.

Структура локальных моделей:

models/
  sdxl_base/
  controlnet/
  ipadapter/
    image_encoder/
    ip-adapter_sdxl.bin
  instantid/
    ControlNetModel/
    ip-adapter.bin
    pipeline_stable_diffusion_xl_instantid.py
  insightface/
    models/
      antelopev2/

Что требуется установить:
- torch
- diffusers
- transformers
- accelerate
- safetensors
- insightface
- onnxruntime

Что делает режим Auto:
- если на фото найдено лицо, выбирается InstantID
- если лицо не найдено, выбирается IP-Adapter

Что делать дальше:
1. Скачать модели в папку models
2. Проверить запуск IP-Adapter
3. Проверить запуск InstantID
4. После этого добавить low-memory fallback и упаковку в exe
