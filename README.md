
```markdown
# VLM_DEMO — SmolVLM2 Demo

Лёгкое production-ready демо на базе **SmolVLM2** (HuggingFaceTB) с поддержкой:

- Visual Question Answering (VQA)  
- Image Captioning  
- OCR (извлечение текста с изображений)

Сервис состоит из FastAPI-бэкенда и Streamlit-фронтенда, полностью контейнеризирован.

## Особенности

- Поддержка моделей 256M / 500M / 2.2B  
- Локальный кэш моделей и токенизаторов  
- Работает на CPU и GPU (NVIDIA)  
- Zero-downtime перезапуск после изменения кода/конфигурации  
- Проброс GPU через NVIDIA Container Toolkit  
- Простая настройка через `.env`

## Быстрый запуск

```bash
git clone https://github.com/nmaxo/VLM_DEMO.git
cd VLM_DEMO
docker compose up  --build
```

Сервисы будут доступны по адресам:  
- Frontend (Streamlit): http://localhost:8501  
- Backend (FastAPI):    http://localhost:8000  
- Health check:         http://localhost:8000/health

## Конфигурация (.env)

```env
# Устройство
DEVICE=gpu          # cpu | gpu | cuda (все варианты поддерживаются)

# Порты
BACKEND_PORT=8000
FRONTEND_PORT=8501

# Модель
MODEL_SIZE=256M     # 256M | 500M | 2.2B
# VQA_MODEL_ID=...  # опционально

# Пути внутри контейнеров
MODELS_DIR=/models
HF_HOME=/root/.cache/huggingface
```

После изменения `.env`:

```bash
docker compose down
docker compose up -d
```

## API

| Метод   | Эндпоинт             | Описание                                    |
|---------|----------------------|---------------------------------------------|
| POST    | `/api/vqa/init`      | Загрузить изображение → caption + session_id |
| POST    | `/api/vqa/ask`       | Задать вопрос по изображению (по session_id) |
| POST    | `/api/vqa/ocr`       | OCR — вернуть только чистый текст           |

Все эндпоинты принимают `multipart/form-data` с полем `image`.

## Полезные команды

```bash
# Пересборка после изменения кода
docker compose build --no-cache backend

# Перезапуск сервисов
docker compose down && docker compose up -d

# Логи бэкенда
docker compose logs -f backend

# Войти в контейнер
docker compose exec backend bash
```

## Требования для GPU

- NVIDIA драйвер ≥ 525  
- NVIDIA Container Toolkit  
- Docker Compose v2

На CPU работает без дополнительных зависимостей.

## Лицензия

MIT
```
