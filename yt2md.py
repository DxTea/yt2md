"""
yt2md.py

CLI-инструмент для:
- загрузки транскрипта YouTube-видео
- кэширования результата
- литературной обработки через Gemini
- сохранения в Markdown

Использование:
    python yt2md.py <youtube_url>
"""

import os
import sys
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import requests

from google import genai
from google.genai.types import GenerateContentConfig

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

# ===================== НАСТРОЙКИ =====================
MAX_CHARS_PER_CHUNK = 40000
GEMINI_MODEL = "gemini-2.5-flash"

TRANSCRIPTS_DIR = Path("transcripts")
TRANSCRIPTS_DIR.mkdir(exist_ok=True)


# ===================== YOUTUBE =====================

def extract_video_id(url: str) -> str:
    """
    Извлекает video_id из ссылки YouTube или принимает ID напрямую.

    Поддерживаемые форматы:
    - https://youtu.be/<id>
    - https://youtube.com/watch?v=<id>
    - <id>

    :param url: URL или ID
    :return: video_id
    :raises ValueError: если ID не удалось извлечь
    """
    if "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0].split("&")[0]

    if "v=" in url:
        return url.split("v=")[-1].split("&")[0].split("#")[0]

    if len(url) == 11 and url.isalnum():
        return url

    raise ValueError(f"Не удалось извлечь video_id из: {url}")


def get_video_title(url: str) -> str:
    """
    Получает заголовок видео через HTML страницы.

    :param url: ссылка на видео
    :return: заголовок или "Без названия"
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        r = requests.get(url, headers=headers, timeout=8)

        if r.status_code != 200:
            return "Без названия"

        start = r.text.find("<title>") + 7
        end = r.text.find("</title>", start)

        if start > 6 and end > start:
            title = r.text[start:end].strip()

            if title.endswith(" - YouTube"):
                title = title[:-10].strip()

            return title or "Без названия"

    except Exception as e:
        print(f"Ошибка получения заголовка: {e}")

    return "Без названия"


# ===================== CACHE =====================

def load_cached_transcript(video_id: str) -> Optional[str]:
    """
    Загружает транскрипт из локального кэша.

    :param video_id: ID видео
    :return: текст или None
    """
    cache_file = TRANSCRIPTS_DIR / f"{video_id}.txt"

    if cache_file.exists():
        print(f"[{video_id}] Используем кэшированный транскрипт ({cache_file})")
        return cache_file.read_text(encoding="utf-8", errors="replace")

    return None


def save_transcript(video_id: str, text: str) -> None:
    """
    Сохраняет транскрипт в кэш.

    :param video_id: ID видео
    :param text: текст транскрипта
    """
    cache_file = TRANSCRIPTS_DIR / f"{video_id}.txt"
    cache_file.write_text(text, encoding="utf-8")
    print(f"✓ Транскрипт сохранён: {cache_file}")


# ===================== TRANSCRIPT =====================

def get_russian_transcript(video_id: str) -> str:
    """
    Получает русский транскрипт видео:
    1. Проверяет кэш
    2. Загружает через API
    3. Объединяет в строку

    :param video_id: ID видео
    :return: полный текст транскрипта
    """
    cached = load_cached_transcript(video_id)
    if cached is not None:
        print(f"✓ Загружено из кэша {len(cached):,} символов")
        return cached

    print(f"[{video_id}] Загружаем субтитры с YouTube...")
    api = YouTubeTranscriptApi()

    try:
        transcript = api.fetch(video_id=video_id, languages=["ru", "ru-RU"])
        print("✓ Русский трек найден")
    except NoTranscriptFound:
        print("✗ Нет русских субтитров")
        sys.exit(1)
    except TranscriptsDisabled:
        print("✗ Субтитры отключены автором")
        sys.exit(1)
    except VideoUnavailable:
        print("✗ Видео недоступно")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Ошибка: {type(e).__name__} → {e}")
        sys.exit(1)

    full_text = " ".join(
        snippet.text.strip()
        for snippet in transcript
        if snippet.text.strip()
    )

    print(f"✓ Получено {len(full_text):,} символов")

    save_transcript(video_id, full_text)
    return full_text


# ===================== GEMINI =====================

def create_gemini_client():
    """
    Создаёт клиент Gemini на основе API-ключа из .env.

    :return: клиент genai
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("Ошибка: в .env отсутствует GEMINI_API_KEY")
        sys.exit(1)

    return genai.Client(api_key=api_key)


def literary_rewrite(client, text: str) -> str:
    """
    Преобразует транскрипт в литературный текст.

    :param client: Gemini клиент
    :param text: исходный текст
    :return: обработанный текст
    """
    prompt = f"""Ты — очень сильный литературный редактор и писатель на русском языке.
Преврати сырой текст субтитров YouTube-видео в качественную, связную, литературную статью / заметку.

Правила:
- Живой, естественный, хороший русский язык
- Правильная пунктуация, абзацы, ритм текста
- Логические переходы между мыслями
- Можно вставлять подзаголовки ## когда это уместно
- Сохраняй ВСЕ факты, имена, цитаты, даты, числа
- Не придумывай ничего от себя
- Не используй списки, если их не было в смысле оригинала
- Стиль — как хорошая статья в «Republic», «Медуза», «The Village» или глава non-fiction книги

Текст транскрипта:
{text}

Результат (только текст заметки, без ``` и других обёрток):"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=GenerateContentConfig(
                temperature=0.65,
                max_output_tokens=12000,
            )
        )
        return response.text.strip()

    except Exception as e:
        print(f"Ошибка Gemini ({GEMINI_MODEL}): {e}")

        if "quota" in str(e).lower() or "limit" in str(e).lower():
            print("Квота исчерпана. Проверь API-ключ или подожди.")

        return text[:3000] + "\n\n… (ошибка обработки Gemini)"


# ===================== MAIN =====================

def main():
    """
    Основной сценарий работы CLI.

    Этапы:
    1. Проверка аргументов
    2. Создание клиента Gemini
    3. Получение транскрипта
    4. Обработка (с учётом чанков)
    5. Сохранение Markdown
    """
    if len(sys.argv) != 2:
        print("Использование:\n    python yt2md.py <youtube_url>")
        sys.exit(1)

    url = sys.argv[1]

    try:
        client = create_gemini_client()
    except Exception as e:
        print("Ошибка инициализации Gemini:", e)
        sys.exit(1)

    try:
        video_id = extract_video_id(url)
    except ValueError as e:
        print(e)
        sys.exit(1)

    title = get_video_title(url)

    print(f"\n🎥 {title}")
    print(f"   https://youtube.com/{video_id}\n")

    raw_text = get_russian_transcript(video_id)

    # ==== CHUNKING ====
    if len(raw_text) > MAX_CHARS_PER_CHUNK * 1.2:
        print(f"⚠️ Очень длинный текст ({len(raw_text):,} символов) → чанки")

        chunks = [
            raw_text[i:i + MAX_CHARS_PER_CHUNK]
            for i in range(0, len(raw_text), MAX_CHARS_PER_CHUNK)
        ]

        parts = []
        for i, chunk in enumerate(chunks, 1):
            print(f"   Часть {i}/{len(chunks)} …")
            parts.append(literary_rewrite(client, chunk))

        body = "\n\n".join(parts)

    else:
        print("✍️ Литературная обработка …")
        body = literary_rewrite(client, raw_text)

    # ==== SAVE RESULT ====
    safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_') or f"video_{video_id}"
    filename = f"{safe_title}.md"

    content = f"# {title}\n\nИсточник: https://youtu.be/{video_id}\n\n{body}"

    Path(filename).write_text(content, encoding="utf-8")

    print(f"\n✓ Готово → {filename}\n")


if __name__ == "__main__":
    main()