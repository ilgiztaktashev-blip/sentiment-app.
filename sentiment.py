"""Анализ тональности русскоязычных текстов с помощью предобученной модели RuBERT."""
from dataclasses import dataclass
from typing import Callable, Iterable, List

import pandas as pd

MODEL_NAME = "blanchefort/rubert-base-cased-sentiment"
MAX_TEXT_LENGTH = 5000  # символов; длиннее — обрезаем, модель всё равно видит 512 токенов

# Метки модели -> русские названия
LABELS_RU = {
    "POSITIVE": "Позитивная",
    "NEUTRAL": "Нейтральная",
    "NEGATIVE": "Негативная",
}


@dataclass
class SentimentResult:
    text: str
    label: str          # метка модели: POSITIVE / NEUTRAL / NEGATIVE
    label_ru: str       # русское название
    score: float        # уверенность модели в метке, 0..1
    scores: dict        # вероятности всех трёх классов


def load_classifier() -> Callable:
    """Загружает модель из Hugging Face Hub (при первом запуске скачивается ~700 МБ)."""
    from transformers import pipeline

    return pipeline("text-classification", model=MODEL_NAME, top_k=None)


def clean_texts(texts: Iterable[str]) -> List[str]:
    """Убирает пустые строки и лишние пробелы, обрезает слишком длинные тексты."""
    result = []
    for t in texts:
        if t is None or (isinstance(t, float) and pd.isna(t)):
            continue
        t = " ".join(str(t).split())
        if t:
            result.append(t[:MAX_TEXT_LENGTH])
    return result


def analyze(texts: Iterable[str], classifier: Callable) -> List[SentimentResult]:
    """Определяет тональность каждого текста."""
    texts = clean_texts(texts)
    if not texts:
        return []
    raw = classifier(texts, truncation=True, max_length=512)
    results = []
    for text, preds in zip(texts, raw):
        scores = {p["label"]: float(p["score"]) for p in preds}
        label = max(scores, key=scores.get)
        results.append(
            SentimentResult(
                text=text,
                label=label,
                label_ru=LABELS_RU.get(label, label),
                score=scores[label],
                scores=scores,
            )
        )
    return results


def results_to_frame(results: List[SentimentResult]) -> pd.DataFrame:
    """Таблица результатов для показа и выгрузки в CSV."""
    return pd.DataFrame(
        {
            "Текст": [r.text for r in results],
            "Тональность": [r.label_ru for r in results],
            "Уверенность": [round(r.score, 3) for r in results],
        }
    )


def summary(results: List[SentimentResult]) -> pd.Series:
    """Количество текстов каждой тональности (в фиксированном порядке)."""
    counts = pd.Series(0, index=list(LABELS_RU.values()), name="Количество")
    for r in results:
        counts[r.label_ru] += 1
    return counts
