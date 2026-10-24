"""Модульные тесты логики анализа. Модель подменяется заглушкой, поэтому тесты идут быстро
и не требуют скачивания весов."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentiment import analyze, clean_texts, results_to_frame, summary  # noqa: E402


def fake_classifier(texts, **kwargs):
    """Заглушка: «хорош» -> позитив, «плох» -> негатив, иначе нейтрально."""
    out = []
    for t in texts:
        low = t.lower()
        if "хорош" in low:
            p = {"POSITIVE": 0.9, "NEUTRAL": 0.07, "NEGATIVE": 0.03}
        elif "плох" in low:
            p = {"POSITIVE": 0.02, "NEUTRAL": 0.08, "NEGATIVE": 0.9}
        else:
            p = {"POSITIVE": 0.1, "NEUTRAL": 0.8, "NEGATIVE": 0.1}
        out.append([{"label": k, "score": v} for k, v in p.items()])
    return out


def test_clean_texts_removes_empty_and_spaces():
    assert clean_texts(["  привет   мир ", "", None, "   "]) == ["привет мир"]


def test_clean_texts_truncates_long_text():
    assert len(clean_texts(["а" * 10_000])[0]) == 5000


@pytest.mark.parametrize(
    "text, label, label_ru",
    [
        ("Очень хороший сервис", "POSITIVE", "Позитивная"),
        ("Всё плохо", "NEGATIVE", "Негативная"),
        ("Заказ пришёл в среду", "NEUTRAL", "Нейтральная"),
    ],
)
def test_analyze_labels(text, label, label_ru):
    r = analyze([text], fake_classifier)[0]
    assert r.label == label
    assert r.label_ru == label_ru
    assert 0 <= r.score <= 1


def test_analyze_empty_input():
    assert analyze(["", "  "], fake_classifier) == []


def test_scores_sum_to_one():
    r = analyze(["хорошо"], fake_classifier)[0]
    assert sum(r.scores.values()) == pytest.approx(1.0)


def test_results_to_frame_and_summary():
    results = analyze(["хорошо", "плохо", "плохо", "нормально"], fake_classifier)
    df = results_to_frame(results)
    assert list(df.columns) == ["Текст", "Тональность", "Уверенность"]
    assert len(df) == 4
    counts = summary(results)
    assert counts["Негативная"] == 2
    assert counts["Позитивная"] == 1
    assert counts["Нейтральная"] == 1
