# Анализ тональности отзывов на русском языке

Веб-приложение на Streamlit, которое определяет тональность отзыва (позитивная, нейтральная, негативная)
с помощью предобученной модели [blanchefort/rubert-base-cased-sentiment](https://huggingface.co/blanchefort/rubert-base-cased-sentiment).

Проект выполнен по дисциплине «Проектный практикум» (КГЭУ, кафедра ИТИС).

## Возможности

- анализ одного отзыва: тональность, уверенность модели, вероятности всех классов;
- пакетный анализ CSV-файла с колонкой `text`: распределение тональности, таблица, выгрузка результата в CSV.

## Запуск локально

```bash
git clone https://github.com/<user>/sentiment-app.git
cd sentiment-app
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Приложение откроется по адресу http://localhost:8501. При первом запуске модель (~700 МБ) скачивается
с Hugging Face Hub и кешируется.

## Запуск в GitHub Codespaces

Кнопка **Code → Codespaces → Create codespace**. Окружение из `.devcontainer/devcontainer.json`
само установит зависимости и запустит приложение на порту 8501.

## Тесты

```bash
pip install -r requirements-dev.txt
pytest -v
```

Тесты подменяют модель заглушкой, поэтому идут за секунды и не скачивают веса.
При каждом push и pull request тесты и проверка стиля запускаются в GitHub Actions (`.github/workflows/ci.yml`).

## Структура

| Файл | Назначение |
| --- | --- |
| `app.py` | интерфейс Streamlit |
| `sentiment.py` | загрузка модели и логика анализа |
| `tests/test_sentiment.py` | модульные тесты |
| `examples/reviews.csv` | пример файла для пакетного анализа |
| `.devcontainer/` | окружение для Codespaces |
| `.github/workflows/ci.yml` | непрерывная интеграция |

## Развёртывание в Streamlit Community Cloud

1. Опубликовать репозиторий на GitHub.
2. На [share.streamlit.io](https://share.streamlit.io) нажать **New app**, выбрать репозиторий, ветку `main` и файл `app.py`.
3. После сборки приложение доступно по ссылке вида `https://<имя>.streamlit.app`.
