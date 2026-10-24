"""Streamlit-приложение: анализ тональности отзывов на русском языке."""
import pandas as pd
import streamlit as st

from sentiment import (
    LABELS_RU,
    MODEL_NAME,
    analyze,
    load_classifier,
    results_to_frame,
    summary,
)

EMOJI = {"POSITIVE": "🙂", "NEUTRAL": "😐", "NEGATIVE": "🙁"}

st.set_page_config(page_title="Анализ тональности отзывов", page_icon="💬")


@st.cache_resource(show_spinner="Загружаем модель…")
def get_classifier():
    return load_classifier()


st.title("💬 Анализ тональности отзывов")
st.caption(f"Модель: [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME})")

classifier = get_classifier()
tab_text, tab_file = st.tabs(["Один отзыв", "Файл с отзывами (CSV)"])

with tab_text:
    text = st.text_area(
        "Введите текст отзыва",
        height=150,
        placeholder="Например: Заказ привезли вовремя, всё понравилось!",
    )
    if st.button("Определить тональность", type="primary"):
        results = analyze([text], classifier)
        if not results:
            st.warning("Введите непустой текст.")
        else:
            r = results[0]
            st.subheader(f"{EMOJI[r.label]} {r.label_ru}")
            st.metric("Уверенность модели", f"{r.score:.1%}")
            probs = pd.DataFrame(
                {"Вероятность": [r.scores.get(k, 0.0) for k in LABELS_RU]},
                index=list(LABELS_RU.values()),
            )
            st.bar_chart(probs)

with tab_file:
    st.write("Загрузите CSV-файл с колонкой **text** — по одному отзыву в строке.")
    uploaded = st.file_uploader("CSV-файл", type="csv")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        if "text" not in df.columns:
            st.error("В файле нет колонки «text».")
        else:
            with st.spinner(f"Анализируем {len(df)} отзывов…"):
                results = analyze(df["text"].tolist(), classifier)
            table = results_to_frame(results)
            st.subheader("Распределение тональности")
            st.bar_chart(summary(results))
            st.dataframe(table, use_container_width=True)
            st.download_button(
                "Скачать результаты (CSV)",
                table.to_csv(index=False).encode("utf-8-sig"),
                file_name="sentiment_results.csv",
                mime="text/csv",
            )
