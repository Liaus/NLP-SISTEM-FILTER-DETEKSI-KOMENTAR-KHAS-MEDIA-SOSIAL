from __future__ import annotations

import streamlit as st

from src.hf_train import load_experiment_model
from src.preprocessing import preprocess_text


st.set_page_config(
    page_title="Deteksi Komentar - Eksperimen HF",
    page_icon="💬",
    layout="centered",
)


@st.cache_resource
def get_experiment_model():
    return load_experiment_model()


st.title("Deteksi Komentar")
st.caption("Model eksperimen dataset Hugging Face — TF-IDF + Multinomial Naive Bayes.")
st.info(
    "Ini model eksperimen terpisah, bukan model utama. Hasilnya merupakan indikasi, "
    "bukan kepastian atau keputusan otomatis."
)

with st.form("comment_detection"):
    comment = st.text_area(
        "Masukkan komentar",
        height=160,
        placeholder="Tulis komentar berbahasa Indonesia...",
    )
    submitted = st.form_submit_button("Deteksi", type="primary")

if submitted:
    if not comment.strip():
        st.warning("Masukkan komentar terlebih dahulu.")
    else:
        processed = preprocess_text(comment)
        if not processed:
            st.warning("Komentar menjadi kosong setelah preprocessing. Coba teks lain.")
        else:
            try:
                model = get_experiment_model()
                label = str(model.predict([processed])[0])
                probabilities = model.predict_proba([processed])[0]
                class_index = list(model.classes_).index(label)
                confidence = float(probabilities[class_index])
            except (FileNotFoundError, OSError, ValueError, AttributeError) as exc:
                st.error(f"Model eksperimen tidak dapat digunakan: {exc}")
            else:
                if label == "aman":
                    st.success("Hasil deteksi: AMAN")
                else:
                    st.error("Hasil deteksi: TOKSIK")
                st.metric("Confidence model", f"{confidence:.1%}")

st.caption(
    "Eksperimen HF menggunakan pemetaan label HS/Abusive yang berbeda dari anotasi "
    "dataset utama. Periksa konteks komentar sebelum mengambil tindakan."
)
