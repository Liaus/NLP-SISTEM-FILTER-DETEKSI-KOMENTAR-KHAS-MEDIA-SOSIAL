from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.config import load_config, project_path
from src.predict import predict_comment


st.set_page_config(
    page_title="NLP Comment Detection",
    page_icon="💬",
    layout="wide",
)


@st.cache_resource
def get_model_status() -> bool:
    config = load_config()
    model_path = project_path(*config["model"]["output_path"].split("/"))
    return model_path.exists()


st.title("NLP Comment Detection")
st.caption("Sistem Filter dan Deteksi Komentar Khas Media Sosial Berbahasa Indonesia.")

with st.sidebar:
    st.markdown("### Informasi")
    st.markdown(
        """
        - Model: TF-IDF + Multinomial Naive Bayes
        - Bahasa: Indonesia
        - Tujuan: prototipe deteksi komentar aman vs toksik
        - Batasan: bukan sistem otomatis untuk memblokir komentar di media sosial
        """
    )

    example_comments = [
        "Kamu memang keren dan sangat membantu.",
        "Dasar goblok, jangan sok pintar di sini!",
        "Terima kasih atas masukannya, semoga bermanfaat.",
    ]
    st.markdown("### Contoh komentar")
    for comment in example_comments:
        if st.button(comment, key=f"example_{comment[:20]}"):
            st.session_state["comment_input"] = comment

comment = st.text_area(
    "Masukkan komentar",
    value=st.session_state.get("comment_input", ""),
    height=180,
    placeholder="Tulis komentar yang ingin dideteksi...",
)

if st.button("Deteksi", type="primary"):
    if not comment or not str(comment).strip():
        st.warning("Komentar tidak boleh kosong. Silakan masukkan teks terlebih dahulu.")
    elif not get_model_status():
        st.error("Model belum tersedia. Jalankan training terlebih dahulu sebelum menggunakan aplikasi.")
    else:
        try:
            label, confidence = predict_comment(comment)
            confidence_text = ""
            if confidence is not None:
                confidence_text = f"Confidence: {confidence:.2%}"

            color = "#2e7d32" if label == "aman" else "#d32f2f"
            label_text = "AMAN" if label == "aman" else "TOKSIK"

            st.markdown(
                f"""
                <div style="background-color:{color}; color:white; padding:20px; border-radius:12px; margin-top:12px;">
                    <h3 style="margin:0;">{label_text}</h3>
                    <p style="margin:8px 0 0;">{confidence_text}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if label == "aman":
                explanation = "Komentar ini memiliki pola yang lebih konsisten dengan konteks aman dan tidak menyerang."
            else:
                explanation = "Komentar ini memiliki pola yang mengarah pada serangan atau ujaran merendahkan, sehingga masuk kategori toksik."
            st.info(explanation)
        except Exception as exc:
            st.error(f"Terjadi kesalahan saat prediksi: {exc}")

st.markdown("---")
st.markdown("Model ini menggunakan TF-IDF dan Multinomial Naive Bayes untuk eksperimen pada dataset komentar media sosial berbahasa Indonesia.")
