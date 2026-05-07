"""
Simple Streamlit app for local prediction.
If you decide to build a React app later, this file can stay as a minimal baseline.
"""

from pathlib import Path
import tempfile

import streamlit as st

try:
    from .predict import predict_genre
except ImportError:  # pragma: no cover
    from predict import predict_genre


st.set_page_config(page_title="Music Genre Classifier", layout="centered")
st.title("Music Genre Classifier (FMA)")
st.write("Upload an `.mp3` or `.wav` file and get the predicted genre.")

uploaded = st.file_uploader("Choose audio file", type=["mp3", "wav"])

if uploaded is not None and st.button("Predict"):
    suffix = Path(uploaded.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = Path(tmp.name)

    try:
        result = predict_genre(tmp_path)
        st.success(f"Predicted genre: **{result['predicted_genre']}**")

        probs = result.get("probabilities")
        if probs:
            st.subheader("Probabilities by genre")
            st.dataframe(
                {
                    "genre": list(probs.keys()),
                    "probability": list(probs.values()),
                },
                use_container_width=True,
            )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
