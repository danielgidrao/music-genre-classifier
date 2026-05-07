from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    from .config import (
        BEST_MODEL_PATH,
        LABEL_ENCODER_PATH,
        SCALER_PATH,
        TRAINED_FEATURE_COLUMNS_PATH,
    )
    from .extract_features import extract_features
except ImportError:  # pragma: no cover
    from config import (
        BEST_MODEL_PATH,
        LABEL_ENCODER_PATH,
        SCALER_PATH,
        TRAINED_FEATURE_COLUMNS_PATH,
    )
    from extract_features import extract_features


def load_artifacts():
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Best model not found: {BEST_MODEL_PATH}")
    if not LABEL_ENCODER_PATH.exists():
        raise FileNotFoundError(f"Label encoder not found: {LABEL_ENCODER_PATH}")
    if not TRAINED_FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError(f"Feature columns file not found: {TRAINED_FEATURE_COLUMNS_PATH}")

    model = joblib.load(BEST_MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    feature_columns = joblib.load(TRAINED_FEATURE_COLUMNS_PATH)

    scaler = None
    if SCALER_PATH.exists():
        scaler = joblib.load(SCALER_PATH)

    return model, label_encoder, feature_columns, scaler


def predict_genre(audio_path: str | Path) -> dict:
    """
    Predict genre for a single .mp3/.wav file.

    Returns:
    {
      "predicted_genre": str,
      "probabilities": {"genre": prob, ...} or None
    }
    """
    model, label_encoder, feature_columns, _ = load_artifacts()
    feature_dict = extract_features(Path(audio_path))

    X = pd.DataFrame([feature_dict])
    X = X[feature_columns]

    pred_encoded = int(model.predict(X)[0])
    predicted_genre = label_encoder.inverse_transform([pred_encoded])[0]

    probas = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        classes = getattr(model, "classes_", np.arange(len(probs)))
        probas = {
            label_encoder.inverse_transform([int(cls_id)])[0]: float(prob)
            for cls_id, prob in zip(classes, probs)
        }
        probas = dict(sorted(probas.items(), key=lambda item: item[1], reverse=True))

    return {
        "predicted_genre": predicted_genre,
        "probabilities": probas,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict music genre from an audio file")
    parser.add_argument("audio_path", type=Path, help="Path to .mp3 or .wav file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = predict_genre(args.audio_path)

    print("Predicted genre:", result["predicted_genre"])
    if result["probabilities"] is not None:
        print("\nProbabilities:")
        for genre, prob in result["probabilities"].items():
            print(f"  {genre}: {prob:.4f}")


if __name__ == "__main__":
    main()
