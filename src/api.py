from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

try:
    from .config import (
        BEST_MODEL_PATH,
        BEST_MODEL_RESULTS_JSON,
        LABEL_ENCODER_PATH,
        PROCESSED_FEATURES_CSV,
        TRAIN_RESULTS_CSV,
        TRAINED_FEATURE_COLUMNS_PATH,
        get_feature_columns,
    )
    from .extract_features import extract_features
except ImportError:  # pragma: no cover
    from config import (
        BEST_MODEL_PATH,
        BEST_MODEL_RESULTS_JSON,
        LABEL_ENCODER_PATH,
        PROCESSED_FEATURES_CSV,
        TRAIN_RESULTS_CSV,
        TRAINED_FEATURE_COLUMNS_PATH,
        get_feature_columns,
    )
    from extract_features import extract_features


class PredictResponse(BaseModel):
    filename: str
    predicted_genre: str
    model_name: str
    probabilities: list[dict[str, Any]]
    extracted_features: dict[str, float]
    feature_comparison: list[dict[str, float | str]]
    notes: list[str]


app = FastAPI(
    title="Music Genre Classifier API",
    description="Backend para classificação supervisionada de gêneros musicais (FMA)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated report images if they exist.
processed_dir = PROCESSED_FEATURES_CSV.parent
if processed_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(processed_dir)), name="assets")


_ARTIFACTS: dict[str, Any] = {}


def _load_artifacts() -> None:
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found: {BEST_MODEL_PATH}")
    if not LABEL_ENCODER_PATH.exists():
        raise FileNotFoundError(f"Label encoder artifact not found: {LABEL_ENCODER_PATH}")
    if not TRAINED_FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError(f"Feature schema artifact not found: {TRAINED_FEATURE_COLUMNS_PATH}")

    _ARTIFACTS["model"] = joblib.load(BEST_MODEL_PATH)
    _ARTIFACTS["label_encoder"] = joblib.load(LABEL_ENCODER_PATH)
    _ARTIFACTS["feature_columns"] = joblib.load(TRAINED_FEATURE_COLUMNS_PATH)


def _get_processed_dataframe() -> pd.DataFrame | None:
    if not PROCESSED_FEATURES_CSV.exists():
        return None
    return pd.read_csv(PROCESSED_FEATURES_CSV)


def _serialize_probabilities(model, X: pd.DataFrame, label_encoder) -> list[dict[str, float | str]]:
    if not hasattr(model, "predict_proba"):
        return []

    probs = model.predict_proba(X)[0]
    classes = getattr(model, "classes_", np.arange(len(probs)))

    payload = []
    for cls_id, prob in zip(classes, probs):
        genre = label_encoder.inverse_transform([int(cls_id)])[0]
        payload.append({"genre": genre, "probability": float(prob)})

    payload.sort(key=lambda item: item["probability"], reverse=True)
    return payload


def _top_feature_comparison(feature_dict: dict[str, float], predicted_genre: str) -> list[dict[str, float | str]]:
    df = _get_processed_dataframe()
    if df is None or "genre_top" not in df.columns:
        return []

    feature_cols = [col for col in get_feature_columns() if col in df.columns]
    if not feature_cols:
        return []

    representative = [
        "tempo_bpm",
        "rms_mean",
        "chroma_mean",
        "spectral_centroid_mean",
        "spectral_bandwidth_mean",
        "zero_crossing_rate_mean",
        "mfcc_1_mean",
        "mfcc_2_mean",
    ]
    representative = [feat for feat in representative if feat in feature_dict and feat in feature_cols]

    global_means = df[feature_cols].mean(numeric_only=True)
    genre_slice = df[df["genre_top"] == predicted_genre]
    if genre_slice.empty:
        return []
    genre_means = genre_slice[feature_cols].mean(numeric_only=True)

    comparison = []
    for feat in representative:
        comparison.append(
            {
                "feature": feat,
                "uploaded_value": float(feature_dict[feat]),
                "predicted_genre_mean": float(genre_means[feat]),
                "global_mean": float(global_means[feat]),
            }
        )

    return comparison


def _best_model_metrics() -> dict[str, Any]:
    if BEST_MODEL_RESULTS_JSON.exists():
        with open(BEST_MODEL_RESULTS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _evaluation_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, str]:
    feature_cols = get_feature_columns()
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].copy()
    y = df["genre_top"].astype(str).copy()

    if "split" in df.columns and set(df["split"].dropna().unique()) >= {"test"}:
        test_mask = df["split"] == "test"
        return X.loc[test_mask], y.loc[test_mask], "official_test_split"

    return X, y, "full_dataset_fallback"


def _model_name(model: Any) -> str:
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        return model.named_steps["model"].__class__.__name__
    return model.__class__.__name__


@app.on_event("startup")
def _startup() -> None:
    try:
        _load_artifacts()
    except FileNotFoundError:
        # API can start even before training artifacts exist.
        pass


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "artifacts_ready": all(path.exists() for path in [BEST_MODEL_PATH, LABEL_ENCODER_PATH, TRAINED_FEATURE_COLUMNS_PATH]),
        "processed_dataset_ready": PROCESSED_FEATURES_CSV.exists(),
    }


@app.get("/api/model-info")
def model_info() -> dict[str, Any]:
    if not _ARTIFACTS:
        try:
            _load_artifacts()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    model = _ARTIFACTS["model"]
    label_encoder = _ARTIFACTS["label_encoder"]

    return {
        "model_name": _model_name(model),
        "classes": list(label_encoder.classes_),
        "n_classes": int(len(label_encoder.classes_)),
        "metrics_summary": _best_model_metrics(),
    }


@app.get("/api/charts/genre-distribution")
def genre_distribution() -> dict[str, Any]:
    df = _get_processed_dataframe()
    if df is None:
        raise HTTPException(status_code=400, detail=f"Processed dataset not found: {PROCESSED_FEATURES_CSV}")

    counts = (
        df["genre_top"]
        .value_counts()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "genre", "genre_top": "count"})
    )

    return {
        "chart": "genre_distribution",
        "data": counts.to_dict(orient="records"),
    }


@app.get("/api/charts/model-comparison")
def model_comparison() -> dict[str, Any]:
    if not TRAIN_RESULTS_CSV.exists():
        raise HTTPException(status_code=400, detail=f"Model comparison file not found: {TRAIN_RESULTS_CSV}")

    df = pd.read_csv(TRAIN_RESULTS_CSV)
    metric_cols = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

    models_payload = []
    for _, row in df.iterrows():
        metrics = {metric: float(row[metric]) for metric in metric_cols if metric in row}
        models_payload.append({"model": str(row["model"]), "metrics": metrics})

    return {
        "chart": "model_comparison",
        "data": models_payload,
    }


@app.get("/api/charts/confusion-matrix")
def confusion_matrix_chart() -> dict[str, Any]:
    if not _ARTIFACTS:
        try:
            _load_artifacts()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    df = _get_processed_dataframe()
    if df is None:
        raise HTTPException(status_code=400, detail=f"Processed dataset not found: {PROCESSED_FEATURES_CSV}")

    model = _ARTIFACTS["model"]
    label_encoder = _ARTIFACTS["label_encoder"]

    X_eval, y_eval_text, split_mode = _evaluation_split(df)
    y_eval = label_encoder.transform(y_eval_text)
    y_pred = model.predict(X_eval)

    cm = confusion_matrix(y_eval, y_pred)
    return {
        "chart": "confusion_matrix",
        "split_mode": split_mode,
        "labels": list(label_encoder.classes_),
        "matrix": cm.tolist(),
    }


@app.get("/api/charts/feature-importance")
def feature_importance() -> dict[str, Any]:
    if not _ARTIFACTS:
        try:
            _load_artifacts()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    model = _ARTIFACTS["model"]
    feature_cols = _ARTIFACTS["feature_columns"]

    estimator = None
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        candidate = model.named_steps["model"]
        if candidate.__class__.__name__ == "RandomForestClassifier":
            estimator = candidate
    elif model.__class__.__name__ == "RandomForestClassifier":
        estimator = model

    if estimator is None or not hasattr(estimator, "feature_importances_"):
        return {
            "chart": "feature_importance",
            "available": False,
            "reason": "Best model is not Random Forest.",
            "data": [],
        }

    importances = estimator.feature_importances_
    imp_df = (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(20)
    )

    return {
        "chart": "feature_importance",
        "available": True,
        "data": imp_df.to_dict(orient="records"),
    }


@app.get("/api/charts/pca")
def pca_projection(max_points: int = 1200) -> dict[str, Any]:
    df = _get_processed_dataframe()
    if df is None:
        raise HTTPException(status_code=400, detail=f"Processed dataset not found: {PROCESSED_FEATURES_CSV}")

    feature_cols = [col for col in get_feature_columns() if col in df.columns]
    if not feature_cols:
        raise HTTPException(status_code=400, detail="Feature columns not found in processed dataset.")

    sample_df = df.copy()
    if len(sample_df) > max_points:
        sample_df = sample_df.sample(n=max_points, random_state=42)

    X = sample_df[feature_cols].values
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X)

    payload = []
    for idx, (_, row) in enumerate(sample_df.iterrows()):
        payload.append(
            {
                "pc1": float(components[idx, 0]),
                "pc2": float(components[idx, 1]),
                "genre": str(row["genre_top"]),
            }
        )

    return {
        "chart": "pca_2d",
        "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "data": payload,
    }


@app.post("/api/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    if not _ARTIFACTS:
        try:
            _load_artifacts()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    filename = file.filename or "uploaded_audio"
    suffix = Path(filename).suffix.lower()
    if suffix not in {".mp3", ".wav"}:
        raise HTTPException(status_code=400, detail="Only .mp3 and .wav files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        raw = await file.read()
        tmp.write(raw)
        tmp_path = Path(tmp.name)

    try:
        feature_dict = extract_features(tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Feature extraction failed: {exc}") from exc
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass

    model = _ARTIFACTS["model"]
    label_encoder = _ARTIFACTS["label_encoder"]
    feature_columns = _ARTIFACTS["feature_columns"]

    X = pd.DataFrame([feature_dict])[feature_columns]
    pred_encoded = int(model.predict(X)[0])
    predicted_genre = str(label_encoder.inverse_transform([pred_encoded])[0])

    probabilities = _serialize_probabilities(model, X, label_encoder)

    # Show top varying features for readability in frontend.
    global_df = _get_processed_dataframe()
    notes = [
        "Predição baseada no Caminho B (features extraídas diretamente do áudio com librosa).",
        "As mesmas features usadas no treino foram usadas para este upload.",
    ]
    if global_df is None:
        extracted_for_return = {k: float(v) for k, v in feature_dict.items()}
    else:
        feature_cols = [col for col in get_feature_columns() if col in global_df.columns]
        means = global_df[feature_cols].mean(numeric_only=True)
        stds = global_df[feature_cols].std(numeric_only=True).replace(0, np.nan)
        zscores = {}
        for feat in feature_cols:
            if feat in feature_dict and pd.notna(stds[feat]):
                zscores[feat] = abs((feature_dict[feat] - means[feat]) / stds[feat])

        top_feats = sorted(zscores.items(), key=lambda item: item[1], reverse=True)[:12]
        selected = [feat for feat, _ in top_feats]
        extracted_for_return = {feat: float(feature_dict[feat]) for feat in selected}

    return PredictResponse(
        filename=filename,
        predicted_genre=predicted_genre,
        model_name=_model_name(model),
        probabilities=probabilities,
        extracted_features=extracted_for_return,
        feature_comparison=_top_feature_comparison(feature_dict, predicted_genre),
        notes=notes,
    )
