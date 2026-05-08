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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

try:
    from .config import (
        BEST_MODEL_PATH,
        BEST_MODEL_RESULTS_JSON,
        LABEL_ENCODER_PATH,
        PATH_A_FEATURES_CSV,
        PROCESSED_FEATURES_CSV,
        TRAIN_RESULTS_CSV,
        TRAINED_FEATURE_COLUMNS_PATH,
    )
    from .load_data import infer_tabular_feature_columns
    from .predict import extract_features_for_model, infer_feature_mode
except ImportError:  # pragma: no cover
    from config import (
        BEST_MODEL_PATH,
        BEST_MODEL_RESULTS_JSON,
        LABEL_ENCODER_PATH,
        PATH_A_FEATURES_CSV,
        PROCESSED_FEATURES_CSV,
        TRAIN_RESULTS_CSV,
        TRAINED_FEATURE_COLUMNS_PATH,
    )
    from load_data import infer_tabular_feature_columns
    from predict import extract_features_for_model, infer_feature_mode


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
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _resolve_processed_dataset_path() -> Path | None:
    if PATH_A_FEATURES_CSV.exists():
        return PATH_A_FEATURES_CSV
    if PROCESSED_FEATURES_CSV.exists():
        return PROCESSED_FEATURES_CSV
    return None


# Serve generated report images if they exist.
processed_path = _resolve_processed_dataset_path()
processed_dir = processed_path.parent if processed_path is not None else PROCESSED_FEATURES_CSV.parent
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
    path = _resolve_processed_dataset_path()
    if path is None:
        return None
    return pd.read_csv(path)


def _get_feature_columns(df: pd.DataFrame | None = None) -> list[str]:
    if "feature_columns" in _ARTIFACTS:
        return list(_ARTIFACTS["feature_columns"])
    if df is not None:
        return infer_tabular_feature_columns(df)
    return []


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

    feature_cols = [col for col in _get_feature_columns(df) if col in df.columns and col in feature_dict]
    if not feature_cols:
        return []

    preferred = [
        "tempo_bpm",
        "rms_mean",
        "chroma_mean",
        "spectral_centroid_mean",
        "spectral_bandwidth_mean",
        "zero_crossing_rate_mean",
        "mfcc_1_mean",
        "mfcc_2_mean",
        "rmse_mean_01",
        "zcr_mean_01",
        "spectral_centroid_mean_01",
        "spectral_bandwidth_mean_01",
        "spectral_rolloff_mean_01",
        "mfcc_mean_01",
        "mfcc_mean_02",
    ]
    representative = [feat for feat in preferred if feat in feature_cols]

    global_means = df[feature_cols].mean(numeric_only=True)
    global_stds = df[feature_cols].std(numeric_only=True).replace(0, np.nan)

    if not representative:
        # Fallback: select most distinctive features for this uploaded sample.
        zscores = {
            feat: abs((feature_dict[feat] - global_means[feat]) / global_stds[feat])
            for feat in feature_cols
            if pd.notna(global_stds[feat])
        }
        representative = [feat for feat, _ in sorted(zscores.items(), key=lambda item: item[1], reverse=True)[:10]]

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


def _evaluation_split(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.Series, str]:
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
        pass


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "artifacts_ready": all(path.exists() for path in [BEST_MODEL_PATH, LABEL_ENCODER_PATH, TRAINED_FEATURE_COLUMNS_PATH]),
        "processed_dataset_ready": _resolve_processed_dataset_path() is not None,
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
    feature_columns = _ARTIFACTS["feature_columns"]

    return {
        "model_name": _model_name(model),
        "classes": list(label_encoder.classes_),
        "n_classes": int(len(label_encoder.classes_)),
        "n_features": int(len(feature_columns)),
        "feature_mode": infer_feature_mode(feature_columns),
        "metrics_summary": _best_model_metrics(),
    }


@app.get("/api/charts/genre-distribution")
def genre_distribution() -> dict[str, Any]:
    df = _get_processed_dataframe()
    if df is None:
        raise HTTPException(status_code=400, detail="Processed dataset not found.")

    if "split" in df.columns:
        train_df = df[df["split"] == "training"].copy()
        validation_count = int((df["split"] == "validation").sum())
        test_count = int((df["split"] == "test").sum())
    else:
        train_df = df.copy()
        validation_count = 0
        test_count = 0

    counts_series = train_df["genre_top"].value_counts().sort_values(ascending=False)
    counts = pd.DataFrame(
        {
            "genre": counts_series.index.astype(str),
            "count": counts_series.values.astype(int),
        }
    )

    return {
        "chart": "genre_distribution",
        "split_mode": "training_only" if "split" in df.columns else "full_dataset_fallback",
        "counts_summary": {
            "training": int(len(train_df)),
            "validation": validation_count,
            "test": test_count,
        },
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
        raise HTTPException(status_code=400, detail="Processed dataset not found.")

    model = _ARTIFACTS["model"]
    label_encoder = _ARTIFACTS["label_encoder"]
    feature_cols = [c for c in _get_feature_columns(df) if c in df.columns]
    if not feature_cols:
        raise HTTPException(status_code=400, detail="No usable feature columns for confusion matrix.")

    X_eval, y_eval_text, split_mode = _evaluation_split(df, feature_cols)
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
        raise HTTPException(status_code=400, detail="Processed dataset not found.")

    feature_cols = [col for col in infer_tabular_feature_columns(df) if col in df.columns]
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


@app.get("/api/charts/tsne")
def tsne_projection(max_points: int = 1000, perplexity: float = 30.0) -> dict[str, Any]:
    df = _get_processed_dataframe()
    if df is None:
        raise HTTPException(status_code=400, detail="Processed dataset not found.")

    feature_cols = [col for col in infer_tabular_feature_columns(df) if col in df.columns]
    if not feature_cols:
        raise HTTPException(status_code=400, detail="Feature columns not found in processed dataset.")

    sample_df = df.copy()
    if len(sample_df) > max_points:
        sample_df = sample_df.sample(n=max_points, random_state=42)

    if len(sample_df) < 10:
        raise HTTPException(status_code=400, detail="Not enough points for t-SNE projection.")

    X = sample_df[feature_cols].values
    pca = PCA(n_components=min(50, X.shape[1]), random_state=42)
    X_reduced = pca.fit_transform(X)

    safe_perplexity = max(5.0, min(perplexity, float(len(sample_df) - 1)))
    tsne = TSNE(
        n_components=2,
        perplexity=safe_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=42,
    )
    embedding = tsne.fit_transform(X_reduced)

    payload = []
    for idx, (_, row) in enumerate(sample_df.iterrows()):
        payload.append(
            {
                "x": float(embedding[idx, 0]),
                "y": float(embedding[idx, 1]),
                "genre": str(row["genre_top"]),
            }
        )

    return {
        "chart": "tsne_2d",
        "n_points": int(len(payload)),
        "perplexity": float(safe_perplexity),
        "data": payload,
    }


@app.get("/api/charts/lda")
def lda_projection(max_points: int = 1200) -> dict[str, Any]:
    df = _get_processed_dataframe()
    if df is None:
        raise HTTPException(status_code=400, detail="Processed dataset not found.")

    feature_cols = [col for col in infer_tabular_feature_columns(df) if col in df.columns]
    if not feature_cols:
        raise HTTPException(status_code=400, detail="Feature columns not found in processed dataset.")

    sample_df = df.copy()
    if len(sample_df) > max_points:
        sample_df = sample_df.sample(n=max_points, random_state=42)

    X = sample_df[feature_cols].values
    y = sample_df["genre_top"].astype(str).values

    lda = LinearDiscriminantAnalysis(n_components=2)
    embedding = lda.fit_transform(X, y)

    payload = []
    for idx, (_, row) in enumerate(sample_df.iterrows()):
        payload.append(
            {
                "x": float(embedding[idx, 0]),
                "y": float(embedding[idx, 1]),
                "genre": str(row["genre_top"]),
            }
        )

    return {
        "chart": "lda_2d",
        "n_points": int(len(payload)),
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

    model = _ARTIFACTS["model"]
    label_encoder = _ARTIFACTS["label_encoder"]
    feature_columns = _ARTIFACTS["feature_columns"]

    try:
        feature_dict = extract_features_for_model(tmp_path, feature_columns)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Feature extraction failed: {exc}") from exc
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass

    X = pd.DataFrame([feature_dict])[feature_columns]
    pred_encoded = int(model.predict(X)[0])
    predicted_genre = str(label_encoder.inverse_transform([pred_encoded])[0])

    probabilities = _serialize_probabilities(model, X, label_encoder)

    global_df = _get_processed_dataframe()
    mode = infer_feature_mode(feature_columns)
    notes = [
        f"Predição usando extrator: {mode}.",
        "O vetor de features do upload foi alinhado com o esquema salvo no treino.",
    ]

    if global_df is None:
        extracted_for_return = {k: float(v) for k, v in list(feature_dict.items())[:12]}
    else:
        usable = [c for c in _get_feature_columns(global_df) if c in global_df.columns and c in feature_dict]
        means = global_df[usable].mean(numeric_only=True)
        stds = global_df[usable].std(numeric_only=True).replace(0, np.nan)
        zscores = {}
        for feat in usable:
            if pd.notna(stds[feat]):
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
