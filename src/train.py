from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from .config import (
        BEST_MODEL_PATH,
        BEST_MODEL_RESULTS_JSON,
        LABEL_ENCODER_PATH,
        PROCESSED_FEATURES_CSV,
        SCALER_PATH,
        TRAINED_FEATURE_COLUMNS_PATH,
        TRAIN_RESULTS_CSV,
        get_feature_columns,
    )
except ImportError:  # pragma: no cover
    from config import (
        BEST_MODEL_PATH,
        BEST_MODEL_RESULTS_JSON,
        LABEL_ENCODER_PATH,
        PROCESSED_FEATURES_CSV,
        SCALER_PATH,
        TRAINED_FEATURE_COLUMNS_PATH,
        TRAIN_RESULTS_CSV,
        get_feature_columns,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


RANDOM_STATE = 42


def load_training_data(path: Path = PROCESSED_FEATURES_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Processed feature file not found: {path}")
    return pd.read_csv(path)


def split_data(df: pd.DataFrame):
    feature_cols = get_feature_columns()
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")

    X = df[feature_cols].copy()
    y_text = df["genre_top"].astype(str).copy()

    le = LabelEncoder()
    y = le.fit_transform(y_text)

    has_official_split = "split" in df.columns and set(df["split"].dropna().unique()) >= {
        "training",
        "validation",
        "test",
    }

    if has_official_split:
        train_mask = df["split"] == "training"
        val_mask = df["split"] == "validation"
        test_mask = df["split"] == "test"

        X_train = X.loc[train_mask]
        y_train = y[train_mask.values]
        X_val = X.loc[val_mask]
        y_val = y[val_mask.values]
        X_test = X.loc[test_mask]
        y_test = y[test_mask.values]

        split_name = "official_fma_split"
    else:
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y_train_full,
        )
        split_name = "stratified_split"

    return X_train, y_train, X_val, y_val, X_test, y_test, le, split_name


def build_predefined_cv(y_train: np.ndarray, y_val: np.ndarray) -> PredefinedSplit:
    test_fold = np.concatenate(
        [
            np.full(shape=(len(y_train),), fill_value=-1, dtype=int),
            np.zeros(shape=(len(y_val),), dtype=int),
        ]
    )
    return PredefinedSplit(test_fold)


def fit_model_with_search(name, estimator, param_grid, X_train, y_train, X_val, y_val):
    X_trainval = pd.concat([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val])
    cv = build_predefined_cv(y_train, y_val)

    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="f1_macro",
        n_jobs=-1,
        cv=cv,
        refit=True,
        verbose=1,
    )

    logger.info("Training %s with GridSearchCV...", name)
    grid.fit(X_trainval, y_trainval)
    logger.info("Best params for %s: %s", name, grid.best_params_)
    logger.info("Best validation f1_macro for %s: %.4f", name, grid.best_score_)

    return grid


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }


def build_models_and_grids(feature_cols: list[str]):
    preprocessor_scaled = ColumnTransformer(
        transformers=[("scale", StandardScaler(), feature_cols)],
        remainder="drop",
    )

    dt_pipeline = Pipeline(
        steps=[
            ("model", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]
    )
    dt_grid = {
        "model__criterion": ["gini", "entropy"],
        "model__max_depth": [None, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    }

    knn_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor_scaled),
            ("model", KNeighborsClassifier()),
        ]
    )
    knn_grid = {
        "model__n_neighbors": [3, 5, 7, 11, 15],
        "model__weights": ["uniform", "distance"],
        "model__metric": ["euclidean", "manhattan"],
    }

    rf_pipeline = Pipeline(
        steps=[
            ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
        ]
    )
    rf_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 20, 40],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    }

    return {
        "Decision Tree": (dt_pipeline, dt_grid),
        "KNN": (knn_pipeline, knn_grid),
        "Random Forest": (rf_pipeline, rf_grid),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train supervised models for FMA genre classification")
    parser.add_argument("--features-csv", type=Path, default=PROCESSED_FEATURES_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_training_data(args.features_csv)
    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder, split_name = split_data(df)
    feature_cols = get_feature_columns()

    logger.info(
        "Split strategy: %s | train=%d, val=%d, test=%d",
        split_name,
        len(X_train),
        len(X_val),
        len(X_test),
    )

    models_and_grids = build_models_and_grids(feature_cols)

    results = []
    trained_models = {}

    for name, (estimator, grid) in models_and_grids.items():
        search = fit_model_with_search(name, estimator, grid, X_train, y_train, X_val, y_val)
        metrics = evaluate_model(search.best_estimator_, X_test, y_test)

        row = {
            "model": name,
            "best_params": json.dumps(search.best_params_),
            **metrics,
        }
        results.append(row)
        trained_models[name] = search.best_estimator_

    results_df = pd.DataFrame(results).sort_values("f1_macro", ascending=False)

    best_row = results_df.iloc[0]
    best_model_name = best_row["model"]
    best_model = trained_models[best_model_name]

    # Save artifacts
    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, BEST_MODEL_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    joblib.dump(feature_cols, TRAINED_FEATURE_COLUMNS_PATH)

    # Save scaler separately as requested.
    scaler = StandardScaler().fit(X_train)
    joblib.dump(scaler, SCALER_PATH)

    TRAIN_RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(TRAIN_RESULTS_CSV, index=False)

    with open(BEST_MODEL_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_model": best_model_name,
                "split_strategy": split_name,
                "metrics": {
                    "accuracy": float(best_row["accuracy"]),
                    "precision_macro": float(best_row["precision_macro"]),
                    "recall_macro": float(best_row["recall_macro"]),
                    "f1_macro": float(best_row["f1_macro"]),
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info("Saved best model: %s", BEST_MODEL_PATH)
    logger.info("Saved scaler: %s", SCALER_PATH)
    logger.info("Saved label encoder: %s", LABEL_ENCODER_PATH)
    logger.info("Saved feature column schema: %s", TRAINED_FEATURE_COLUMNS_PATH)
    logger.info("Saved model comparison: %s", TRAIN_RESULTS_CSV)
    logger.info("Best model selected: %s", best_model_name)


if __name__ == "__main__":
    main()
