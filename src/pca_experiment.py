from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from .config import PATH_A_FEATURES_CSV, PROCESSED_DIR
    from .train import prepare_training_dataframe, split_data
    from .load_data import infer_tabular_feature_columns
except ImportError:  # pragma: no cover
    from config import PATH_A_FEATURES_CSV, PROCESSED_DIR
    from train import prepare_training_dataframe, split_data
    from load_data import infer_tabular_feature_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42

BASELINE_PARAMS = {
    "Decision Tree": {
        "criterion": "entropy",
        "max_depth": None,
        "min_samples_leaf": 1,
        "min_samples_split": 10,
    },
    "KNN": {
        "metric": "manhattan",
        "n_neighbors": 5,
        "weights": "distance",
    },
    "Random Forest": {
        "max_depth": None,
        "min_samples_leaf": 1,
        "min_samples_split": 5,
        "n_estimators": 100,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test PCA impact on FMA feature classification")
    parser.add_argument(
        "--training-source",
        type=str,
        default="metadata",
        choices=["metadata", "audio"],
        help="metadata: uses precomputed FMA features; audio: expects a prebuilt CSV from audio extraction.",
    )
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=PATH_A_FEATURES_CSV,
        help="Input tabular CSV used for the PCA experiment.",
    )
    parser.add_argument(
        "--pca-components",
        nargs="+",
        default=["0.90", "0.95", "0.99", "120", "200"],
        help="List of PCA component targets. Floats in (0,1) are treated as explained variance ratios.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROCESSED_DIR / "pca_experiment_results.csv",
        help="Path to save experiment results.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROCESSED_DIR / "pca_experiment_summary.json",
        help="Path to save experiment summary.",
    )
    return parser.parse_args()


def parse_pca_targets(raw_targets: list[str]) -> list[int | float]:
    parsed: list[int | float] = []
    for raw in raw_targets:
        value = float(raw)
        if 0 < value < 1:
            parsed.append(value)
        else:
            parsed.append(int(value))
    return parsed


def build_model(name: str, params: dict[str, Any]):
    if name == "Decision Tree":
        return DecisionTreeClassifier(random_state=RANDOM_STATE, **params)
    if name == "KNN":
        return KNeighborsClassifier(**params)
    if name == "Random Forest":
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **params)
    raise ValueError(f"Unsupported model: {name}")


def build_pipeline(name: str, params: dict[str, Any], pca_components: int | float | None = None) -> Pipeline:
    steps: list[tuple[str, Any]] = []

    if name == "KNN" or pca_components is not None:
        steps.append(("scale", StandardScaler()))

    if pca_components is not None:
        steps.append(("pca", PCA(n_components=pca_components, random_state=RANDOM_STATE)))

    steps.append(("model", build_model(name, params)))
    return Pipeline(steps=steps)


def compute_metrics(model: Pipeline, X: pd.DataFrame, y) -> dict[str, float]:
    y_pred = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision_macro": float(precision_score(y, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, y_pred, average="macro", zero_division=0)),
    }


def stringify_pca_target(target: int | float | None) -> str:
    if target is None:
        return "none"
    if isinstance(target, float):
        return f"{target:.2f}"
    return str(target)


def main() -> None:
    args = parse_args()
    pca_targets = parse_pca_targets(args.pca_components)

    df, source_used = prepare_training_dataframe(args.training_source, args.features_csv)
    feature_cols = infer_tabular_feature_columns(df)
    X_train, y_train, X_val, y_val, X_test, y_test, _, split_name = split_data(df, feature_cols)

    logger.info(
        "Running PCA experiment with split=%s, source=%s, features=%d",
        split_name,
        source_used,
        len(feature_cols),
    )

    X_trainval = pd.concat([X_train, X_val], axis=0)
    y_trainval = list(y_train) + list(y_val)

    results: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "split_strategy": split_name,
        "training_source": args.training_source,
        "feature_table": source_used,
        "n_features": len(feature_cols),
        "models": {},
    }

    for model_name, params in BASELINE_PARAMS.items():
        logger.info("Testing baseline %s", model_name)
        baseline_pipeline = build_pipeline(model_name, params)
        baseline_pipeline.fit(X_trainval, y_trainval)
        baseline_metrics = compute_metrics(baseline_pipeline, X_test, y_test)

        results.append(
            {
                "model": model_name,
                "variant": "baseline",
                "pca_components": "none",
                **baseline_metrics,
            }
        )

        best_val_result: dict[str, Any] | None = None
        logger.info("Testing PCA variants for %s", model_name)
        for target in pca_targets:
            candidate = build_pipeline(model_name, params, pca_components=target)
            candidate.fit(X_train, y_train)
            val_metrics = compute_metrics(candidate, X_val, y_val)

            current = {
                "model": model_name,
                "variant": "pca_validation",
                "pca_components": stringify_pca_target(target),
                **val_metrics,
            }
            results.append(current)

            if best_val_result is None or current["f1_macro"] > best_val_result["f1_macro"]:
                best_val_result = current

        assert best_val_result is not None
        best_target_raw = best_val_result["pca_components"]
        best_target = float(best_target_raw) if "." in best_target_raw else int(best_target_raw)

        logger.info("Best PCA target for %s based on validation: %s", model_name, best_target_raw)
        best_pca_pipeline = build_pipeline(model_name, params, pca_components=best_target)
        best_pca_pipeline.fit(X_trainval, y_trainval)
        best_pca_metrics = compute_metrics(best_pca_pipeline, X_test, y_test)

        results.append(
            {
                "model": model_name,
                "variant": "pca_test",
                "pca_components": best_target_raw,
                **best_pca_metrics,
            }
        )

        summary["models"][model_name] = {
            "baseline_test": baseline_metrics,
            "best_pca_components": best_target_raw,
            "best_pca_validation": {
                "accuracy": best_val_result["accuracy"],
                "precision_macro": best_val_result["precision_macro"],
                "recall_macro": best_val_result["recall_macro"],
                "f1_macro": best_val_result["f1_macro"],
            },
            "best_pca_test": best_pca_metrics,
            "accuracy_delta": float(best_pca_metrics["accuracy"] - baseline_metrics["accuracy"]),
            "f1_delta": float(best_pca_metrics["f1_macro"] - baseline_metrics["f1_macro"]),
        }

    results_df = pd.DataFrame(results)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output_csv, index=False)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("Saved PCA experiment results to %s", args.output_csv)
    logger.info("Saved PCA experiment summary to %s", args.output_json)


if __name__ == "__main__":
    main()
