from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

try:
    from .config import (
        BEST_MODEL_PATH,
        LABEL_ENCODER_PATH,
        PATH_A_FEATURES_CSV,
        PROCESSED_FEATURES_CSV,
        TRAIN_RESULTS_CSV,
        TRAINED_FEATURE_COLUMNS_PATH,
    )
    from .load_data import infer_tabular_feature_columns
except ImportError:  # pragma: no cover
    from config import (
        BEST_MODEL_PATH,
        LABEL_ENCODER_PATH,
        PATH_A_FEATURES_CSV,
        PROCESSED_FEATURES_CSV,
        TRAIN_RESULTS_CSV,
        TRAINED_FEATURE_COLUMNS_PATH,
    )
    from load_data import infer_tabular_feature_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def split_for_evaluation(df: pd.DataFrame, feature_cols: list[str]):
    X = df[feature_cols].copy()
    y_text = df["genre_top"].astype(str).copy()

    if "split" in df.columns and set(df["split"].dropna().unique()) >= {"test"}:
        test_mask = df["split"] == "test"
        return X.loc[test_mask], y_text.loc[test_mask], "official_test_split"

    return X, y_text, "full_dataset_fallback"


def resolve_features_csv(path_arg: Path | None) -> Path:
    if path_arg and path_arg.exists():
        return path_arg
    if PATH_A_FEATURES_CSV.exists():
        return PATH_A_FEATURES_CSV
    return PROCESSED_FEATURES_CSV


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate best trained model")
    parser.add_argument("--features-csv", type=Path, default=PATH_A_FEATURES_CSV)
    parser.add_argument("--save-dir", type=Path, default=Path("data/processed"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    features_csv = resolve_features_csv(args.features_csv)
    if not features_csv.exists():
        raise FileNotFoundError(
            f"Processed file not found: {features_csv}. Build it first (metadata path or audio path)."
        )
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {BEST_MODEL_PATH}")
    if not LABEL_ENCODER_PATH.exists():
        raise FileNotFoundError(f"Label encoder not found: {LABEL_ENCODER_PATH}")

    df = pd.read_csv(features_csv)
    model = joblib.load(BEST_MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    if TRAINED_FEATURE_COLUMNS_PATH.exists():
        feature_cols = joblib.load(TRAINED_FEATURE_COLUMNS_PATH)
    else:
        feature_cols = infer_tabular_feature_columns(df)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected feature columns: {missing[:10]}")

    X_eval, y_eval_text, eval_mode = split_for_evaluation(df, feature_cols)
    y_eval = label_encoder.transform(y_eval_text)

    y_pred = model.predict(X_eval)

    metrics = {
        "accuracy": accuracy_score(y_eval, y_pred),
        "precision_macro": precision_score(y_eval, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_eval, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_eval, y_pred, average="macro", zero_division=0),
    }

    print("Evaluation mode:", eval_mode)
    print("Feature table:", features_csv)
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    report = classification_report(
        y_eval,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0,
    )
    print("\nClassification report:\n")
    print(report)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.save_dir / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(y_eval, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    ax.set_title("Confusion Matrix - Best Model")
    plt.tight_layout()
    cm_path = args.save_dir / "confusion_matrix_best_model.png"
    fig.savefig(cm_path, dpi=150)

    if TRAIN_RESULTS_CSV.exists():
        results_df = pd.read_csv(TRAIN_RESULTS_CSV)
        metric_cols = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
        fig2, ax2 = plt.subplots(figsize=(10, 5))

        x = np.arange(len(results_df))
        width = 0.2
        for i, metric in enumerate(metric_cols):
            ax2.bar(x + (i - 1.5) * width, results_df[metric], width=width, label=metric)

        ax2.set_xticks(x)
        ax2.set_xticklabels(results_df["model"], rotation=0)
        ax2.set_ylim(0, 1)
        ax2.set_title("Model Comparison")
        ax2.legend()
        plt.tight_layout()
        comparison_path = args.save_dir / "model_comparison_metrics.png"
        fig2.savefig(comparison_path, dpi=150)
        logger.info("Saved model comparison chart: %s", comparison_path)

    rf_path = args.save_dir / "random_forest_feature_importance.png"
    rf_estimator = None

    if hasattr(model, "named_steps") and "model" in model.named_steps:
        if model.named_steps["model"].__class__.__name__ == "RandomForestClassifier":
            rf_estimator = model.named_steps["model"]
    elif model.__class__.__name__ == "RandomForestClassifier":
        rf_estimator = model

    if rf_estimator is not None:
        importances = rf_estimator.feature_importances_
        imp_df = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values(
            "importance", ascending=False
        )
        top_n = min(20, len(imp_df))
        fig3, ax3 = plt.subplots(figsize=(10, 7))
        ax3.barh(imp_df["feature"].head(top_n)[::-1], imp_df["importance"].head(top_n)[::-1])
        ax3.set_title("Random Forest Feature Importance (Top 20)")
        plt.tight_layout()
        fig3.savefig(rf_path, dpi=150)
        logger.info("Saved RF feature importance chart: %s", rf_path)

    metrics_path = args.save_dir / "best_model_metrics_eval.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    logger.info("Saved classification report: %s", report_path)
    logger.info("Saved confusion matrix: %s", cm_path)
    logger.info("Saved metrics JSON: %s", metrics_path)


if __name__ == "__main__":
    main()
