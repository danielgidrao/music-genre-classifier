from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# FMA expected locations
FMA_AUDIO_DIR = RAW_DIR / "fma_small"
FMA_METADATA_DIR = RAW_DIR / "fma_metadata"
TRACKS_CSV = FMA_METADATA_DIR / "tracks.csv"
FEATURES_CSV = FMA_METADATA_DIR / "features.csv"

# Processed artifacts
PROCESSED_FEATURES_CSV = PROCESSED_DIR / "fma_features_clean.csv"
PATH_A_FEATURES_CSV = PROCESSED_DIR / "fma_precomputed_features.csv"
TRAIN_RESULTS_CSV = PROCESSED_DIR / "model_comparison.csv"
BEST_MODEL_RESULTS_JSON = PROCESSED_DIR / "best_model_metrics.json"

# Model artifacts
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
TRAINED_FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.pkl"

# Audio + feature extraction
SAMPLE_RATE = 22050
AUDIO_DURATION_SECONDS = 30
N_MFCC = 20


def get_feature_columns() -> list[str]:
    """Return feature names in a deterministic order."""
    columns = []

    # MFCC statistics
    for i in range(1, N_MFCC + 1):
        columns.append(f"mfcc_{i}_mean")
    for i in range(1, N_MFCC + 1):
        columns.append(f"mfcc_{i}_std")

    # Global descriptors (mean/std where applicable)
    columns.extend(
        [
            "chroma_mean",
            "chroma_std",
            "spectral_centroid_mean",
            "spectral_centroid_std",
            "spectral_rolloff_mean",
            "spectral_rolloff_std",
            "zero_crossing_rate_mean",
            "zero_crossing_rate_std",
            "rms_mean",
            "rms_std",
            "tempo_bpm",
            "spectral_bandwidth_mean",
            "spectral_bandwidth_std",
        ]
    )

    return columns
