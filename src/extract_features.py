from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

try:
    from .config import (
        AUDIO_DURATION_SECONDS,
        FMA_AUDIO_DIR,
        PROCESSED_FEATURES_CSV,
        SAMPLE_RATE,
        TRACKS_CSV,
        get_feature_columns,
        get_fma_compatible_feature_columns,
    )
    from .load_data import load_tracks_metadata, filter_small_subset_with_known_genre
except ImportError:  # pragma: no cover
    from config import (
        AUDIO_DURATION_SECONDS,
        FMA_AUDIO_DIR,
        PROCESSED_FEATURES_CSV,
        SAMPLE_RATE,
        TRACKS_CSV,
        get_feature_columns,
        get_fma_compatible_feature_columns,
    )
    from load_data import load_tracks_metadata, filter_small_subset_with_known_genre

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


FEATURE_COLUMNS_BASIC = get_feature_columns()
FEATURE_COLUMNS_FMA_COMPAT = get_fma_compatible_feature_columns()


def build_audio_path(track_id: int, audio_root: Path = FMA_AUDIO_DIR) -> Path:
    tid = f"{int(track_id):06d}"
    return audio_root / tid[:3] / f"{tid}.mp3"


def _safe_stats(array_2d_or_1d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if array_2d_or_1d.ndim == 1:
        return np.array([float(np.mean(array_2d_or_1d))]), np.array([float(np.std(array_2d_or_1d))])
    return np.mean(array_2d_or_1d, axis=1), np.std(array_2d_or_1d, axis=1)


def _feature_stats_to_flattened(name: str, values: np.ndarray) -> dict[str, float]:
    """Match flattened naming used by fma_metadata/features.csv."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)

    moments = {
        "mean": np.mean(values, axis=1),
        "std": np.std(values, axis=1),
        "skew": stats.skew(values, axis=1),
        "kurtosis": stats.kurtosis(values, axis=1),
        "median": np.median(values, axis=1),
        "min": np.min(values, axis=1),
        "max": np.max(values, axis=1),
    }

    out: dict[str, float] = {}
    size = values.shape[0]
    for moment_name, arr in moments.items():
        for i in range(size):
            out[f"{name}_{moment_name}_{i+1:02d}"] = float(arr[i])
    return out


def extract_features_basic(
    audio_path: Path,
    sample_rate: int = SAMPLE_RATE,
    duration: Optional[float] = AUDIO_DURATION_SECONDS,
    n_mfcc: int = 20,
) -> dict[str, float]:
    """
    Basic extractor used in the original Path B.
    """
    y, sr = librosa.load(audio_path, sr=sample_rate, mono=True, duration=duration)

    if y is None or len(y) == 0:
        raise ValueError("Empty audio signal")

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    tempo = librosa.feature.tempo(y=y, sr=sr)

    mfcc_mean, mfcc_std = _safe_stats(mfcc)

    values = []
    values.extend(mfcc_mean.tolist())
    values.extend(mfcc_std.tolist())
    values.extend(
        [
            float(np.mean(chroma)),
            float(np.std(chroma)),
            float(np.mean(centroid)),
            float(np.std(centroid)),
            float(np.mean(rolloff)),
            float(np.std(rolloff)),
            float(np.mean(zcr)),
            float(np.std(zcr)),
            float(np.mean(rms)),
            float(np.std(rms)),
            float(tempo[0]) if np.size(tempo) else 0.0,
            float(np.mean(bandwidth)),
            float(np.std(bandwidth)),
        ]
    )

    if len(values) != len(FEATURE_COLUMNS_BASIC):
        raise RuntimeError(
            f"Feature length mismatch: got {len(values)}, expected {len(FEATURE_COLUMNS_BASIC)}"
        )

    return dict(zip(FEATURE_COLUMNS_BASIC, values))


def extract_features_fma_compatible(audio_path: Path) -> dict[str, float]:
    """
    Extract features compatible with FMA precomputed features.csv schema.
    The implementation follows the public FMA extraction script (features.py).
    """
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    if y is None or len(y) == 0:
        raise ValueError("Empty audio signal")

    output: dict[str, float] = {}

    zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)
    output.update(_feature_stats_to_flattened("zcr", zcr))

    cqt = np.abs(
        librosa.cqt(
            y,
            sr=sr,
            hop_length=512,
            bins_per_octave=12,
            n_bins=7 * 12,
            tuning=None,
        )
    )

    chroma_cqt = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
    output.update(_feature_stats_to_flattened("chroma_cqt", chroma_cqt))

    chroma_cens = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
    output.update(_feature_stats_to_flattened("chroma_cens", chroma_cens))

    tonnetz = librosa.feature.tonnetz(chroma=chroma_cens)
    output.update(_feature_stats_to_flattened("tonnetz", tonnetz))

    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))

    chroma_stft = librosa.feature.chroma_stft(S=stft**2, n_chroma=12)
    output.update(_feature_stats_to_flattened("chroma_stft", chroma_stft))

    # rmse in the original script -> rms in modern librosa API
    rmse = librosa.feature.rms(S=stft)
    output.update(_feature_stats_to_flattened("rmse", rmse))

    centroid = librosa.feature.spectral_centroid(S=stft)
    output.update(_feature_stats_to_flattened("spectral_centroid", centroid))

    bandwidth = librosa.feature.spectral_bandwidth(S=stft)
    output.update(_feature_stats_to_flattened("spectral_bandwidth", bandwidth))

    contrast = librosa.feature.spectral_contrast(S=stft, n_bands=6)
    output.update(_feature_stats_to_flattened("spectral_contrast", contrast))

    rolloff = librosa.feature.spectral_rolloff(S=stft)
    output.update(_feature_stats_to_flattened("spectral_rolloff", rolloff))

    mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
    output.update(_feature_stats_to_flattened("mfcc", mfcc))

    missing = [c for c in FEATURE_COLUMNS_FMA_COMPAT if c not in output]
    if missing:
        raise RuntimeError(f"Missing FMA-compatible features: {missing[:5]}...")

    # Keep deterministic ordering for downstream consistency.
    return {col: output[col] for col in FEATURE_COLUMNS_FMA_COMPAT}


def extract_features(
    audio_path: Path,
    feature_mode: str = "basic",
    sample_rate: int = SAMPLE_RATE,
    duration: Optional[float] = AUDIO_DURATION_SECONDS,
    n_mfcc: int = 20,
) -> dict[str, float]:
    """Unified extractor entrypoint."""
    if feature_mode == "fma_compatible":
        return extract_features_fma_compatible(audio_path)
    if feature_mode == "basic":
        return extract_features_basic(
            audio_path=audio_path,
            sample_rate=sample_rate,
            duration=duration,
            n_mfcc=n_mfcc,
        )
    raise ValueError("feature_mode must be one of: basic, fma_compatible")


def build_feature_dataset(
    tracks_csv: Path = TRACKS_CSV,
    audio_root: Path = FMA_AUDIO_DIR,
    output_csv: Path = PROCESSED_FEATURES_CSV,
    feature_mode: str = "basic",
) -> pd.DataFrame:
    """Extract features for all valid fma_small tracks with known genre."""
    tracks = load_tracks_metadata(tracks_csv)
    filtered = filter_small_subset_with_known_genre(tracks)

    rows = []
    skipped_missing = 0
    skipped_corrupted = 0

    for track_id, row in tqdm(filtered.iterrows(), total=len(filtered), desc="Extracting features"):
        audio_path = build_audio_path(track_id, audio_root)
        if not audio_path.exists():
            skipped_missing += 1
            continue

        try:
            feature_dict = extract_features(audio_path, feature_mode=feature_mode)
            feature_dict["track_id"] = int(track_id)
            feature_dict["genre_top"] = str(row[("track", "genre_top")])
            if ("set", "split") in filtered.columns:
                feature_dict["split"] = str(row[("set", "split")])
            rows.append(feature_dict)
        except Exception as exc:
            skipped_corrupted += 1
            logger.warning("Skipping track %s (%s): %s", track_id, audio_path, exc)

    if not rows:
        raise RuntimeError(
            "No features extracted. Check dataset paths and files. "
            f"Skipped missing={skipped_missing}, corrupted={skipped_corrupted}"
        )

    df = pd.DataFrame(rows)

    ordered_features = FEATURE_COLUMNS_BASIC if feature_mode == "basic" else FEATURE_COLUMNS_FMA_COMPAT
    ordered_cols = ["track_id"] + ordered_features + ["genre_top"]
    if "split" in df.columns:
        ordered_cols.append("split")
    df = df[ordered_cols]

    df = df.dropna(axis=0).reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    logger.info("Saved processed dataset: %s", output_csv)
    logger.info("Feature mode: %s", feature_mode)
    logger.info("Total extracted: %d", len(df))
    logger.info("Skipped missing files: %d", skipped_missing)
    logger.info("Skipped corrupted files: %d", skipped_corrupted)

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract audio features from FMA small subset")
    parser.add_argument("--tracks-csv", type=Path, default=TRACKS_CSV)
    parser.add_argument("--audio-root", type=Path, default=FMA_AUDIO_DIR)
    parser.add_argument("--output", type=Path, default=PROCESSED_FEATURES_CSV)
    parser.add_argument(
        "--feature-mode",
        type=str,
        default="basic",
        choices=["basic", "fma_compatible"],
        help="basic: lightweight custom features | fma_compatible: schema matching features.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_feature_dataset(args.tracks_csv, args.audio_root, args.output, args.feature_mode)


if __name__ == "__main__":
    main()
