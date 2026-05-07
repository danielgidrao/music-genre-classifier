from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from .config import (
        AUDIO_DURATION_SECONDS,
        FMA_AUDIO_DIR,
        PROCESSED_FEATURES_CSV,
        SAMPLE_RATE,
        TRACKS_CSV,
        get_feature_columns,
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
    )
    from load_data import load_tracks_metadata, filter_small_subset_with_known_genre

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


FEATURE_COLUMNS = get_feature_columns()


def build_audio_path(track_id: int, audio_root: Path = FMA_AUDIO_DIR) -> Path:
    tid = f"{int(track_id):06d}"
    return audio_root / tid[:3] / f"{tid}.mp3"


def _safe_stats(array_2d_or_1d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if array_2d_or_1d.ndim == 1:
        return np.array([float(np.mean(array_2d_or_1d))]), np.array([float(np.std(array_2d_or_1d))])
    return np.mean(array_2d_or_1d, axis=1), np.std(array_2d_or_1d, axis=1)


def extract_features(
    audio_path: Path,
    sample_rate: int = SAMPLE_RATE,
    duration: Optional[float] = AUDIO_DURATION_SECONDS,
    n_mfcc: int = 20,
) -> dict[str, float]:
    """
    Extract a fixed-size feature vector from an audio file.

    Features:
    - MFCCs (20): mean and std -> 40
    - chroma_stft: mean and std
    - spectral_centroid: mean and std
    - spectral_rolloff: mean and std
    - zero_crossing_rate: mean and std
    - rms: mean and std
    - tempo (BPM)
    - spectral_bandwidth: mean and std
    """
    y, sr = librosa.load(audio_path, sr=sample_rate, mono=True, duration=duration)

    # Small guard to avoid failures on silent/corrupted decodes.
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

    chroma_mean = float(np.mean(chroma))
    chroma_std = float(np.std(chroma))
    centroid_mean = float(np.mean(centroid))
    centroid_std = float(np.std(centroid))
    rolloff_mean = float(np.mean(rolloff))
    rolloff_std = float(np.std(rolloff))
    zcr_mean = float(np.mean(zcr))
    zcr_std = float(np.std(zcr))
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))
    tempo_bpm = float(tempo[0]) if np.size(tempo) else 0.0
    bandwidth_mean = float(np.mean(bandwidth))
    bandwidth_std = float(np.std(bandwidth))

    values = []
    values.extend(mfcc_mean.tolist())
    values.extend(mfcc_std.tolist())
    values.extend(
        [
            chroma_mean,
            chroma_std,
            centroid_mean,
            centroid_std,
            rolloff_mean,
            rolloff_std,
            zcr_mean,
            zcr_std,
            rms_mean,
            rms_std,
            tempo_bpm,
            bandwidth_mean,
            bandwidth_std,
        ]
    )

    if len(values) != len(FEATURE_COLUMNS):
        raise RuntimeError(
            f"Feature length mismatch: got {len(values)}, expected {len(FEATURE_COLUMNS)}"
        )

    return dict(zip(FEATURE_COLUMNS, values))


def build_feature_dataset(
    tracks_csv: Path = TRACKS_CSV,
    audio_root: Path = FMA_AUDIO_DIR,
    output_csv: Path = PROCESSED_FEATURES_CSV,
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
            feature_dict = extract_features(audio_path)
            feature_dict["track_id"] = int(track_id)
            feature_dict["genre_top"] = str(row[("track", "genre_top")])
            if ("set", "split") in filtered.columns:
                feature_dict["split"] = str(row[("set", "split")])
            rows.append(feature_dict)
        except Exception as exc:
            skipped_corrupted += 1
            logger.warning("Skipping track %s (%s): %s", track_id, audio_path, exc)

    if not rows:
        raise RuntimeError("No features extracted. Check dataset paths and files.")

    df = pd.DataFrame(rows)

    # Ensure deterministic column order.
    ordered_cols = ["track_id"] + FEATURE_COLUMNS + ["genre_top"]
    if "split" in df.columns:
        ordered_cols.append("split")
    df = df[ordered_cols]

    df = df.dropna(axis=0).reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    logger.info("Saved processed dataset: %s", output_csv)
    logger.info("Total extracted: %d", len(df))
    logger.info("Skipped missing files: %d", skipped_missing)
    logger.info("Skipped corrupted files: %d", skipped_corrupted)

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract audio features from FMA small subset")
    parser.add_argument("--tracks-csv", type=Path, default=TRACKS_CSV)
    parser.add_argument("--audio-root", type=Path, default=FMA_AUDIO_DIR)
    parser.add_argument("--output", type=Path, default=PROCESSED_FEATURES_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_feature_dataset(args.tracks_csv, args.audio_root, args.output)


if __name__ == "__main__":
    main()
