from __future__ import annotations

from pathlib import Path
import ast
import logging
from typing import Optional

import pandas as pd

try:
    from .config import TRACKS_CSV, FEATURES_CSV
except ImportError:  # pragma: no cover
    from config import TRACKS_CSV, FEATURES_CSV

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _literal_eval_or_none(value):
    if pd.isna(value):
        return value
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def load_tracks_metadata(tracks_csv: Path = TRACKS_CSV) -> pd.DataFrame:
    """
    Load FMA tracks metadata (multi-index columns).
    Expected format from official FMA metadata package.
    """
    if not tracks_csv.exists():
        raise FileNotFoundError(f"tracks.csv not found at: {tracks_csv}")

    tracks = pd.read_csv(tracks_csv, header=[0, 1], index_col=0)

    # Known columns with list-like values in FMA metadata.
    list_columns = [
        ("track", "tags"),
        ("album", "tags"),
        ("artist", "tags"),
        ("track", "genres"),
        ("track", "genres_all"),
    ]
    for col in list_columns:
        if col in tracks.columns:
            tracks[col] = tracks[col].map(_literal_eval_or_none)

    # Basic type handling used in official examples.
    if ("set", "subset") in tracks.columns:
        tracks[("set", "subset")] = pd.Categorical(
            tracks[("set", "subset")],
            categories=["small", "medium", "large"],
            ordered=True,
        )

    return tracks


def load_precomputed_features(features_csv: Path = FEATURES_CSV) -> pd.DataFrame:
    """Load FMA precomputed audio features table (multi-index columns)."""
    if not features_csv.exists():
        raise FileNotFoundError(f"features.csv not found at: {features_csv}")

    return pd.read_csv(features_csv, header=[0, 1, 2], index_col=0)


def filter_small_subset_with_known_genre(tracks: pd.DataFrame) -> pd.DataFrame:
    """Filter rows to fma_small + known top-level genre."""
    mask = pd.Series(True, index=tracks.index)

    if ("set", "subset") in tracks.columns:
        mask &= tracks[("set", "subset")] <= "small"

    if ("track", "genre_top") in tracks.columns:
        mask &= tracks[("track", "genre_top")].notna()
    else:
        raise KeyError("Expected column ('track', 'genre_top') in tracks.csv")

    filtered = tracks.loc[mask].copy()
    logger.info("Filtered dataset size: %s", filtered.shape)
    return filtered


def build_path_a_dataframe(
    tracks: pd.DataFrame,
    precomputed_features: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Build tabular dataset for Path A (using features.csv + tracks.csv).
    """
    filtered_tracks = filter_small_subset_with_known_genre(tracks)

    common_ids = filtered_tracks.index.intersection(precomputed_features.index)
    merged = precomputed_features.loc[common_ids].copy()

    # Flatten feature columns
    merged.columns = ["_".join([str(c) for c in col]).strip("_") for col in merged.columns]
    merged.insert(0, "track_id", common_ids)
    merged["genre_top"] = filtered_tracks.loc[common_ids, ("track", "genre_top")].astype(str).values

    if ("set", "split") in filtered_tracks.columns:
        merged["split"] = filtered_tracks.loc[common_ids, ("set", "split")].astype(str).values

    merged = merged.dropna(axis=0).reset_index(drop=True)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_path, index=False)
        logger.info("Saved Path A dataframe to %s", output_path)

    return merged
