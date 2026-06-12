"""Helpers for building and validating multi-symbol (panel) DataFrames.

A panel is a single DataFrame whose columns are a 2-level MultiIndex of
``(symbol, field)``, e.g. ``("BTCUSDT", "close")``, sharing one datetime
index across all symbols.
"""
import logging
from typing import Literal

import pandas as pd

PANEL_COLUMN_NAMES = ("symbol", "field")


def build_panel(
    frames: dict,
    join: Literal["inner", "outer"] = "inner",
) -> pd.DataFrame:
    """
    Builds a panel DataFrame from per-symbol OHLCV DataFrames.

    Parameters
    ----------
    frames : dict[str, pd.DataFrame]
        Mapping of symbol to its OHLCV DataFrame (datetime index).
    join : "inner" or "outer", optional
        How to align the symbol indexes. "inner" (default) keeps only
        timestamps present for every symbol; "outer" keeps the union
        (note that the backtesters drop rows with missing values, so an
        outer join is reduced back to the intersection at backtest time).

    Returns
    -------
    pd.DataFrame
        Panel with (symbol, field) MultiIndex columns, sorted by index.
    """
    if not isinstance(frames, dict) or len(frames) == 0:
        raise ValueError("frames must be a non-empty dict of {symbol: DataFrame}")

    # duplicated timestamps break index alignment; keep the last occurrence,
    # like load_data does for single-symbol CSVs
    frames = {
        symbol: frame[~frame.index.duplicated(keep='last')]
        for symbol, frame in frames.items()
    }

    panel = pd.concat(frames, axis=1, join=join)
    panel.columns = panel.columns.set_names(PANEL_COLUMN_NAMES)
    panel = panel.sort_index()

    union_rows = max(len(frame) for frame in frames.values())
    if join == "inner" and union_rows > 0 and len(panel) < 0.75 * union_rows:
        logging.warning(
            "Aligning the symbol indexes dropped more than 25%% of rows "
            "(%d -> %d); check that the input frames cover the same period.",
            union_rows, len(panel),
        )

    return panel


def is_panel(data) -> bool:
    """True when `data` is a (symbol, field) MultiIndex-column panel."""
    return isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex)


def panel_symbols(data) -> list:
    """The symbols of a panel, in column order (the canonical engine order)."""
    return list(data.columns.get_level_values(0).unique())


def validate_panel(data, required_fields=("open", "high", "low", "close")) -> None:
    """
    Validates a panel DataFrame: 2-level columns and the required OHLC
    fields present for every symbol. Raises ValueError with the full list
    of missing (symbol, field) pairs.
    """
    if not is_panel(data):
        raise ValueError(
            "Multi-symbol data must be a DataFrame with (symbol, field) "
            "MultiIndex columns - use stratestic.utils.panel.build_panel."
        )

    if data.columns.nlevels != 2:
        raise ValueError(
            f"Panel columns must have exactly 2 levels (symbol, field); got {data.columns.nlevels}."
        )

    missing = [
        (symbol, field)
        for symbol in panel_symbols(data)
        for field in required_fields
        if (symbol, field) not in data.columns
    ]

    if missing:
        raise ValueError(f"Panel is missing required columns: {missing}")
