import pandas as pd
import pytest

from stratestic.utils.panel import build_panel, is_panel, panel_symbols, validate_panel
from tests.setup.panel_fixtures import make_flat_ohlc


def test_build_panel_inner_join_aligns_indexes():
    long = make_flat_ohlc([100] * 10)
    short = make_flat_ohlc([50] * 6)

    panel = build_panel({"A": long, "B": short})

    assert is_panel(panel)
    assert panel_symbols(panel) == ["A", "B"]
    assert len(panel) == 6
    assert panel.columns.names == ("symbol", "field")


def test_build_panel_outer_join_keeps_union():
    long = make_flat_ohlc([100] * 10)
    short = make_flat_ohlc([50] * 6)

    panel = build_panel({"A": long, "B": short}, join="outer")

    assert len(panel) == 10
    assert panel[("B", "close")].isna().sum() == 4


def test_build_panel_warns_on_heavy_truncation(caplog):
    import logging

    long = make_flat_ohlc([100] * 100)
    short = make_flat_ohlc([50] * 10)

    with caplog.at_level(logging.WARNING):
        build_panel({"A": long, "B": short})

    assert any("dropped" in record.message for record in caplog.records)


def test_build_panel_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        build_panel({})


def test_validate_panel_rejects_flat_columns():
    with pytest.raises(ValueError, match="MultiIndex"):
        validate_panel(make_flat_ohlc([100] * 5))


def test_validate_panel_reports_missing_fields():
    panel = build_panel({"A": make_flat_ohlc([100] * 5)})
    panel = panel.drop(columns=[("A", "close")])

    with pytest.raises(ValueError, match="close"):
        validate_panel(panel)
