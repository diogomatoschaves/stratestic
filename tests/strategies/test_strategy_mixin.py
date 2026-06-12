import numpy as np
import pandas as pd
import pytest

from stratestic.strategies import MovingAverageConvergenceDivergence
from stratestic.strategies._helpers import get_moving_average
from stratestic.strategies._mixin import StrategyMixin
from tests.setup.test_data.sample_data import data


def test_set_data_uses_provided_data():
    """Regression: the self-branch of set_data re-processed the existing
    data instead of the DataFrame passed in."""
    strategy = StrategyMixin(data=data[:10].set_index("open_time"))

    new_data = data[10:30].set_index("open_time")
    strategy.set_data(new_data)

    assert len(strategy.data) == len(new_data)
    assert strategy.data.index.equals(new_data.index)


def test_unimplemented_strategy_contract_raises():
    strategy = StrategyMixin()

    with pytest.raises(NotImplementedError, match="calculate_positions"):
        strategy.calculate_positions(pd.DataFrame())

    with pytest.raises(NotImplementedError, match="get_signal"):
        strategy.get_signal()


def test_get_params_without_params_attribute():
    assert StrategyMixin().get_params() == {}


def test_get_moving_average_dispatch():
    series = pd.Series(np.arange(10, dtype="float64"))

    sma = get_moving_average(series, "sma", 3)
    assert sma.iloc[-1] == pytest.approx(8.0)

    ema = get_moving_average(series, "ema", 3)
    assert np.isfinite(ema.iloc[-1])

    with pytest.raises(ValueError, match="not supported"):
        get_moving_average(series, "wma", 3)


def test_macd_signal_zero_on_tie():
    """Regression: get_signal returned None on a zero macd_diff while
    calculate_positions mapped it to -1; both now return 0 (neutral)."""
    macd = MovingAverageConvergenceDivergence(4, 2, 3, data=data.set_index("open_time"))

    row = macd.data.iloc[-1].copy()
    row["macd_diff"] = 0.0

    assert macd.get_signal(row) == 0

    df = macd.data.copy()
    df["macd_diff"] = 0.0
    assert (macd.calculate_positions(df)["side"] == 0).all()
