from collections import OrderedDict

import pandas as pd
import pytest

from stratestic.backtesting import VectorizedBacktester, IterativeBacktester
from stratestic.backtesting.combining import StrategyCombiner
from stratestic.backtesting.helpers.evaluation import CUM_SUM_STRATEGY, CUM_SUM_STRATEGY_TC, SIDE
from stratestic.strategies import Momentum, MovingAverage
from stratestic.strategies._mixin import StrategyMixin
from tests.setup.test_data.sample_data import data
from tests.setup.fixtures.external_modules import mocked_plotly_figure_show

# Exercises flat -> entry, entry -> flat and flip transitions, which the
# strategy-based fixtures don't cover systematically.
SIDE_PATTERN = [0, 1, 1, 0, -1, -1, 0, 1, -1, 0]


class PresetSides(StrategyMixin):
    """Test strategy emitting a fixed, repeating side pattern."""

    def __init__(self, pattern=None, data=None, **kwargs):
        self.params = OrderedDict()
        self._pattern = pattern if pattern is not None else SIDE_PATTERN
        StrategyMixin.__init__(self, data, **kwargs)

    def update_data(self, data):
        data = super().update_data(data)
        self._sides = pd.Series(
            [self._pattern[i % len(self._pattern)] for i in range(len(data))],
            index=data.index,
        )
        return data

    def calculate_positions(self, data):
        data[SIDE] = self._sides.reindex(data.index).values
        return data

    def get_signal(self, row=None):
        return int(self._sides.loc[row.name])


def make_flat_ohlc(prices):
    """OHLCV data where every bar's open/high/low/close equal the given price."""
    index = pd.date_range("2023-01-01", periods=len(prices), freq="1h", tz="UTC")
    prices = pd.Series(prices, index=index, dtype="float64")
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": 1.0,
        }
    )


class TestEngineConsistency:

    @pytest.mark.parametrize("short_model", ["inverse", "static"])
    @pytest.mark.parametrize("trading_costs", [0, 0.1])
    @pytest.mark.parametrize("leverage", [1, 10])
    def test_engines_agree_on_flat_transitions_with_costs(
        self, leverage, trading_costs, short_model, mocked_plotly_figure_show
    ):
        """Regression: the engines' notional-rebalancing previously diverged on
        flat -> entry transitions when combining costs and leverage."""
        test_data = data.set_index("open_time")

        strategy = PresetSides(data=test_data)

        kwargs = dict(
            symbol="BTCUSDT",
            trading_costs=trading_costs,
            leverage=leverage,
            short_model=short_model,
        )

        vect = VectorizedBacktester(strategy, **kwargs)
        ite = IterativeBacktester(strategy, **kwargs)

        vect.run(print_results=False, plot_results=False)
        ite.run(print_results=False, plot_results=False)

        pd.testing.assert_series_equal(vect.results, ite.results)
        pd.testing.assert_series_equal(
            vect.processed_data[CUM_SUM_STRATEGY], ite.processed_data[CUM_SUM_STRATEGY]
        )
        pd.testing.assert_series_equal(
            vect.processed_data[CUM_SUM_STRATEGY_TC], ite.processed_data[CUM_SUM_STRATEGY_TC]
        )
        pd.testing.assert_frame_equal(pd.DataFrame(vect.trades), pd.DataFrame(ite.trades))


class TestShortModelSemantics:
    """Documented behavior of the two short models (see BacktestMixin docs).

    A short entered at 100:
    - price falls to 50: static +50%, inverse +100% (entry/exit - 1)
    - price doubles to 200: static -100% (wiped out), inverse -50%
    Longs are identical under both models.
    """

    @pytest.mark.parametrize(
        "prices,expected",
        [
            pytest.param([100, 100, 100, 50, 50], {"static": 50.0, "inverse": 100.0}, id="falling"),
            pytest.param([100, 100, 100, 200, 200], {"static": -100.0, "inverse": -50.0}, id="rising"),
        ],
    )
    @pytest.mark.parametrize("short_model", ["static", "inverse"])
    @pytest.mark.parametrize("backtester_class", [VectorizedBacktester, IterativeBacktester])
    def test_single_short_trade(
        self, backtester_class, short_model, prices, expected, mocked_plotly_figure_show
    ):
        # first bar is dropped (NaN return); the short is entered at the close
        # of the second 100-bar and force-closed on the last bar
        test_data = make_flat_ohlc(prices)

        strategy = PresetSides(pattern=[0, -1, -1, -1], data=test_data)

        backtester = backtester_class(strategy, short_model=short_model)
        backtester.run(print_results=False, plot_results=False)

        assert backtester.results["return_pct"] == pytest.approx(expected[short_model], abs=1e-9)

    def test_long_identical_under_both_models(self, mocked_plotly_figure_show):
        test_data = make_flat_ohlc([100, 100, 100, 150, 150])

        results = {}
        for short_model in ("static", "inverse"):
            strategy = PresetSides(pattern=[0, 1, 1, 1], data=test_data)
            backtester = VectorizedBacktester(strategy, short_model=short_model)
            backtester.run(print_results=False, plot_results=False)
            results[short_model] = backtester.results["return_pct"]

        assert results["static"] == pytest.approx(results["inverse"], rel=1e-12)
        assert results["static"] == pytest.approx(50.0, abs=1e-9)


def test_backtester_getattr_raises_attribute_error():
    """Regression: missing attributes used to cause infinite recursion."""
    vect = VectorizedBacktester(Momentum(2))

    with pytest.raises(AttributeError):
        vect.nonexistent_attribute

    # hasattr relies on AttributeError being raised
    assert not hasattr(vect, "nonexistent_attribute")


def test_combiner_ignores_user_columns_containing_side(mocked_plotly_figure_show):
    """Regression: substring matching used to pick up user columns like 'upside'."""
    test_data = data.set_index("open_time")

    combiner = StrategyCombiner(
        [Momentum(2), MovingAverage(2)], method="Majority", data=test_data
    )

    expected = combiner.calculate_positions(combiner.data.copy())[SIDE]

    polluted = combiner.data.copy()
    polluted["upside"] = 5

    result = combiner.calculate_positions(polluted)[SIDE]

    pd.testing.assert_series_equal(result, expected)
