import os

import numpy as np
import pandas as pd
import pytest

from stratestic.backtesting import VectorizedBacktester, IterativeBacktester
from stratestic.backtesting.helpers.evaluation import CUM_SUM_STRATEGY, CUM_SUM_STRATEGY_TC, MARGIN_RATIO
from stratestic.strategies import MovingAverage
from stratestic.utils.exceptions import SymbolInvalid
from stratestic.utils.exceptions.leverage_invalid import LeverageInvalid
from tests.setup.test_data.sample_data import data
from tests.setup.test_setup import get_fixtures
from tests.setup.fixtures.external_modules import mocked_plotly_figure_show

current_path = os.path.dirname(os.path.realpath(__file__))

fixtures = get_fixtures(current_path, keys=["in_margin", "out_margin"])


class TestVectorizedBacktesterMargin:

    @pytest.mark.parametrize(
        "leverage",
        [
            pytest.param(1, id="leverage=1"),
            pytest.param(10, id="leverage=10"),
            pytest.param(100, id="leverage=100")
        ],
    )
    @pytest.mark.parametrize(
        "fixture",
        [
            pytest.param(fixture, id=fixture_name)
            for fixture_name, fixture in fixtures.items()
        ],
    )
    def test_run(self, leverage, fixture, mocked_plotly_figure_show):

        strategy = fixture["in_margin"]["strategy"]
        params = fixture["in_margin"]["params"]
        trading_costs = fixture["in_margin"]["trading_costs"]

        amount = 1000

        test_data = data.set_index("open_time")

        strategy_instance = strategy(**params, data=test_data)

        vect = VectorizedBacktester(
            strategy_instance,
            symbol="BTCUSDT",
            amount=amount,
            trading_costs=trading_costs,
            leverage=leverage,
        )

        vect.run()

        print(vect.processed_data.reset_index().to_dict(orient="records"))

        if len(vect.trades) > 0:
            assert (
                round(vect.trades[0].profit / vect.trades[0].pnl) == amount
            )

        if leverage > 1:
            for trade in vect.trades:
                assert trade.liquidation_price is not None

        for i, d in enumerate(vect.processed_data.to_dict(orient="records")):
            for key in d:
                result = fixture["out_margin"]["expected_results"][leverage][i][key]
                try:
                    assert np.isnan(d[key]) == np.isnan(result)
                except TypeError:
                    assert d[key] == pytest.approx(result, 0.2)

    @pytest.mark.parametrize(
        "leverage,symbol,second_leverage,exception",
        [
            pytest.param(-1, "BTCUSDT", 1, LeverageInvalid, id="invalid_leverage"),
            pytest.param(10, "Invalid Symbol", 1, SymbolInvalid, id="invalid_symbol"),
            pytest.param(
                1, "BTCUSDT", -1, LeverageInvalid, id="invalid_leverage_on_run"
            ),
        ],
    )
    def test_exceptions(
        self, leverage, symbol, second_leverage, exception, mocked_plotly_figure_show
    ):

        strategy = MovingAverage(10)

        with pytest.raises(exception):

            vect = VectorizedBacktester(
                strategy, symbol=symbol, leverage=leverage
            )

            vect.run(leverage=second_leverage)

    @pytest.mark.parametrize(
        "leverage",
        [pytest.param(1, id="leverage=1"), pytest.param(10, id="leverage=10")],
    )
    @pytest.mark.parametrize(
        "fixture",
        [
            pytest.param(fixture, id=fixture_name)
            for fixture_name, fixture in fixtures.items()
        ],
    )
    def test_results_equal_to_iterative(
        self, leverage, fixture, mocked_plotly_figure_show
    ):
        strategy = fixture["in_margin"]["strategy"]
        params = fixture["in_margin"]["params"]
        trading_costs = fixture["in_margin"]["trading_costs"]

        test_data = data.set_index("open_time")

        strategy_instance = strategy(**params, data=test_data)

        vect = VectorizedBacktester(
            strategy_instance,
            symbol="BTCUSDT",
            trading_costs=trading_costs,
            leverage=leverage,
        )

        ite = IterativeBacktester(
            strategy_instance,
            symbol="BTCUSDT",
            trading_costs=trading_costs,
            leverage=leverage,
        )

        vect.run()
        ite.run()

        trades_vect = pd.DataFrame(vect.trades)
        trades_ite = pd.DataFrame(ite.trades)

        pd.testing.assert_series_equal(vect.results, ite.results)
        pd.testing.assert_series_equal(
            vect.processed_data[CUM_SUM_STRATEGY], ite.processed_data[CUM_SUM_STRATEGY]
        )
        pd.testing.assert_series_equal(
            vect.processed_data[CUM_SUM_STRATEGY_TC], ite.processed_data[CUM_SUM_STRATEGY_TC]
        )
        if leverage != 1:
            pd.testing.assert_series_equal(
                vect.processed_data[MARGIN_RATIO], ite.processed_data[MARGIN_RATIO]
            )

        pd.testing.assert_frame_equal(trades_vect, trades_ite)

    @pytest.mark.parametrize(
        "margin_threshold, expected_result",
        [
            pytest.param(0.1, 24, id="margin_threshold=0.1"),
            pytest.param(0.5, 93, id="margin_threshold=0.5"),
            pytest.param(1, 124, id="margin_threshold=1")
        ],
    )
    def test_maximum_leverage(
        self, margin_threshold, expected_result, mocked_plotly_figure_show
    ):
        test_data = data.set_index("open_time")

        strategy_instance = MovingAverage(4, data=test_data)

        vect = VectorizedBacktester(
            strategy_instance,
            symbol="BTCUSDT",
            leverage=2,
        )

        result = vect.maximum_leverage(margin_threshold=margin_threshold)

        assert result == expected_result

    @pytest.mark.parametrize(
        "backtester",
        [
            pytest.param(VectorizedBacktester, id="VectorizedBacktester"),
            pytest.param(IterativeBacktester, id="IterativeBacktester")
        ],
    )
    @pytest.mark.parametrize(
        "margin_threshold",
        [
            pytest.param(0, id="margin_threshold==0"),
            pytest.param(1.2,  id="margin_threshold>1"),
            pytest.param('abc',  id="margin_threshold=='abc")
        ],
    )
    def test_invalid_margin_threshold(self, backtester, margin_threshold):
        strategy = MovingAverage(10, data=data.set_index("open_time"))

        with pytest.raises(ValueError):
            vect = backtester(
                strategy, symbol="BTCUSDT"
            )

            vect.maximum_leverage(margin_threshold=margin_threshold)