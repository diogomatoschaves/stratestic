import os

import pytest

from stratestic.backtesting import IterativeBacktester
from stratestic.strategies import MovingAverage
from stratestic.tests.setup.fixtures.external_modules import mocked_plotly_figure_show
from stratestic.tests.setup.test_data.sample_data import data
from stratestic.utils.exceptions import SymbolInvalid
from stratestic.utils.exceptions.leverage_invalid import LeverageInvalid
from stratestic.utils.test_setup import get_fixtures

current_path = os.path.dirname(os.path.realpath(__file__))

fixtures = get_fixtures(current_path, keys=["in_margin", "out_margin"])


class TestIterativeBacktesterMargin:

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "leverage",
        [
            pytest.param(
                1,
                id='leverage=1'
            ),
            pytest.param(
                10,
                id='leverage=10'
            )
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

        ite = IterativeBacktester(
            strategy_instance,
            symbol='BTCUSDT',
            amount=amount,
            trading_costs=trading_costs,
            include_margin=True,
            leverage=leverage
        )

        ite.run()

        print(ite.processed_data.to_dict(orient="records"))

        assert round(ite.trades[0].profit / ite.trades[0].pnl * leverage) == amount

        for trade in ite.trades:
            assert trade.liquidation_price is not None

        for i, d in enumerate(ite.processed_data.to_dict(orient="records")):
            for key in d:
                assert d[key] == pytest.approx(
                    fixture["out_margin"]["expected_results"][leverage][i][key], 0.2
                )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "leverage,symbol,second_leverage,exception",
        [
            pytest.param(
                -1,
                'BTCUSDT',
                1,
                LeverageInvalid,
                id='invalid_leverage'
            ),
            pytest.param(
                1,
                'Invalid Symbol',
                1,
                SymbolInvalid,
                id='invalid_symbol'
            ),
            pytest.param(
                1,
                'BTCUSDT',
                -1,
                LeverageInvalid,
                id='invalid_leverage_on_run'
            )
        ],
    )
    def test_exceptions(self, leverage, symbol, second_leverage, exception, mocked_plotly_figure_show):

        strategy = MovingAverage(10)

        with pytest.raises(exception):

            ite = IterativeBacktester(
                strategy,
                symbol=symbol,
                include_margin=True,
                leverage=leverage
            )

            ite.run(leverage=second_leverage)