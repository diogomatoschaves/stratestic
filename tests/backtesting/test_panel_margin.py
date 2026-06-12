import numpy as np
import pandas as pd
import pytest

from stratestic.backtesting import VectorizedBacktester, IterativeBacktester
from stratestic.backtesting.helpers.evaluation import CUM_SUM_STRATEGY_TC, MARGIN_RATIO
from stratestic.utils.panel import build_panel
from tests.setup.panel_fixtures import PresetSidesPanel, make_flat_ohlc, make_random_panel
from tests.setup.fixtures.external_modules import mocked_plotly_figure_show

PATTERNS = {
    "BTCUSDT": [0, 1, 1, 0, -1, -1, 0, 1, -1, 0],
    "ETHUSDT": [1, 1, 0, 0, 1, -1, -1, 0, 0, 1],
    "SOLUSDT": [0, 0, -1, -1, 0, 1, 1, 1, 0, -1],
}


class TestPanelLeverage:

    @pytest.mark.parametrize("leverage", [2, 10])
    @pytest.mark.parametrize("trading_costs", [0, 0.1])
    def test_engines_agree_with_leverage(self, leverage, trading_costs, mocked_plotly_figure_show):
        panel = make_random_panel()

        strategy = PresetSidesPanel(PATTERNS, data=panel)

        vect = VectorizedBacktester(strategy, trading_costs=trading_costs, leverage=leverage)
        ite = IterativeBacktester(strategy, trading_costs=trading_costs, leverage=leverage)

        vect.run(print_results=False, plot_results=False)
        ite.run(print_results=False, plot_results=False)

        pd.testing.assert_series_equal(vect.results, ite.results)
        pd.testing.assert_series_equal(
            vect.processed_data[CUM_SUM_STRATEGY_TC], ite.processed_data[CUM_SUM_STRATEGY_TC]
        )
        pd.testing.assert_series_equal(
            vect.processed_data[MARGIN_RATIO], ite.processed_data[MARGIN_RATIO]
        )
        pd.testing.assert_frame_equal(pd.DataFrame(vect.trades), pd.DataFrame(ite.trades))

    def test_leverage_amplifies_pnl(self, mocked_plotly_figure_show):
        """Long-only +10% move at 2x leverage with no costs: +20%."""
        panel = build_panel({
            "BTCUSDT": make_flat_ohlc([100, 100, 100, 110, 110]),
            "ETHUSDT": make_flat_ohlc([100, 100, 100, 110, 110]),
        })
        patterns = {"BTCUSDT": [0, 1, 1, 1, 1], "ETHUSDT": [0, 1, 1, 1, 1]}

        strategy = PresetSidesPanel(patterns, data=panel)
        backtester = VectorizedBacktester(strategy, leverage=2)
        backtester.run(print_results=False, plot_results=False)

        assert backtester.results["return_pct"] == pytest.approx(20.0, abs=1e-9)

        for trade in backtester.trades:
            assert trade.liquidation_price is not None
            assert trade.maintenance_rate is not None


class TestPanelLiquidation:

    def make_liquidation_scenario(self):
        """BTC long at 10x gets liquidated by a -30% crash (margin ratio
        crosses 1 well before -30% at 10x); ETH holds flat."""
        panel = build_panel({
            "BTCUSDT": make_flat_ohlc([100, 100, 100, 100, 70, 70, 70]),
            "ETHUSDT": make_flat_ohlc([100, 100, 100, 100, 100, 100, 100]),
        })
        patterns = {"BTCUSDT": [0, 1, 1, 1, 1, 1, 1], "ETHUSDT": [0, 1, 1, 1, 1, 1, 1]}

        return PresetSidesPanel(patterns, data=panel)

    @pytest.mark.parametrize("backtester_class", [VectorizedBacktester, IterativeBacktester])
    def test_isolated_liquidation_caps_position_loss(self, backtester_class, mocked_plotly_figure_show):
        strategy = self.make_liquidation_scenario()

        backtester = backtester_class(strategy, leverage=10)
        backtester.run(print_results=False, plot_results=False)

        trades = {trade.symbol: trade for trade in backtester.trades}

        # the BTC position is liquidated: it forfeits exactly its isolated
        # margin (its allocation), not the full 10x crash loss
        assert trades["BTCUSDT"].pnl == pytest.approx(-1.0, abs=1e-9)
        assert trades["BTCUSDT"].profit == pytest.approx(-trades["BTCUSDT"].amount / 10, rel=1e-9)

        # the ETH position is untouched and the portfolio survives: the BTC
        # margin was 5000/10 = 500 (half the equity) -> -50%, NOT the -150%
        # the unbounded 10x crash loss would have implied
        assert trades["ETHUSDT"].pnl == pytest.approx(0.0, abs=1e-9)
        assert backtester.results["return_pct"] == pytest.approx(-50.0, abs=1e-9)

        # the margin ratio column records the breach (clipped at 1)
        assert backtester.processed_data[MARGIN_RATIO].max() == pytest.approx(1.0)

    def test_engines_agree_on_liquidations(self, mocked_plotly_figure_show):
        strategy = self.make_liquidation_scenario()

        vect = VectorizedBacktester(strategy, leverage=10)
        ite = IterativeBacktester(strategy, leverage=10)

        vect.run(print_results=False, plot_results=False)
        ite.run(print_results=False, plot_results=False)

        pd.testing.assert_series_equal(vect.results, ite.results)
        pd.testing.assert_frame_equal(pd.DataFrame(vect.trades), pd.DataFrame(ite.trades))

    def test_maximum_leverage_works_on_panels(self, mocked_plotly_figure_show):
        strategy = PresetSidesPanel(PATTERNS, data=make_random_panel())

        backtester = VectorizedBacktester(strategy)
        max_leverage = backtester.maximum_leverage(margin_threshold=0.8)

        assert isinstance(max_leverage, (int, np.integer))
        assert 1 <= max_leverage <= 125
