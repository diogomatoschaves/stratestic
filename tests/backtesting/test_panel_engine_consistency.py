import pandas as pd
import pytest

from stratestic.backtesting import VectorizedBacktester, IterativeBacktester
from stratestic.backtesting.helpers.evaluation import CUM_SUM_STRATEGY, CUM_SUM_STRATEGY_TC
from tests.setup.panel_fixtures import PresetSidesPanel, make_random_panel
from tests.setup.fixtures.external_modules import mocked_plotly_figure_show

# staggered entries/exits, simultaneous flips, and bars where one symbol's
# exit funds another symbol's entry
PATTERNS = {
    "BTCUSDT": [0, 1, 1, 0, -1, -1, 0, 1, -1, 0],
    "ETHUSDT": [1, 1, 0, 0, 1, -1, -1, 0, 0, 1],
    "SOLUSDT": [0, 0, -1, -1, 0, 1, 1, 1, 0, -1],
}


class TestPanelEngineConsistency:
    """The two engines must produce exactly equal results on panels - the
    same invariant that protects the single-symbol engines."""

    @pytest.mark.parametrize("weights", [None, {"BTCUSDT": 0.5, "ETHUSDT": 0.3, "SOLUSDT": 0.2}])
    @pytest.mark.parametrize("trading_costs", [0, 0.1])
    def test_engines_agree(self, trading_costs, weights, mocked_plotly_figure_show):
        panel = make_random_panel()

        strategy = PresetSidesPanel(PATTERNS, weights=weights, data=panel)

        vect = VectorizedBacktester(strategy, trading_costs=trading_costs)
        ite = IterativeBacktester(strategy, trading_costs=trading_costs)

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

        assert all(trade.symbol in PATTERNS for trade in vect.trades)
