import pandas as pd
import pytest

from stratestic.backtesting import VectorizedBacktester, IterativeBacktester
from stratestic.backtesting.combining import StrategyCombiner
from stratestic.backtesting.helpers.evaluation import CUM_SUM_STRATEGY_TC, SIDE
from stratestic.strategies import Momentum, MovingAverage
from stratestic.strategies.multi import BroadcastStrategy
from stratestic.utils.panel import build_panel
from tests.setup.panel_fixtures import PresetSidesPanel, make_random_ohlc, make_random_panel
from tests.setup.fixtures.external_modules import mocked_plotly_figure_show


class TestPanelGuards:

    def test_inverse_short_model_rejected(self, mocked_plotly_figure_show):
        strategy = PresetSidesPanel({"BTCUSDT": [1], "ETHUSDT": [-1]}, data=make_random_panel(20))
        backtester = VectorizedBacktester(strategy, short_model="inverse")

        with pytest.raises(ValueError, match="static"):
            backtester.run(print_results=False, plot_results=False)

    def test_symbol_argument_rejected(self, mocked_plotly_figure_show):
        strategy = PresetSidesPanel({"BTCUSDT": [1], "ETHUSDT": [-1]}, data=make_random_panel(20))
        backtester = VectorizedBacktester(strategy, symbol="BTCUSDT")

        with pytest.raises(ValueError, match="symbol"):
            backtester.run(print_results=False, plot_results=False)

    def test_panel_with_unknown_symbol_raises(self, mocked_plotly_figure_show):
        from stratestic.utils.exceptions import SymbolInvalid
        from stratestic.utils.panel import build_panel
        from tests.setup.panel_fixtures import make_random_ohlc

        panel = build_panel({"BTCUSDT": make_random_ohlc(1, n=20), "FAKEUSDT": make_random_ohlc(2, n=20)})
        strategy = PresetSidesPanel({"BTCUSDT": [1], "FAKEUSDT": [-1]}, data=panel)
        backtester = VectorizedBacktester(strategy, leverage=2)

        with pytest.raises(SymbolInvalid, match="FAKEUSDT"):
            backtester.run(print_results=False, plot_results=False)

    def test_single_symbol_strategy_with_panel_data_rejected(self, mocked_plotly_figure_show):
        strategy = Momentum(3)
        strategy.data = make_random_panel(20)

        backtester = VectorizedBacktester(strategy)

        with pytest.raises(TypeError, match="BroadcastStrategy"):
            backtester.run(print_results=False, plot_results=False)

    def test_combiner_rejects_panel(self):
        with pytest.raises(NotImplementedError, match="panel"):
            StrategyCombiner([Momentum(2), MovingAverage(2)], method="Majority", data=make_random_panel(20))

    def test_broadcast_rejects_combiner(self):
        with pytest.raises(NotImplementedError, match="Combiner"):
            BroadcastStrategy(StrategyCombiner([Momentum(2)]))


class TestBroadcastStrategy:

    def test_sides_match_independent_runs(self):
        frames = {"BTCUSDT": make_random_ohlc(7), "ETHUSDT": make_random_ohlc(8)}
        panel = build_panel(frames)

        broadcast = BroadcastStrategy(MovingAverage(4), data=panel)
        data = broadcast.calculate_positions(broadcast.data.copy())

        for symbol, frame in frames.items():
            solo = MovingAverage(4, data=frame)
            solo_sides = solo.calculate_positions(solo.data)[SIDE]
            pd.testing.assert_series_equal(data[(symbol, SIDE)], solo_sides, check_names=False)

    def test_get_signal_returns_dict(self):
        broadcast = BroadcastStrategy(MovingAverage(4), data=make_random_panel(40))
        signals = broadcast.get_signal()

        assert set(signals) == set(broadcast.symbols)
        assert all(signal in (-1, 0, 1) for signal in signals.values())

    def test_set_parameters_propagates_to_all_clones(self):
        broadcast = BroadcastStrategy(MovingAverage(4), data=make_random_panel(40))
        broadcast.set_parameters({"ma": 7})

        assert broadcast._ma == 7
        assert all(clone._ma == 7 for clone in broadcast._strategies.values())

    def test_getattr_raises_attribute_error(self):
        broadcast = BroadcastStrategy(MovingAverage(4), data=make_random_panel(40))

        with pytest.raises(AttributeError):
            broadcast.nonexistent_attribute

        assert not hasattr(broadcast, "nonexistent_attribute")

    def test_optimize_smoke(self, mocked_plotly_figure_show):
        broadcast = BroadcastStrategy(MovingAverage(3), data=make_random_panel(60))
        backtester = VectorizedBacktester(broadcast)

        optimized_params, _ = backtester.optimize({"ma": (2, 4)})

        assert set(optimized_params) == {"ma"}
        assert 2 <= optimized_params["ma"] <= 4

    def test_single_symbol_panel_equals_scalar_path(self, mocked_plotly_figure_show):
        """A 1-symbol panel through the panel engine must reproduce the
        scalar path's results exactly - the degeneration anchor."""
        frame = make_random_ohlc(11)

        broadcast = BroadcastStrategy(MovingAverage(4), data=build_panel({"BTCUSDT": frame}))
        panel_backtester = VectorizedBacktester(broadcast, trading_costs=0.1)
        panel_backtester.run(print_results=False, plot_results=False)

        scalar = MovingAverage(4, data=frame)
        scalar_backtester = VectorizedBacktester(scalar, trading_costs=0.1)
        scalar_backtester.run(print_results=False, plot_results=False)

        pd.testing.assert_series_equal(
            panel_backtester.processed_data[CUM_SUM_STRATEGY_TC],
            scalar_backtester.processed_data[CUM_SUM_STRATEGY_TC],
        )

        panel_trades = pd.DataFrame(panel_backtester.trades)
        scalar_trades = pd.DataFrame(scalar_backtester.trades)

        for column in ("entry_date", "exit_date", "side", "pnl"):
            pd.testing.assert_series_equal(
                panel_trades[column], scalar_trades[column], check_names=False
            )
