import numpy as np
import pandas as pd
import pytest

from stratestic.backtesting import VectorizedBacktester, IterativeBacktester
from stratestic.utils.panel import build_panel
from tests.setup.panel_fixtures import PresetSidesPanel, make_flat_ohlc
from tests.setup.fixtures.external_modules import mocked_plotly_figure_show


def pairs_panel():
    # first bar is dropped (NaN returns); positions enter at the close of
    # the second 100-bar and are force-closed on the final bar
    return build_panel({
        "BTCUSDT": make_flat_ohlc([100, 100, 100, 110, 110]),
        "ETHUSDT": make_flat_ohlc([100, 100, 100, 90, 90]),
    })


PAIRS_PATTERNS = {"BTCUSDT": [0, 1, 1, 1, 1], "ETHUSDT": [0, -1, -1, -1, -1]}


class TestPanelHandComputed:

    @pytest.mark.parametrize("backtester_class", [VectorizedBacktester, IterativeBacktester])
    def test_equal_weight_pairs_trade(self, backtester_class, mocked_plotly_figure_show):
        """Long BTC 100->110 and short ETH 100->90 at equal weight, no costs:
        each leg gains 10% on half the capital -> exactly +10% portfolio."""
        strategy = PresetSidesPanel(PAIRS_PATTERNS, data=pairs_panel())

        backtester = backtester_class(strategy)
        backtester.run(print_results=False, plot_results=False)

        assert backtester.results["return_pct"] == pytest.approx(10.0, abs=1e-9)
        assert backtester.results["nr_trades"] == 2

        trades = {trade.symbol: trade for trade in backtester.trades}
        assert trades["BTCUSDT"].pnl == pytest.approx(0.1, abs=1e-12)
        assert trades["ETHUSDT"].pnl == pytest.approx(0.1, abs=1e-12)

        # equal-weight unrebalanced basket: +10% and -10% legs cancel
        assert backtester.results["buy_and_hold_return"] == pytest.approx(0.0, abs=1e-9)

    def test_single_active_symbol_gets_full_capital(self, mocked_plotly_figure_show):
        """With one symbol flat, equal weight across *active* positions
        deploys the full capital into the single active symbol."""
        patterns = {"BTCUSDT": [0, 1, 1, 1, 1], "ETHUSDT": [0, 0, 0, 0, 0]}
        strategy = PresetSidesPanel(patterns, data=pairs_panel())

        backtester = VectorizedBacktester(strategy)
        backtester.run(print_results=False, plot_results=False)

        assert backtester.results["return_pct"] == pytest.approx(10.0, abs=1e-9)
        assert backtester.results["nr_trades"] == 1

    def test_all_flat_keeps_initial_equity(self, mocked_plotly_figure_show):
        patterns = {"BTCUSDT": [0], "ETHUSDT": [0]}
        strategy = PresetSidesPanel(patterns, data=pairs_panel())

        backtester = VectorizedBacktester(strategy)
        backtester.run(print_results=False, plot_results=False)

        assert backtester.results["return_pct"] == pytest.approx(0.0, abs=1e-12)
        assert backtester.results["nr_trades"] == 0
        assert (backtester.processed_data["equity"] == 1000).all()

    def test_trading_costs_charged_per_leg(self, mocked_plotly_figure_show):
        """Replays the documented recurrence arithmetic explicitly."""
        tc = 0.001  # passed as 0.1 (percent)
        amount = 1000.0

        strategy = PresetSidesPanel(PAIRS_PATTERNS, data=pairs_panel())
        backtester = VectorizedBacktester(strategy, trading_costs=0.1)
        backtester.run(print_results=False, plot_results=False)

        # entries (equal weight, both enter on the first backtest bar):
        equity = amount
        notional_btc = 0.5 * equity * 1.0 / (1 + tc)   # long
        notional_eth = 0.5 * equity * 1.0 / (1 - tc)   # short
        equity -= tc * (notional_btc + notional_eth)
        equity_after_first_bar = equity

        # marks: BTC +10%, ETH -10%
        equity += 1 * (1.1 - 1) * notional_btc
        equity += -1 * (0.9 - 1) * notional_eth
        notional_btc *= 1.1
        notional_eth *= 0.9

        # forced exits on the final bar: cost on the marked notionals
        equity -= tc * (notional_btc + notional_eth)

        # the equity column carries the full arithmetic...
        assert backtester.processed_data["equity"].iloc[-1] == pytest.approx(equity, rel=1e-12)

        # ...while the return metrics are measured from the post-first-bar
        # equity (the first bar's log return is zeroed by construction) -
        # the same convention as the single-symbol path
        expected_return = (equity / equity_after_first_bar - 1) * 100
        assert backtester.results["return_pct"] == pytest.approx(expected_return, rel=1e-12)

    def test_zero_weight_entry_emits_no_trade(self, mocked_plotly_figure_show):
        strategy = PresetSidesPanel(
            PAIRS_PATTERNS,
            weights={"BTCUSDT": 1.0, "ETHUSDT": 0.0},
            data=pairs_panel(),
        )

        backtester = VectorizedBacktester(strategy)
        backtester.run(print_results=False, plot_results=False)

        # only the BTC trade carries capital; the ETH entry is a ghost
        assert backtester.results["nr_trades"] == 1
        assert backtester.trades[0].symbol == "BTCUSDT"
        assert backtester.results["return_pct"] == pytest.approx(10.0, abs=1e-9)

    def test_custom_weights(self, mocked_plotly_figure_show):
        strategy = PresetSidesPanel(
            PAIRS_PATTERNS,
            weights={"BTCUSDT": 0.8, "ETHUSDT": 0.2},
            data=pairs_panel(),
        )

        backtester = VectorizedBacktester(strategy)
        backtester.run(print_results=False, plot_results=False)

        # 0.8 * 10% + 0.2 * 10% = 10% still, but with asymmetric notionals
        assert backtester.results["return_pct"] == pytest.approx(10.0, abs=1e-9)

        trades = {trade.symbol: trade for trade in backtester.trades}
        assert trades["BTCUSDT"].amount == pytest.approx(800.0)
        assert trades["ETHUSDT"].amount == pytest.approx(200.0)

    @pytest.mark.parametrize(
        "weights,error_match",
        [
            ({"BTCUSDT": 0.8, "ETHUSDT": 0.4}, "sum to at most 1"),
            ({"BTCUSDT": -0.1, "ETHUSDT": 0.5}, "non-negative"),
            ({"BTCUSDT": 0.5, "ETHUSDT": np.nan}, "NaN"),
        ],
    )
    def test_invalid_weights_raise(self, weights, error_match, mocked_plotly_figure_show):
        strategy = PresetSidesPanel(PAIRS_PATTERNS, weights=weights, data=pairs_panel())

        backtester = VectorizedBacktester(strategy)

        with pytest.raises(ValueError, match=error_match):
            backtester.run(print_results=False, plot_results=False)

    @pytest.mark.parametrize("backtester_class", [VectorizedBacktester, IterativeBacktester])
    def test_portfolio_wipeout_is_permanent(self, backtester_class, mocked_plotly_figure_show):
        """A short whose loss exceeds the whole portfolio (price > doubles)
        wipes out the cross-collateralized equity permanently."""
        panel = build_panel({
            "BTCUSDT": make_flat_ohlc([100, 100, 100, 350, 350, 350]),
            "ETHUSDT": make_flat_ohlc([100, 100, 100, 100, 100, 100]),
        })
        patterns = {"BTCUSDT": [0, -1, -1, -1, -1, -1], "ETHUSDT": [0, 1, 1, 1, 1, 1]}

        strategy = PresetSidesPanel(patterns, data=panel)
        backtester = backtester_class(strategy)
        backtester.run(print_results=False, plot_results=False)

        # short loses 0.5 * 2.5 = 125% of equity; long is flat -> wiped out
        assert backtester.results["return_pct"] == pytest.approx(-100.0, abs=1e-9)
        equity = backtester.processed_data["equity"]
        first_zero = equity[equity == 0].index[0]
        assert (equity.loc[first_zero:] == 0).all()

    def test_partial_loss_survives(self, mocked_plotly_figure_show):
        """A losing short that doesn't exhaust the portfolio drags equity
        but the other position's gains still count."""
        panel = build_panel({
            "BTCUSDT": make_flat_ohlc([100, 100, 100, 150, 150]),
            "ETHUSDT": make_flat_ohlc([100, 100, 100, 120, 120]),
        })
        patterns = {"BTCUSDT": [0, -1, -1, -1, -1], "ETHUSDT": [0, 1, 1, 1, 1]}

        strategy = PresetSidesPanel(patterns, data=panel)
        backtester = VectorizedBacktester(strategy)
        backtester.run(print_results=False, plot_results=False)

        # short: -50% on half the capital; long: +20% on half -> -15%
        assert backtester.results["return_pct"] == pytest.approx(-15.0, abs=1e-9)
