import numpy as np
import pytest

from stratestic.backtesting.helpers._equity import calculate_leveraged_equity, calculate_static_equity


def test_rebalances_on_each_new_trade_marker():
    # hand-traced recurrence: notional resets to equity * leverage on each
    # marker change, compounds with the pnl otherwise
    markers = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0])
    returns = np.array([0.0, 0.1, -0.05, 0.1, 0.0, -0.6, 0.05])

    equity = calculate_leveraged_equity(markers, returns, 1000.0, 2.0)

    amount = 2000.0
    expected = []
    current = 1000.0
    current_marker = 0.0
    for marker, r in zip(markers, returns):
        pnl = (np.exp(r) - 1) * amount
        current = current + pnl
        if marker != current_marker:
            amount = current * 2.0
            current_marker = marker
        else:
            amount = amount + pnl
        expected.append(current)

    assert equity == pytest.approx(expected, rel=1e-12)


def test_no_rebalance_on_first_bar():
    # a trade opening on the very first bar must not rebalance: the notional
    # is already amount * leverage
    markers = np.array([1.0, 1.0])
    returns = np.array([0.1, 0.1])

    equity = calculate_leveraged_equity(markers, returns, 1000.0, 2.0)

    pnl_0 = (np.exp(0.1) - 1) * 2000.0
    pnl_1 = (np.exp(0.1) - 1) * (2000.0 + pnl_0)

    assert equity[1] == pytest.approx(1000.0 + pnl_0 + pnl_1, rel=1e-12)


def test_equity_floors_at_zero_after_wipeout():
    # a leveraged loss larger than the account wipes it out; equity must
    # stay at zero instead of going negative
    markers = np.array([0.0, 0.0, 0.0, 0.0])
    returns = np.array([0.0, -0.5, 0.1, 0.2])

    equity = calculate_leveraged_equity(markers, returns, 1000.0, 5.0)

    assert (np.exp(-0.5) - 1) * 5000.0 < -1000.0  # sanity: the loss exceeds equity
    assert list(equity) == [1000.0, 0.0, 0.0, 0.0]


def test_unleveraged_equity_equals_compounded_returns():
    markers = np.array([0.0, 1.0, 1.0, 2.0, 2.0])
    returns = np.array([0.0, 0.05, -0.02, 0.03, 0.01])

    equity = calculate_leveraged_equity(markers, returns, 1000.0, 1.0)

    assert equity == pytest.approx(1000.0 * np.exp(np.cumsum(returns)), rel=1e-12)


class TestStaticEquity:

    def test_short_gains_are_fixed_units(self):
        # short 1000 at 100, price falls to 50: a static short gains +50%
        # (the inverse model would report +100%)
        sides = np.array([-1.0, -1.0, 0.0])
        returns = np.array([0.0, np.log(0.5), 0.0])

        equity = calculate_static_equity(sides, returns, 0.0, 1000.0, 1.0)

        assert equity[-1] == pytest.approx(1500.0, rel=1e-12)

    def test_short_loss_wipes_out_account(self):
        # short 1000 at 100, price doubles: a static short loses -100%
        # (the inverse model would report only -50%)
        sides = np.array([-1.0, -1.0, 0.0])
        returns = np.array([0.0, np.log(2.0), 0.0])

        equity = calculate_static_equity(sides, returns, 0.0, 1000.0, 1.0)

        assert equity[-1] == 0.0

    def test_long_equals_inverse_model(self):
        # longs are fixed-units under both models; without costs the curves
        # must be identical while a long position is held
        sides = np.array([1.0, 1.0, 1.0, 0.0])
        returns = np.array([0.0, 0.05, -0.02, 0.01])
        markers = np.array([1.0, 1.0, 1.0, 1.0])
        strategy_returns = np.array([0.0, 0.05, -0.02, 0.01])  # side 1: same as price returns

        static = calculate_static_equity(sides, returns, 0.0, 1000.0, 2.0)
        inverse = calculate_leveraged_equity(markers, strategy_returns, 1000.0, 2.0)

        # position closes on the last bar in the static run (side -> 0), which
        # is cost-free here, so the full curves must agree
        assert static == pytest.approx(inverse, rel=1e-12)

    def test_costs_charged_per_leg_on_notional(self):
        # single long round trip with flat prices: equity only pays the costs.
        # entry: notional = E*L/(1+tc), cost tc*notional; exit: cost tc*notional
        tc, leverage = 0.01, 2.0
        sides = np.array([1.0, 1.0, 0.0])
        returns = np.array([0.0, 0.0, 0.0])

        equity = calculate_static_equity(sides, returns, tc, 1000.0, leverage)

        notional = 1000.0 * leverage / (1 + tc)
        after_entry = 1000.0 - tc * notional
        after_exit = after_entry - tc * notional

        assert equity[0] == pytest.approx(after_entry, rel=1e-12)
        assert equity[-1] == pytest.approx(after_exit, rel=1e-12)
