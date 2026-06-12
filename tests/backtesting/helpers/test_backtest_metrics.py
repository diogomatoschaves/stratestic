import pytest
from pandas import Timedelta, Timestamp

from stratestic.backtesting.helpers.evaluation.metrics import *
from stratestic.utils.helpers import geometric_mean
from tests.setup.test_data.returns import (
    cum_returns,
    returns,
    index,
    daily_returns,
    daily_cum_returns,
)

# Expected values below are hand-computed from the fixture data; the
# arithmetic is documented next to each assertion.
#
# Hourly data (10 bars, 1h apart, spanning 9h):
#   log_returns = [0.025, 0.016, -0.10, 0.03, 0.07, 0.04, -0.02, -0.05, 0.01, 0.03]
#   daily sums: Sep 1 (4 bars) = -0.029, Sep 2 (6 bars) = 0.08
#   cumulative = [1.25, 1.26, 1.27, 1.23, 1.20, 1.22, 1.25, 1.30, 1.29, 1.26]
#
# Daily data (550 days from 2022-01-01, repeating pattern
# [0.01, -0.005, 0.002, -0.001, 0.004], spanning 549/365.25 = 1.503080 years).


@pytest.fixture
def trades():
    # Generate some sample trades
    return [
        Trade(entry_date=datetime(2022, 1, 1, 0, 0),
              exit_date=datetime(2022, 1, 3, 0, 0),
              entry_price=10, exit_price=4, units=100, side=1, profit=-600, pnl=-0.4),
        Trade(entry_date=datetime(2022, 1, 2, 0, 0),
              exit_date=datetime(2022, 1, 4, 0, 0),
              entry_price=12, exit_price=10, units=100, side=-1, profit=200, pnl=0.2),
        Trade(entry_date=datetime(2022, 1, 3, 0, 0),
              exit_date=datetime(2022, 1, 5, 0, 0),
              entry_price=11, exit_price=12, units=100, side=1, profit=100, pnl=0.1),
        Trade(entry_date=datetime(2022, 1, 4, 0, 0),
              exit_date=datetime(2022, 1, 8, 0, 0),
              entry_price=13, exit_price=14, units=100, side=-1, profit=-100, pnl=-0.1),
        Trade(entry_date=datetime(2022, 1, 5, 0, 0),
              exit_date=datetime(2022, 1, 10, 0, 0),
              entry_price=12, exit_price=16, units=100, side=1, profit=400, pnl=0.4)
    ]


@pytest.fixture
def losing_only_trades():
    return [
        Trade(entry_date=datetime(2022, 1, 1, 0, 0),
              exit_date=datetime(2022, 1, 2, 0, 0),
              entry_price=10, exit_price=9, units=100, side=1, profit=-100, pnl=-0.1),
        Trade(entry_date=datetime(2022, 1, 3, 0, 0),
              exit_date=datetime(2022, 1, 4, 0, 0),
              entry_price=10, exit_price=6, units=100, side=1, profit=-400, pnl=-0.4),
    ]


@pytest.fixture
def break_even_trades():
    return [
        Trade(entry_date=datetime(2022, 1, 1, 0, 0),
              exit_date=datetime(2022, 1, 2, 0, 0),
              entry_price=10, exit_price=11, units=100, side=1, profit=100, pnl=0.1),
        Trade(entry_date=datetime(2022, 1, 3, 0, 0),
              exit_date=datetime(2022, 1, 4, 0, 0),
              entry_price=10, exit_price=9, units=100, side=1, profit=-100, pnl=-0.1),
    ]


@pytest.fixture
def cumulative_returns():
    return cum_returns


@pytest.fixture
def log_returns():
    return returns


def test_get_total_duration():

    close_date = index.shift(1)

    total_duration = get_total_duration(index, close_date)

    assert total_duration == Timedelta('0 days 10:00:00')


def test_get_start_date():

    start_date = get_start_date(index)

    assert start_date == Timestamp('2023-09-01 20:00:00+0000', tz='UTC')


def test_get_end_date():
    close_date = index.shift(1)

    end_date = get_end_date(close_date)

    assert end_date == Timestamp('2023-09-02 06:00:00+0000', tz='UTC')


def test_exposure_time():
    positions = np.array([0, 1, 1, 0, 1, 0])
    assert exposure_time(positions) == 50.0


def test_equity_final():
    equity_curve = pd.Series([100, 120, 80, 110, 90])
    assert equity_final(equity_curve) == 90


def test_equity_peak():
    equity_curve = pd.Series([100, 120, 80, 110, 90])
    assert equity_peak(equity_curve) == 120


def test_return_pct():
    equity_curve = pd.Series([100, 120, 80, 110, 90])
    # 90 / 100 - 1 = -10%
    assert return_pct(equity_curve) == pytest.approx(-10.0, rel=1e-6)


def test_annual_return_pct():
    # growth = cum[-1]/cum[0] = exp(1.1 - 0.01) = exp(1.09)
    # years = 549 / 365.25 = 1.503080
    # annualized = exp(1.09 / 1.503080) - 1 = 106.50978%
    assert return_pct_annualized(daily_cum_returns) == pytest.approx(106.50978, rel=1e-5)


def test_annual_return_pct_sub_daily(cumulative_returns):
    # 9-hour span: years = 9/8766 = 0.00102669
    # (1.26/1.25)^(1/years) - 1 = 234624.42% -- annualizing very short
    # backtests produces extreme values by construction.
    assert return_pct_annualized(cumulative_returns) == pytest.approx(234624.42, rel=1e-5)


def test_annual_return_independent_of_year_boundary():
    # Two identical 60-day return series, one straddling New Year, one not,
    # must annualize identically (the old implementation counted calendar
    # year bins and would halve the exponent for the straddling one).
    r60 = np.tile([0.002, -0.001], 30)

    straddling = pd.Series(
        np.exp(np.cumsum(r60)),
        index=pd.date_range('2022-12-02', periods=60, freq='D', tz='UTC')
    )
    contained = pd.Series(
        np.exp(np.cumsum(r60)),
        index=pd.date_range('2023-03-02', periods=60, freq='D', tz='UTC')
    )

    expected = 18.926918  # (exp(0.03 - 0.002))^(365.25/59) - 1

    assert return_pct_annualized(straddling) == pytest.approx(expected, rel=1e-5)
    assert return_pct_annualized(contained) == pytest.approx(expected, rel=1e-5)


def test_annual_volatility_pct(log_returns):
    # daily sums: [-0.029, 0.08]; sample std (ddof=1) = 0.0545 * sqrt(2)
    # = 0.0770747; * sqrt(365) * 100 = 147.2509
    assert volatility_pct_annualized(log_returns) == pytest.approx(147.2509, rel=1e-5)


def test_sharpe_ratio(log_returns):
    # daily excess returns: [-0.039, 0.07]; mean = 0.0155,
    # std (ddof=1) = 0.0770747; sharpe = 0.0155/0.0770747*sqrt(365) = 3.84208
    assert sharpe_ratio(log_returns, 0.01) == pytest.approx(3.84208, rel=1e-5)


def test_sortino_ratio():
    # downside (returns > 0 zeroed): -0.005 and -0.001 each 110x, 0 otherwise;
    # sample std * sqrt(365) * 100 = 3.7079642 (denominator)
    # annualized return = 106.50978% (numerator)
    # ratio = 28.72460
    assert sortino_ratio(daily_returns) == pytest.approx(28.72460, rel=1e-5)


def test_sortino_ratio_no_downside():
    rets = pd.Series(
        [0.01, 0.02, 0.005],
        index=pd.date_range('2022-01-01', periods=3, freq='D', tz='UTC')
    )
    assert np.isnan(sortino_ratio(rets))


def test_calmar_ratio():
    # max drawdown: trough of the repeating pattern = exp(-0.005) - 1
    # = -0.4987521%; annual return = 106.50978%
    # calmar = (1.0650978 - 0.01) / 0.004987521 = 211.54755
    assert calmar_ratio(daily_cum_returns, 0.01) == pytest.approx(211.54755, rel=1e-5)


def test_calmar_ratio_no_drawdown():
    cum = pd.Series(
        [1.0, 1.1, 1.2],
        index=pd.date_range('2022-01-01', periods=3, freq='D', tz='UTC')
    )
    assert np.isnan(calmar_ratio(cum))


def test_max_drawdown_pct(cumulative_returns):
    # deepest trough: (1.20 - 1.27) / 1.27 = -5.51181%
    assert max_drawdown_pct(cumulative_returns) == pytest.approx(-5.51181, rel=1e-5)


def test_avg_drawdown_pct(cumulative_returns):
    # two drawdown periods: bars 3-6 (trough (1.20-1.27)/1.27 = -5.51181%)
    # and bars 8-9 (trough (1.26-1.30)/1.30 = -3.07692%); mean = -4.29437%
    assert avg_drawdown_pct(cumulative_returns) == pytest.approx(-4.29437, rel=1e-5)


def test_avg_drawdown_pct_single_bar_drawdown():
    # one single-bar drawdown: (0.9 - 1.0) / 1.0 = -10%; the old
    # implementation skipped the first bar of each drawdown and dropped it.
    cum = pd.Series(
        [1.0, 0.9, 1.1, 1.2],
        index=pd.date_range('2022-01-01', periods=4, freq='D', tz='UTC')
    )
    assert avg_drawdown_pct(cum) == pytest.approx(-10.0, rel=1e-6)


def test_avg_drawdown_pct_no_drawdown():
    cum = pd.Series(
        [1.0, 1.1, 1.2],
        index=pd.date_range('2022-01-01', periods=3, freq='D', tz='UTC')
    )
    assert avg_drawdown_pct(cum) == 0


def test_max_drawdown_duration(cumulative_returns):
    close_date = cumulative_returns.index.shift(1)

    assert max_drawdown_duration(cumulative_returns, close_date) == timedelta(hours=5)


def test_avg_drawdown_duration(cumulative_returns):
    close_date = cumulative_returns.index.shift(1)

    assert round(avg_drawdown_duration(cumulative_returns, close_date), 2) == 14400.0


def test_win_rate_pct(trades):
    assert win_rate_pct(trades) == 60.0


def test_best_trade_pct(trades):
    # max pnl = 0.4 -> 40%
    assert best_trade_pct(trades) == pytest.approx(40.0, rel=1e-6)


def test_worst_trade_pct(trades):
    # min pnl = -0.4 -> -40%
    assert worst_trade_pct(trades) == pytest.approx(-40.0, rel=1e-6)


def test_best_worst_trade_pct_all_losing(losing_only_trades):
    # the old implementation clamped at 0 and reported 0% as the best trade
    assert best_trade_pct(losing_only_trades) == pytest.approx(-10.0, rel=1e-6)
    assert worst_trade_pct(losing_only_trades) == pytest.approx(-40.0, rel=1e-6)


def test_best_worst_trade_pct_no_trades():
    assert np.isnan(best_trade_pct([]))
    assert np.isnan(worst_trade_pct([]))


def test_avg_trade_pct(trades):
    # geometric mean: (0.6 * 1.2 * 1.1 * 0.9 * 1.4)^(1/5) - 1
    # = 0.99792^(1/5) - 1 = -0.0416347%
    assert avg_trade_pct(trades) == pytest.approx(-0.0416347, rel=1e-5)


def test_max_trade_duration(trades):
    assert max_trade_duration(trades) == timedelta(days=5)


def test_avg_trade_duration(trades):
    assert avg_trade_duration(trades) == 259200.0


def test_winning_trades(trades):
    assert len(winning_trades(trades)) == 3


def test_losing_trades(trades):
    assert len(losing_trades(trades)) == 2


def test_trades_net_sum(trades):
    assert trades_net_profit_sum(trades) == 0


def test_profit_factor(trades):
    assert profit_factor(trades) == 1


def test_expectancy_pct(trades):
    # avg_win = (1.2 * 1.1 * 1.4)^(1/3) - 1 = 22.71585%
    # avg_loss = (0.6 * 0.9)^(1/2) - 1 = -26.51531%
    # expectancy = 0.6 * 22.71585 + 0.4 * (-26.51531) = 3.02339%
    assert expectancy_pct(trades) == pytest.approx(3.02339, rel=1e-5)


def test_expectancy_pct_break_even(break_even_trades):
    # 50% win rate, avg_win = 10%, avg_loss = -10% -> expectancy = 0
    assert expectancy_pct(break_even_trades) == pytest.approx(0.0, abs=1e-9)


def test_sqn(trades):
    # net profits: [-600, 200, 100, -100, 400]; mean = 0 -> SQN = 0
    assert system_quality_number(trades) == 0


def test_sqn_guards(trades):
    assert np.isnan(system_quality_number([]))
    assert np.isnan(system_quality_number(trades[:1]))


def test_geometric_mean_total_loss():
    # a leveraged pnl <= -100% makes the growth product non-positive;
    # must return NaN instead of silently propagating a complex/NaN power
    assert np.isnan(geometric_mean(pd.Series([-1.2, 0.5])))
