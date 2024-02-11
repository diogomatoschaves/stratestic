import pytest
from pandas import Timedelta, Timestamp

from stratestic.backtesting.helpers.evaluation.metrics import *
from tests.setup.test_data.returns import cum_returns, returns, index


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

    assert start_date == Timestamp('2023-09-01 20:00:00+0000', tz='UTC', freq='H')


def test_get_end_date():
    close_date = index.shift(1)

    end_date = get_end_date(close_date)

    assert end_date == Timestamp('2023-09-02 06:00:00+0000', tz='UTC', freq='H')


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
    assert return_pct(equity_curve) == pytest.approx(-10.0, 1)


def test_annual_return_pct(cumulative_returns):
    assert return_pct_annualized(cumulative_returns) == pytest.approx(0.8, 3)


def test_annual_volatility_pct(log_returns):
    assert volatility_pct_annualized(log_returns) == pytest.approx(104.12, 3)


def test_sharpe_ratio(log_returns):
    assert sharpe_ratio(log_returns, 0.01) == pytest.approx(3.84, 1)


def test_sortino_ratio(log_returns):
    assert sortino_ratio(log_returns, 0.01) == pytest.approx(0.0689, 3)


def test_calmar_ratio(cumulative_returns):
    assert calmar_ratio(cumulative_returns, 0.01) == pytest.approx(-0.14, 3)


def test_max_drawdown_pct(cumulative_returns):
    assert max_drawdown_pct(cumulative_returns) == pytest.approx(-5.51, 1)


def test_avg_drawdown_pct(cumulative_returns):
    assert avg_drawdown_pct(cumulative_returns) == pytest.approx(4.29, 3)


def test_max_drawdown_duration(cumulative_returns):
    close_date = cumulative_returns.index.shift(1)

    assert max_drawdown_duration(cumulative_returns, close_date) == timedelta(hours=5)


def test_avg_drawdown_duration(cumulative_returns):
    close_date = cumulative_returns.index.shift(1)

    assert round(avg_drawdown_duration(cumulative_returns, close_date), 2) == 14400.0


def test_win_rate_pct(trades):
    assert win_rate_pct(trades) == 60.0


def test_best_trade_pct(trades):
    assert best_trade_pct(trades) == pytest.approx(33.3, 1)


def test_worst_trade_pct(trades):
    assert worst_trade_pct(trades) == pytest.approx(-60, 1)


def test_avg_trade_pct(trades):
    assert avg_trade_pct(trades) == pytest.approx(-8.926, 2)


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
    assert expectancy_pct(trades) == pytest.approx(1546.15, 2)


def test_sqn(trades):
    assert system_quality_number(trades) == 0


