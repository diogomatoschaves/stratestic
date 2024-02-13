from datetime import datetime, timedelta
from functools import reduce
from typing import List, Tuple

import numpy as np
import pandas as pd

from stratestic.backtesting.helpers import Trade
from stratestic.utils.helpers import geometric_mean


def get_total_duration(index: pd.DatetimeIndex, close_date: pd.Series) -> timedelta:
    return close_date[-1] - index[0]


def get_start_date(index: pd.DatetimeIndex) -> datetime:
    return index[0]


def get_end_date(close_date: pd.Series) -> datetime:
    return close_date[-1]


def exposure_time(positions: np.ndarray) -> float:
    """Calculate the percentage of time the strategy was exposed to the market."""
    return np.count_nonzero(positions) / len(positions) * 100


def equity_final(equity_curve: pd.Series) -> float:
    """Calculate the final equity value."""
    return equity_curve.iloc[-1]


def equity_peak(equity_curve: pd.Series) -> float:
    """Calculate the peak equity value."""
    return np.max(equity_curve)


def return_pct(equity_curve: pd.Series) -> float:
    """Calculate the total return percentage."""
    return (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100


def return_buy_and_hold_pct(cum_returns: pd.Series) -> float:
    """Retrieve the buy and hold return pct."""
    return (cum_returns[-1] - 1) * 100


def return_pct_annualized(cum_returns: pd.Series) -> float:
    """Calculate the annualized return percentage."""

    years_df = cum_returns.resample('1Y').count()

    return ((cum_returns[-1] / cum_returns[0])**(1 / len(years_df))-1) * 100


def volatility_pct_annualized(returns: pd.Series, trading_days: int = 365) -> float:
    """Calculate the annualized volatility percentage."""

    daily_returns = returns.resample('1D').sum()

    return np.std(daily_returns) * np.sqrt(trading_days) * 100


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0, trading_days: int = 365) -> float:
    """Calculate the Sharpe ratio."""

    # returns must be converted to daily returns

    daily_returns = returns.resample('1D').sum()

    excess_returns = daily_returns - risk_free_rate
    excess_returns_mean = excess_returns.mean()
    excess_returns_std = excess_returns.std()

    return excess_returns_mean / excess_returns_std * np.sqrt(trading_days) if excess_returns_std != 0 else np.nan


def sortino_ratio(returns: pd.Series, target_return: float = 0, risk_free_rate: float = 0) -> float:
    """Calculate the Sortino ratio."""
    downside_returns = returns.copy()
    downside_returns[downside_returns < target_return] = 0
    downside_deviation = volatility_pct_annualized(downside_returns)
    cumulative_returns = returns.cumsum().apply(np.exp)
    return (return_pct_annualized(cumulative_returns) - risk_free_rate) / downside_deviation


def calmar_ratio(cum_returns: pd.Series, risk_free_rate: float = 0) -> float:
    """Calculate the Calmar ratio."""
    max_drawdown = max_drawdown_pct(cum_returns) / 100
    annual_return = return_pct_annualized(cum_returns) / 100
    return (annual_return - risk_free_rate) / -max_drawdown


def get_drawdowns(cum_returns: pd.Series) -> pd.Series:
    """Retrieves the drawdown periods."""
    peaks = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - peaks) / peaks

    return drawdowns


def max_drawdown_pct(cum_returns: pd.Series) -> float:
    """Calculate the maximum drawdown percentage."""
    drawdowns = get_drawdowns(cum_returns)

    return np.min(drawdowns) * 100


def avg_drawdown_pct(cum_returns: pd.Series) -> float:
    """Calculate the average drawdown percentage."""
    drawdowns = get_drawdowns(cum_returns)

    drawdown_maximums = []
    drawdown_period = []

    is_zero = True
    for dd in drawdowns:
        if dd == 0:
            if not is_zero:
                if len(drawdown_period) != 0:
                    drawdown_maximums.append(np.min(drawdown_period))
            is_zero = True
        else:
            if is_zero:
                drawdown_period = []
                is_zero = False
            else:
                drawdown_period.append(dd)

    if len(drawdown_period) != 0:
        drawdown_maximums.append(np.min(drawdown_period))

    return np.mean(drawdown_maximums) * 100


def max_drawdown_duration(cum_returns: pd.Series, close_date: pd.Series) -> int:
    """Calculate the duration of the maximum drawdown in terms of the period between peak to peak.

    Parameters
    ----------
    cum_returns : pd.Series
        An array of cumulative returns.
    close_date : pd.Series
        An array of dates with the close of the period.

    Returns
    -------
    int
        The duration of the maximum drawdown between peak to peak in the used time unit.
    """
    peak_index = 0
    max_drawdown_dur = 0
    dd_start = cum_returns.index[0]
    dd_end = cum_returns.index[-1]

    for i in range(1, len(cum_returns)):
        if cum_returns[i] > cum_returns[peak_index]:
            peak_index = i
        else:
            drawdown_duration = i - peak_index
            if drawdown_duration > max_drawdown_dur:
                max_drawdown_dur = drawdown_duration
                dd_start = cum_returns.index[peak_index]
                dd_end = close_date[i]

    return dd_end - dd_start


def get_dd_durations_limits(cum_returns: pd.Series, close_date: pd.Series) -> Tuple[List, List]:
    peak_index = 0

    durations = []
    limits = []

    drawdown = False
    dd_duration = 0
    dd_start_end = (None, None)
    dd_start = cum_returns.index[0]
    for i in range(1, len(cum_returns)):
        if cum_returns[i] > cum_returns[peak_index]:
            dd_start = cum_returns.index[i]
            peak_index = i

            if drawdown:
                durations.append(dd_duration)
                limits.append(dd_start_end)
                drawdown = False
        else:
            drawdown = True
            dd_end = close_date[i]
            dd_start_end = (dd_start, dd_end)
            dd_duration = (dd_end - dd_start).total_seconds()

    if drawdown:
        durations.append(dd_duration)
        limits.append(dd_start_end)

    return durations, limits


def avg_drawdown_duration(cum_returns: pd.Series, close_date: pd.Series) -> float:
    """Calculate the average duration of drawdowns."""
    durations, _ = get_dd_durations_limits(cum_returns, close_date)

    return np.mean(durations) if len(durations) > 0 else 0


def win_rate_pct(trades: List[Trade]) -> float:
    """Calculate the percentage of winning trades."""
    winning_trades = reduce(
        lambda accum, trade: accum + 1 if (trade.exit_price - trade.entry_price) * trade.side > 0 else accum,
        trades,
        0
    )

    nr_trades = len(trades)

    return winning_trades / nr_trades * 100 if nr_trades > 0 else 0


def best_trade_pct(trades: List[Trade]) -> float:
    """Calculate the percentage of the best trade."""

    def reducer(accum, trade):
        return trade.pnl if trade.pnl > accum else accum

    best_trade = reduce(
        reducer,
        trades,
        0
    )
    return best_trade * 100


def worst_trade_pct(trades: List[Trade]) -> float:
    """Calculate the percentage of the worst trade."""

    def reducer(accum, trade):
        return trade.pnl if trade.pnl < accum else accum

    worst_trade = reduce(
        reducer,
        trades,
        0
    )

    return worst_trade * 100


def avg_trade_pct(trades: List[Trade]) -> float:
    """Calculate the average trade percentage."""
    trades_pct = map(
        lambda trade: trade.pnl,
        trades
    )

    intermediate = pd.Series(list(trades_pct))

    return geometric_mean(intermediate) * 100


def max_trade_duration(trades: List[Trade]) -> int:
    """Calculate the duration of the longest trade."""
    durations = list(map(
        lambda trade: trade.exit_date - trade.entry_date,
        trades,
    ))

    return np.max(durations) if len(durations) > 0 else 0


def avg_trade_duration(trades: List[Trade]) -> int:
    """Calculate the duration of the longest trade."""
    durations = map(
        lambda trade: (trade.exit_date - trade.entry_date).total_seconds(),
        trades,
    )

    durations = list(durations)

    return np.mean(durations) if len(durations) > 0 else 0


def winning_trades(trades: List[Trade]) -> List[Trade]:
    return reduce(
        lambda accum, trade: [*accum, trade]
        if (trade.exit_price - trade.entry_price) * trade.side > 0
        else accum,
        trades,
        []
    )


def losing_trades(trades: List[Trade]) -> List[Trade]:
    return reduce(
        lambda accum, trade: [*accum, trade]
        if (trade.exit_price - trade.entry_price) * trade.side < 0
        else accum,
        trades,
        []
    )


def trades_net_profit_sum(trades: List[Trade]) -> float:
    return reduce(
        lambda accum, trade: accum + trade.units * (trade.exit_price - trade.entry_price) * trade.side,
        trades,
        0
    )


def trades_net_profit(trades: List[Trade]) -> List[float]:
    return list(map(
        lambda trade: trade.units * (trade.exit_price - trade.entry_price) * trade.side,
        trades,
    ))


def expectancy_pct(trades: List[Trade]) -> float:
    """Calculate the expectancy percentage."""
    win_trades = winning_trades(trades)
    lose_trades = losing_trades(trades)
    if len(lose_trades) == 0:
        return avg_trade_pct(win_trades)
    elif len(win_trades) == 0:
        return avg_trade_pct(lose_trades)
    else:
        win_rate = win_rate_pct(trades) / 100
        avg_win = avg_trade_pct(win_trades) / 100
        avg_loss = avg_trade_pct(lose_trades) / 100
        return (win_rate * avg_win - (1 - win_rate) * avg_loss) * 100


def profit_factor(trades: List[Trade]) -> float:
    """Calculate the profit factor."""
    win_trades = winning_trades(trades)
    lose_trades = losing_trades(trades)

    if len(lose_trades) == 0:
        return np.inf
    else:
        return trades_net_profit_sum(win_trades) / np.abs(trades_net_profit_sum(lose_trades))


def system_quality_number(trades):
    """Calculate the System Quality Number."""

    net_profit = trades_net_profit(trades)

    N = len(trades)
    avg = np.mean(net_profit)
    std = np.std(net_profit)

    return np.sqrt(N) * (avg / std)
