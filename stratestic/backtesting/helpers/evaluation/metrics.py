from datetime import datetime, timedelta
from functools import reduce
from typing import List, Tuple

import numpy as np
import pandas as pd

from stratestic.backtesting.helpers import Trade
from stratestic.utils.helpers import geometric_mean


def _positional(series_like) -> pd.Series:
    """Return a positionally-indexable Series copy of a Series or Index."""
    return pd.Series(np.asarray(series_like))


def get_total_duration(index: pd.DatetimeIndex, close_date: pd.Series) -> timedelta:
    return _positional(close_date).iloc[-1] - index[0]


def get_start_date(index: pd.DatetimeIndex) -> datetime:
    return index[0]


def get_end_date(close_date: pd.Series) -> datetime:
    return _positional(close_date).iloc[-1]


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
    return (cum_returns.iloc[-1] - 1) * 100


def return_pct_annualized(cum_returns: pd.Series) -> float:
    """Calculate the annualized return percentage.

    Annualization uses the actual elapsed time spanned by the index
    (in fractions of a 365.25-day year), not the number of calendar
    years touched.
    """
    years = (cum_returns.index[-1] - cum_returns.index[0]) / timedelta(days=365.25)

    if years <= 0:
        return np.nan

    # annualizing very short backtests can overflow to inf, which is fine
    with np.errstate(over='ignore'):
        return ((cum_returns.iloc[-1] / cum_returns.iloc[0]) ** (1 / years) - 1) * 100


def volatility_pct_annualized(returns: pd.Series, trading_days: int = 365) -> float:
    """Calculate the annualized volatility percentage."""

    daily_returns = returns.resample('1D').sum()

    return daily_returns.std() * np.sqrt(trading_days) * 100


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0, trading_days: int = 365) -> float:
    """Calculate the Sharpe ratio."""

    # returns must be converted to daily returns

    daily_returns = returns.resample('1D').sum()

    excess_returns = daily_returns - risk_free_rate
    excess_returns_mean = excess_returns.mean()
    excess_returns_std = excess_returns.std()

    return excess_returns_mean / excess_returns_std * np.sqrt(trading_days) if excess_returns_std != 0 else np.nan


def sortino_ratio(
    returns: pd.Series,
    target_return: float = 0,
    risk_free_rate: float = 0,
    trading_days: int = 365
) -> float:
    """Calculate the Sortino ratio.

    The downside deviation only penalizes returns below ``target_return``;
    returns above the target are zeroed out (kept in the count, as per the
    standard downside-deviation definition).
    """
    downside_returns = returns.copy()
    downside_returns[downside_returns > target_return] = 0
    downside_deviation = volatility_pct_annualized(downside_returns, trading_days)

    if downside_deviation == 0 or np.isnan(downside_deviation):
        return np.nan

    cumulative_returns = returns.cumsum().apply(np.exp)
    return (return_pct_annualized(cumulative_returns) - risk_free_rate) / downside_deviation


def calmar_ratio(cum_returns: pd.Series, risk_free_rate: float = 0) -> float:
    """Calculate the Calmar ratio."""
    max_drawdown = max_drawdown_pct(cum_returns) / 100

    if max_drawdown == 0:
        return np.nan

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
    """Calculate the average drawdown percentage.

    Each maximal run of consecutive bars below the running peak counts as
    one drawdown period; the average is taken over each period's trough.
    """
    drawdowns = get_drawdowns(cum_returns)

    in_drawdown = drawdowns < 0

    if not in_drawdown.any():
        return 0

    period_ids = (in_drawdown != in_drawdown.shift()).cumsum()
    troughs = drawdowns[in_drawdown].groupby(period_ids[in_drawdown]).min()

    return troughs.mean() * 100


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
    timedelta
        The duration of the maximum drawdown between peak to peak.
    """
    close_date = _positional(close_date)

    peak_index = 0
    max_drawdown_dur = 0
    dd_start = cum_returns.index[0]
    dd_end = cum_returns.index[-1]

    for i in range(1, len(cum_returns)):
        if cum_returns.iloc[i] > cum_returns.iloc[peak_index]:
            peak_index = i
        else:
            drawdown_duration = i - peak_index
            if drawdown_duration > max_drawdown_dur:
                max_drawdown_dur = drawdown_duration
                dd_start = cum_returns.index[peak_index]
                dd_end = close_date.iloc[i]

    return dd_end - dd_start


def get_dd_durations_limits(cum_returns: pd.Series, close_date: pd.Series) -> Tuple[List, List]:
    close_date = _positional(close_date)

    peak_index = 0

    durations = []
    limits = []

    drawdown = False
    dd_duration = 0
    dd_start_end = (None, None)
    dd_start = cum_returns.index[0]
    for i in range(1, len(cum_returns)):
        if cum_returns.iloc[i] > cum_returns.iloc[peak_index]:
            dd_start = cum_returns.index[i]
            peak_index = i

            if drawdown:
                durations.append(dd_duration)
                limits.append(dd_start_end)
                drawdown = False
        else:
            drawdown = True
            dd_end = close_date.iloc[i]
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
    if len(trades) == 0:
        return np.nan

    return max(trade.pnl for trade in trades) * 100


def worst_trade_pct(trades: List[Trade]) -> float:
    """Calculate the percentage of the worst trade."""
    if len(trades) == 0:
        return np.nan

    return min(trade.pnl for trade in trades) * 100


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
        avg_loss = avg_trade_pct(lose_trades) / 100  # negative by construction
        return (win_rate * avg_win + (1 - win_rate) * avg_loss) * 100


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

    if len(trades) < 2:
        return np.nan

    net_profit = trades_net_profit(trades)

    N = len(trades)
    avg = np.mean(net_profit)
    std = np.std(net_profit)

    if std == 0:
        return np.nan

    return np.sqrt(N) * (avg / std)
