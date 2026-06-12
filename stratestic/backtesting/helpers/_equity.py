import numpy as np
from numba import jit

SHORT_MODELS = ("inverse", "static")


@jit(nopython=True, cache=True)
def calculate_leveraged_equity(
    trade_markers: np.ndarray,
    strategy_returns: np.ndarray,
    amount: float,
    leverage: float,
) -> np.ndarray:
    """
    Compute the equity curve of a leveraged strategy from per-bar log returns.

    The traded notional is reset to ``equity * leverage`` on each bar where a
    new trade begins (marked by a change in ``trade_markers``), and compounds
    with the pnl otherwise. Equity is floored at zero - once the account is
    wiped out it stays at zero.

    Parameters
    ----------
    trade_markers : np.ndarray
        Array of floats that changes value on each bar where a new trade is
        opened (e.g. an increasing trade id, forward-filled).
    strategy_returns : np.ndarray
        Per-bar strategy log returns (after costs, if applicable).
    amount : float
        Initial amount of capital.
    leverage : float
        Leverage multiplier.

    Returns
    -------
    np.ndarray
        Equity value at each bar.
    """
    equity_arr = np.empty(strategy_returns.shape[0])

    equity = amount
    current_marker = trade_markers[0]
    amount = amount * leverage

    for index in range(strategy_returns.shape[0]):

        pnl = (np.exp(strategy_returns[index]) - 1) * amount

        equity = equity + pnl

        if equity <= 0:
            equity = 0.0
            amount = 0.0
        elif trade_markers[index] != current_marker:
            amount = equity * leverage
            current_marker = trade_markers[index]
        else:
            amount = amount + pnl

        equity_arr[index] = equity

    return equity_arr


@jit(nopython=True, cache=True)
def calculate_static_equity(
    sides: np.ndarray,
    returns: np.ndarray,
    trading_costs: float,
    amount: float,
    leverage: float,
) -> np.ndarray:
    """
    Compute the equity curve of a strategy holding *static* positions.

    Under the static model, the number of units is fixed when a trade is
    opened (``units = equity * leverage / (price * (1 + tc * side))``, i.e.
    real cash accounting), so the marked notional compounds with the *price*
    return while the pnl accrues on the fixed units. For longs this is
    identical to the inverse model; for shorts it reproduces a real
    fixed-units short instead of a continuously-rebalanced inverse position.

    Trading costs are charged per leg on the traded notional: on the marked
    notional when closing, and embedded in the entry by allocating
    ``equity * leverage / (1 + tc * side)`` and paying ``tc`` on it.

    Parameters
    ----------
    sides : np.ndarray
        Per-bar position (1 long, -1 short, 0 flat) effective from the close
        of that bar.
    returns : np.ndarray
        Per-bar *price* log returns (no costs, no position sign).
    trading_costs : float
        Trading cost per leg as a fraction of the traded notional.
    amount : float
        Initial amount of capital.
    leverage : float
        Leverage multiplier.

    Returns
    -------
    np.ndarray
        Equity value at each bar.
    """
    equity_arr = np.empty(returns.shape[0])

    equity = amount
    notional = 0.0
    prev_side = 0.0

    for index in range(returns.shape[0]):
        new_side = sides[index]

        # mark the fixed units to market
        growth = np.exp(returns[index])
        equity = equity + prev_side * (growth - 1.0) * notional
        notional = notional * growth

        if equity <= 0:
            equity = 0.0
            notional = 0.0
            new_side = 0.0
        elif new_side != prev_side:
            if prev_side != 0:
                # close the open position: cost on the marked notional
                equity = equity - trading_costs * notional
                notional = 0.0

            if equity <= 0:
                equity = 0.0
                new_side = 0.0
            elif new_side != 0:
                # open the new position: entry cost embedded in the units
                notional = equity * leverage / (1.0 + trading_costs * new_side)
                equity = equity - trading_costs * notional

                if equity <= 0:
                    equity = 0.0
                    notional = 0.0
                    new_side = 0.0

        prev_side = new_side
        equity_arr[index] = equity

    return equity_arr
