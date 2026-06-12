import numpy as np
from numba import jit

SHORT_MODELS = ("inverse", "static")


def static_trade_result(entry_price_raw, exit_price_raw, side, trading_costs, leverage, entry_notional):
    """
    Pnl and cash profit of one static trade, consistent with the equity
    recurrence (costs per leg on the traded notional, entry cost embedded
    in the sizing). Shared by both engines so their floats match exactly.

    pnl is relative to the capital allocated to the trade and is
    independent of the allocation weight; capped at -100%.
    """
    ratio = exit_price_raw / entry_price_raw
    edge = side * (ratio - 1.0) - trading_costs * (1.0 + ratio)

    pnl = max(leverage * edge / (1.0 + trading_costs * side), -1.0)
    profit = entry_notional * edge

    return pnl, profit


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


@jit(nopython=True, cache=True)
def calculate_static_equity_panel(
    sides: np.ndarray,          # float64 [n_bars, n_symbols]
    returns: np.ndarray,        # float64 [n_bars, n_symbols], NaN -> 0, row 0 zeroed
    weights: np.ndarray,        # float64 [n_bars, n_symbols], read on entry bars only
    liquidations: np.ndarray,   # bool    [n_bars, n_symbols]
    trading_costs: float,
    amount: float,
    leverage: float,
):
    """
    Portfolio equity curve for static positions across multiple symbols.

    The static model is additive across positions: per bar, each position
    contributes ``side * (exp(r) - 1) * notional`` to the single shared
    equity pot. Per bar, three phases run in symbol (column) order:

    1. mark every position to market;
    2. exits (including the exit leg of flips): cost on the marked
       notional; a liquidation instead debits the position's residual
       isolated margin (loss capped at the margin);
    3. entries, all sized off the same post-exit equity:
       ``notional = w * equity * leverage / (1 + tc * side)`` with the
       entry cost embedded, exactly like the single-symbol static model.

    Wipeout is portfolio-level: equity <= 0 at any check floors everything
    permanently (the cash pot is cross-collateralized).

    Returns
    -------
    (equity_arr, notionals)
        equity_arr: float64 [n_bars] - portfolio equity at each bar.
        notionals: float64 [n_bars, n_symbols] - marked notional at each
        bar end; on an entry bar this is the entry notional (needed by
        trade retrieval, since it depends on intra-bar equity).
    """
    n_bars, n_symbols = returns.shape

    equity_arr = np.zeros(n_bars)
    notionals_out = np.zeros((n_bars, n_symbols))

    equity = amount
    notional = np.zeros(n_symbols)
    entry_notional = np.zeros(n_symbols)
    prev_side = np.zeros(n_symbols)
    wiped = False

    for i in range(n_bars):
        if wiped:
            continue  # arrays already zero-initialized

        # ---- Phase 1: mark every position to market ----
        # (the wipeout check happens after Phase 2: a liquidation's
        # margin-bounded debit may bring a gapped-through equity back)
        for j in range(n_symbols):
            growth = np.exp(returns[i, j])
            equity = equity + prev_side[j] * (growth - 1.0) * notional[j]
            notional[j] = notional[j] * growth

        # ---- Phase 2: exits (and exit leg of flips) ----
        for j in range(n_symbols):
            if liquidations[i, j]:
                # isolated-margin liquidation: forfeit the position's
                # residual margin; the cumulative marks already booked the
                # running pnl, so the net loss lands at exactly the margin
                residual = entry_notional[j] / leverage \
                    + prev_side[j] * (notional[j] - entry_notional[j])
                equity = equity - residual
                notional[j] = 0.0
                entry_notional[j] = 0.0
                prev_side[j] = 0.0
            elif sides[i, j] != prev_side[j] and prev_side[j] != 0.0:
                equity = equity - trading_costs * notional[j]
                notional[j] = 0.0
                entry_notional[j] = 0.0
                prev_side[j] = 0.0

        if equity <= 0.0:
            wiped = True
            equity = 0.0
            continue

        # ---- Phase 3: entries, all sized off the same post-exit equity ----
        equity_mid = equity
        for j in range(n_symbols):
            new_side = sides[i, j]
            if new_side != 0.0 and new_side != prev_side[j] and not liquidations[i, j]:
                notional[j] = weights[i, j] * equity_mid * leverage / (1.0 + trading_costs * new_side)
                entry_notional[j] = notional[j]
                equity = equity - trading_costs * notional[j]
                prev_side[j] = new_side

        if equity <= 0.0:
            wiped = True
            equity = 0.0
            continue

        equity_arr[i] = equity
        for j in range(n_symbols):
            notionals_out[i, j] = notional[j]

    return equity_arr, notionals_out
