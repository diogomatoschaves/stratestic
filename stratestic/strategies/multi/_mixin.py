import numpy as np
import pandas as pd

from stratestic.backtesting.helpers.evaluation import SIDE
from stratestic.strategies._mixin import StrategyMixin
from stratestic.utils.panel import panel_symbols, validate_panel


class MultiSymbolStrategyMixin(StrategyMixin):
    """
    Base class for multi-symbol (cross-sectional) strategies operating on a
    panel: a DataFrame with (symbol, field) MultiIndex columns (see
    stratestic.utils.panel.build_panel).

    A subclass must implement:
    - update_data(data): call super().update_data(data), then compute its
      per-symbol indicator columns, e.g. data[(symbol, "SMA")] = ...
    - calculate_positions(data): write (symbol, 'side') in {-1, 0, 1} for
      EVERY symbol in self.symbols (vectorized path).
    - get_signal(row=None): return {symbol: side} for a panel row
      (iterative path); row[symbol] is that symbol's cross-section.

    Optionally, for dynamic position sizing:
    - calculate_weights(data): write (symbol, 'weight') target fractions
      of equity * leverage, read on each symbol's trade-entry bars. The
      active-position weights of a bar must sum to at most 1.
    - get_weights(row=None): the per-row counterpart, {symbol: weight}.
    When absent, capital is split equally across active positions.
    """

    is_multi_symbol = True

    def __init__(
        self,
        data=None,
        trade_on_close=True,
        close_col='close',
        open_col='open',
        high_col='high',
        low_col='low',
        returns_col='returns'
    ):
        self.symbols = []

        StrategyMixin.__init__(
            self, data, trade_on_close, close_col, open_col, high_col, low_col, returns_col
        )

    def update_data(self, data) -> pd.DataFrame:
        """
        Validates the panel, captures the symbol order, and computes the
        per-symbol returns columns. Subclasses call this via super() and
        then add their indicator columns.
        """
        validate_panel(data, required_fields=(self._close_col,))

        data = data[~data.index.duplicated(keep='first')].copy()

        if not data.index.is_monotonic_increasing:
            data = data.sort_index()

        self.symbols = panel_symbols(data)

        data = self._calculate_returns(data)

        self._original_data = data.copy()

        return data

    def _calculate_returns(self, data) -> pd.DataFrame:
        """Per-symbol returns, with the same formula as the scalar mixin."""
        for symbol in self.symbols:
            price = data[(symbol, self._price_col)]

            if self._trade_on_close:
                data[(symbol, self._returns_col)] = np.log(price / price.shift(1))
            else:
                returns = np.log(price.shift(-1) / price)
                returns.iloc[-1] = np.log(
                    data[(symbol, self._close_col)].iloc[-1] / data[(symbol, self._open_col)].iloc[-1]
                )
                data[(symbol, self._returns_col)] = returns

        return data

    def calculate_positions(self, data) -> pd.DataFrame:
        raise NotImplementedError(
            f"{type(self).__name__} must implement calculate_positions(data), "
            f"writing a (symbol, '{SIDE}') column for every symbol"
        )

    def get_signal(self, row=None):
        raise NotImplementedError(
            f"{type(self).__name__} must implement get_signal(row), "
            f"returning a dict of symbol -> side"
        )

    def calculate_weights(self, data) -> pd.DataFrame:
        """Optional vectorized sizing hook; default: equal weight (no columns)."""
        return data

    def get_weights(self, row=None):
        """Optional per-row sizing hook; default None -> equal weight."""
        return None
