from collections import OrderedDict

import numpy as np
import pandas as pd

from stratestic.backtesting.helpers.evaluation import SIDE, WEIGHT
from stratestic.strategies.multi import MultiSymbolStrategyMixin
from stratestic.utils.panel import build_panel


def make_flat_ohlc(prices, freq="1h", start="2023-01-01"):
    """OHLCV frame where every bar's open/high/low/close equal the price."""
    index = pd.date_range(start, periods=len(prices), freq=freq, tz="UTC")
    prices = pd.Series(prices, index=index, dtype="float64")
    return pd.DataFrame(
        {"open": prices, "high": prices, "low": prices, "close": prices, "volume": 1.0}
    )


def make_random_ohlc(seed, n=120, scale=0.01, start_price=100.0):
    rng = np.random.RandomState(seed)
    prices = start_price * np.exp(np.cumsum(rng.normal(0, scale, n)))
    return make_flat_ohlc(prices)


def make_random_panel(n=120):
    return build_panel({
        "BTCUSDT": make_random_ohlc(1, n=n, scale=0.01),
        "ETHUSDT": make_random_ohlc(2, n=n, scale=0.015, start_price=50.0),
        "SOLUSDT": make_random_ohlc(3, n=n, scale=0.02, start_price=10.0),
    })


class PresetSidesPanel(MultiSymbolStrategyMixin):
    """Panel test strategy emitting fixed, repeating per-symbol side patterns
    (and optionally fixed per-symbol weights)."""

    def __init__(self, patterns, weights=None, data=None, **kwargs):
        self.params = OrderedDict()
        self._patterns = patterns
        self._weights = weights
        MultiSymbolStrategyMixin.__init__(self, data, **kwargs)

    def update_data(self, data):
        data = super().update_data(data)
        self._sides = {
            symbol: pd.Series(
                [pattern[i % len(pattern)] for i in range(len(data))], index=data.index
            )
            for symbol, pattern in self._patterns.items()
        }
        return data

    def calculate_positions(self, data):
        for symbol in self.symbols:
            data[(symbol, SIDE)] = self._sides[symbol].reindex(data.index).values
        return data

    def get_signal(self, row=None):
        return {symbol: int(self._sides[symbol].loc[row.name]) for symbol in self.symbols}

    def calculate_weights(self, data):
        if self._weights is not None:
            for symbol, weight in self._weights.items():
                data[(symbol, WEIGHT)] = weight
        return data

    def get_weights(self, row=None):
        return None if self._weights is None else dict(self._weights)
