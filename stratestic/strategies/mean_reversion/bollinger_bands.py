from collections import OrderedDict

import numpy as np

from stratestic.backtesting.helpers.evaluation import SIDE
from stratestic.strategies._mixin import StrategyMixin


class BollingerBands(StrategyMixin):
    """ Bollinger Bands Strategy:

    This strategy follows the principle of mean reversion, ie. that
    the price will revert back to the mean if it deviates by a certain amount,
    essentially going long or short when the price crosses a
    certain low and high threshold respectively.

    Parameters:
    -----------
    ma : int
        Moving average window.
    sd : float
        Number of standard deviations for the band width.
    data : pd.DataFrame, default None
        Data to use in the strategy.
    **kwargs
        Additional arguments to pass to the `StrategyMixin` superclass.

    Attributes:
    -----------
    params : OrderedDict
        Ordered dictionary containing the strategy's parameters:
        - `ma`: moving average window.
        - `sd`: number of standard deviations.

    Methods:
    --------
    __repr__(self)
        Return a string representation of the class instance.
    update_data(self)
        Retrieves and prepares the data.
    calculate_positions(self, data)
        Calculate the side for each row in the data.
    get_signal(self, row=None)
        Return the side signal for a given row.

    """

    def __init__(self, ma: int, sd: float, data=None, **kwargs):

        self._ma = ma
        self._sd = sd

        self.params = OrderedDict(
            ma=lambda x: int(x),
            sd=lambda x: float(x),
        )

        StrategyMixin.__init__(self, data, **kwargs)

    def __repr__(self):
        return "{}(symbol = {}, ma = {}, sd = {})".format(self.__class__.__name__, self.symbol, self._ma, self._sd)

    def update_data(self, data):
        """
        Updates the input data with additional columns required for the strategy.

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV data to be updated.

        Returns
        -------
        pd.DataFrame
            Updated OHLCV data containing additional columns.
        """
        data = super().update_data(data)

        rolling_std = data[self._close_col].rolling(self._ma).std()

        data["sma"] = data[self._close_col].rolling(self._ma).mean()
        data["upper"] = data["sma"] + rolling_std * self._sd
        data["lower"] = data["sma"] - rolling_std * self._sd

        return self.calculate_positions(data)

    def calculate_positions(self, data):
        data["distance"] = data[self._close_col] - data["sma"]
        data[SIDE] = np.where(data[self._close_col] > data["upper"], -1, np.nan)
        data[SIDE] = np.where(data[self._close_col] < data["lower"], 1, data[SIDE])
        data[SIDE] = np.where(data["distance"] * data["distance"].shift(1) < 0, 0, data[SIDE])
        data[SIDE] = data[SIDE].ffill().fillna(0)

        return data

    def get_signal(self, row=None):

        if row is None:
            row = self.data.iloc[-1]

        return int(row[SIDE])
