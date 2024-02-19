import numpy as np
import pandas as pd


class StrategyMixin:
    """
    A mixin class that provides foundational functionality for backtesting trading strategies,
    including data preparation and parameter management. It is designed to be extended
    by specific trading strategy implementations.

    Parameters
    ----------
    data : pd.DataFrame, optional
        The DataFrame containing the historical price data (OHLCV) for the asset.
    trade_on_close : bool, optional
        Indicates whether trades are executed at the close price of the current bar or the open price of the next bar.
        Default is True, meaning trades are executed at the close price.
    close_col : str, optional
        The name of the column in 'data' that contains the close price data. Default is 'close'.
    open_col : str, optional
        The name of the column in 'data' that contains the open price data. Default is 'open'.
    high_col : str, optional
        The name of the column in 'data' that contains the high price data. Default is 'high'.
    low_col : str, optional
        The name of the column in 'data' that contains the low price data. Default is 'low'.
    returns_col : str, optional
        The name of the column in 'data' that will contain the calculated returns data. Default is 'returns'.

    Attributes
    ----------
    data : pd.DataFrame
        The DataFrame containing the historical price data for the asset. It is updated with additional calculations necessary for the strategy.
    _close_col : str
        The name of the column containing the close price data.
    _open_col : str
        The name of the column containing the open price data.
    _high_col : str
        The name of the column containing the high price data.
    _low_col : str
        The name of the column containing the low price data.
    _trade_on_close : bool
        Indicator of whether trading actions are taken based on the close price of the current period or the open price of the next period.
    _price_col : str
        The column name used for price data in trading calculations, determined by 'trade_on_close'.
    _returns_col : str
        The name of the column containing the returns data.
    symbol : str
        The identifier for the asset being traded. This is typically set by subclasses.
    _original_data : pd.DataFrame
        A copy of the original input data, preserved for reference.

    Methods
    -------
    __init__(self, data=None, trade_on_close=True, close_col='close',
             open_col='open', high_col='high', low_col='low', returns_col='returns'):
        Initializes the StrategyMixin instance with the provided parameters and
        prepares the data for trading strategy analysis.
    __repr__(self):
        Returns a string representation of the strategy, typically the class name.
    get_params(self, **kwargs):
        Retrieves the parameters for the strategy.
    _get_test_title(self):
        Generates a title for the backtest report based on the strategy details.
    _get_data(self) -> pd.DataFrame:
        Provides access to the current price data being used by the strategy.
    set_data(self, data: pd.DataFrame, strategy_obj=None) -> None:
        Updates the strategy's data with a new DataFrame, recalculating necessary
        components based on the new data.
    set_parameters(self, params=None, data=None) -> None:
        Allows for dynamic updating of strategy parameters and optionally updates the
        strategy data.
    _calculate_returns(self, data) -> pd.DataFrame:
        Calculates returns based on the specified price column and updates the data
        DataFrame with these calculations.
    update_data(self, data) -> pd.DataFrame:
        Prepares the input data for the strategy by calculating returns and performing
        any other necessary preprocessing steps.
    """

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
        """
        Initializes a new instance of the StrategyMixin class.

        Parameters
        ----------
        data : pd.DataFrame, optional
            The DataFrame containing the historical price data for the asset.
        close_col : str, optional
            The name of the column in the data that contains the close price data.
        open_col : str, optional
            The name of the column in the data that contains the open price data.
        high_col : str, optional
            The name of the column in the data that contains the high price data.
        low_col : str, optional
            The name of the column in the data that contains the low price data.
        returns_col : str, optional
            The name of the column in the data that will contain the returns' data.
        """

        self._close_col = close_col
        self._open_col = open_col
        self._high_col = high_col
        self._low_col = low_col
        self._trade_on_close = trade_on_close
        self._price_col = close_col if trade_on_close else open_col
        self._returns_col = returns_col
        self.symbol = None

        self._original_data = None

        if data is not None:
            self.data = self.update_data(data.copy())

    def __repr__(self):
        """
        Returns a string representation of the strategy.

        Returns:
        --------
        str:
            A string representation of the strategy.
        """
        return "{}".format(self.__class__.__name__)

    def get_params(self, **kwargs):
        return self.params if self.params else {}

    def _get_test_title(self):
        """
        Returns the title for the backtest report.

        Returns:
        --------
        str:
            The title for the backtest report.
        """
        return f"{self.__repr__()} strategy backtest."

    def _get_data(self) -> pd.DataFrame:
        """
        Returns the current DataFrame containing the historical price data for the asset.

        Returns
        -------
        pd.DataFrame
            The current DataFrame containing the historical price data for the asset.
        """

        return self.data

    def set_data(self, data: pd.DataFrame, strategy_obj=None) -> None:
        """
        Sets the DataFrame containing the historical price data for the asset.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame containing the historical price data for the asset.
        strategy_obj : Strategy object

        """

        if data is not None:
            if strategy_obj is not None:
                strategy_obj.data = strategy_obj.update_data(data.copy())
            else:
                self.data = self.update_data(self.data.copy())

    def set_parameters(self, params=None, data=None) -> None:
        """
        Updates the parameters of the strategy.

        Parameters
        ----------
        data
        params : dict, optional
            A dictionary containing the parameters to be updated.
        """

        if params is None:
            return

        strategy_params = self.get_params()

        for param, new_value in params.items():
            setattr(self, f"_{param}", strategy_params[param](new_value))

        data = data.copy() if data is not None \
            else self._original_data.copy() if self._original_data is not None \
            else self.data.copy()

        self.data = self.update_data(data)

    def _calculate_returns(self, data) -> pd.DataFrame:
        """
        Calculates the returns of the asset and updates the data DataFrame.
        """

        if self._trade_on_close:
            data[self._returns_col] = np.log(data[self._price_col] / data[self._price_col].shift(1))
        else:
            data[self._returns_col] = np.log(data[self._price_col].shift(-1) / data[self._price_col])
            data.loc[data.index[-1], self._returns_col] = \
                np.log(data.loc[data.index[-1], self._close_col] / data.loc[data.index[-1], self._open_col])

        return data

    def update_data(self, data) -> pd.DataFrame:
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
        data = data[~data.index.duplicated(keep='first')].copy()

        data = self._calculate_returns(data)

        self._original_data = data.copy()

        return data
