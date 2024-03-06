import numpy as np
import numba
from numba import jit

from stratestic.backtesting._mixin import BacktestMixin
from stratestic.backtesting.helpers import Trade
from stratestic.backtesting.helpers.evaluation import STRATEGY_RETURNS, STRATEGY_RETURNS_TC, BUY_AND_HOLD, \
    CUM_SUM_STRATEGY_TC, MARGIN_RATIO, SIDE
from stratestic.backtesting.helpers.margin import (
    get_maintenance_margin,
    calculate_liquidation_price,
    calculate_margin_ratio
)

np.seterr(divide='ignore')
np.seterr(invalid='ignore')


@jit(nopython=True, cache=True)
def process_leveraged_returns(
    equity_arr: np.array,
    pnl_arr: np.array,
    amount_arr: np.array,
    notional_value_arr: np.array,
    strategy_returns_tc_arr: np.array,
    amount: numba.uint32,
    leverage: numba.uint32
):
    """
    Process leveraged returns based on strategy returns and leverage.

    Parameters
    ----------
    equity_arr : np.ndarray
        Array to store equity values at each time step.
    pnl_arr : np.ndarray
        Array to store profit/loss values at each time step.
    amount_arr : np.ndarray
        Array to store leveraged amount at each time step.
    notional_value_arr : np.ndarray
        Array containing notional values at each time step.
    strategy_returns_tc_arr : np.ndarray
        Array containing strategy returns at each time step, adjusted for transaction costs.
    amount : np.uint32
        Initial amount of capital.
    leverage : np.uint32
        Leverage multiplier.

    Returns
    -------
    np.ndarray
        Array containing equity values after processing leveraged returns.
    """
    equity = amount
    notional_value = amount
    amount = amount * leverage
    for index in np.arange(equity_arr.shape[0]):

        pnl = (np.exp(strategy_returns_tc_arr[index]) - 1) * amount

        equity = equity + pnl
        if notional_value_arr[index] != notional_value:
            amount = equity * leverage
            notional_value = notional_value_arr[index]
        else:
            amount = amount + pnl

        pnl_arr[index] = pnl
        equity_arr[index] = equity
        amount_arr[index] = amount

    return equity_arr


class VectorizedBacktester(BacktestMixin):
    """ Class for vectorized backtesting.
    """

    def __init__(
        self,
        strategy,
        symbol=None,
        amount=1000,
        trading_costs=0.0,
        leverage=1,
        margin_threshold=0.8
    ):
        """
        Initializes the Backtester object.

        Parameters
        ----------
        strategy : StrategyType
            A valid strategy class as defined in the 'stratestic.strategies' __init__ file.
        symbol : str, optional
            The trading symbol. Default is None.
        amount : float, optional
            The initial amount of currency available for trading. Default is 1000.
        trading_costs : float
            The trading cost per trade as a percentage of the value being traded.
        leverage : float, optional
            The initial leverage to apply for margin trading. Default is 1.
        margin_threshold : float, optional
            The margin ratio threshold for margin call detection. Default is 0.8.

        Notes
        -----
        The Backtester is initialized with a specified trading strategy, initial trading parameters,
        and optional margin trading features. It inherits from the BacktestMixin class.

        If a trading symbol is provided, it is assigned to the trading strategy.

        Example
        -------
        >>> strategy = MyTradingStrategy()
        >>> backtester = VectorizedBacktester(strategy=strategy, symbol='BTCUSDT', amount=5000, trading_costs=0.01)
        """

        BacktestMixin.__init__(self, symbol, amount, trading_costs, leverage, margin_threshold)

        self.strategy = strategy

        if symbol is not None:
            self.strategy.symbol = symbol

    def _test_strategy(
        self,
        params=None,
        leverage=None,
        print_results=True,
        plot_results=True,
        show_plot_no_tc=False
    ):
        """
        Parameters
        ----------
        params : dict
            Dictionary containing the keywords and respective values of the parameters to be updated.
        print_results: bool, optional
            Flag for whether to print the results of the backtest.
        plot_results: boolean
            Flag for whether to plot the results of the backtest.
        show_plot_no_tc: bool, optional
            Whether to plot the equity curve without the trading_costs applied

        """
        super()._test_strategy(params, leverage)

        data = self._get_data().dropna().copy()

        if data.empty:
            return 0, 0, None

        data, trades = self._vectorized_backtest(data)

        results, nr_trades, perf, outperf = self._evaluate_backtest(data, trades)

        self.print_results(results, print_results)

        self.plot_results(self.processed_data, plot_results, show_plot_no_tc=show_plot_no_tc)

        return perf, outperf, results

    def _vectorized_backtest(self, data):
        """
        Assess the performance of the trading strategy on historical data.

        Parameters:
        -----------
        data : pandas.DataFrame
            Historical price data for the trading symbol. Pre sanitized.

        Returns:
        --------
        None
        """
        data = self.calculate_positions(data)
        data["trades"] = data.side.diff().fillna(0).abs()
        data.loc[data.index[0], "trades"] = np.abs(data.iloc[0][SIDE])
        data.loc[data.index[-1], "trades"] = np.abs(data.iloc[-2][SIDE])
        data.loc[data.index[-1], SIDE] = 0

        data["trades"] = data["trades"].astype('int')
        data[SIDE] = data[SIDE].astype('int')

        data[STRATEGY_RETURNS] = (data.side.shift(1) * data.returns).fillna(0)
        data[STRATEGY_RETURNS_TC] = (data[STRATEGY_RETURNS] - data["trades"] * self.tc).fillna(0)

        trades = self._retrieve_trades(data, self.tc)

        if self.include_margin:
            data = self._calculate_margin_ratio(self._trades_df, data)
            data = self._sanitize_margin_ratio(data)

        data = self.process_leveraged_returns(data, self._trades_df)
        data = self._sanitize_equity(data, trades)
        data = self._calculate_strategy_returns(data)
        data = self._calculate_cumulative_returns(data)
        trades = self._sanitize_trades(data, trades)

        return data, trades

    def process_leveraged_returns(self, df, trades_df):
        """
        Process leveraged returns based on strategy returns and leverage.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing trading data.
        trades_df : pd.DataFrame
            DataFrame containing trade information.

        Returns
        -------
        pd.DataFrame
            DataFrame with processed leveraged returns.
        """

        df.loc[df.index[0], self._returns_col] = 0

        if len(trades_df) == 0:
            df["equity"] = 0
            return df

        df_filter = (df.trades != 0) & (df.side != 0)

        df.loc[df[df_filter].index, 'notional_value'] = trades_df["equity"].shift(1).values
        df.loc[df.index[0], 'notional_value'] = self.amount
        df['notional_value'].ffill(inplace=True)

        df["pnl"] = np.nan
        df["equity"] = np.nan
        df["amount"] = np.nan

        equity = df["equity"].values
        amount = df["amount"].values
        pnl = df["pnl"].values
        notional_value = df["notional_value"].values
        strategy_returns_tc = df[STRATEGY_RETURNS_TC].values

        df["equity"] = process_leveraged_returns(
            equity,
            pnl,
            amount,
            notional_value,
            strategy_returns_tc,
            self.amount,
            self.leverage
        )

        df.drop(columns=['pnl', 'amount', 'notional_value'], inplace=True)

        return df

    def _retrieve_trades(self, processed_data, trading_costs=0):
        """
        Computes the trades made based on the input processed data and returns a list of Trade objects.

        Parameters
        ----------
        processed_data : pandas.DataFrame
            The DataFrame containing the processed data for the strategy backtest.
        trading_costs: float
            The trading costs as a raw percent value of each trade.

        Returns
        -------
        trades_list : list of Trade objects
            A list containing information about each trade made during the backtest, represented as Trade objects.
            Each Trade object has the following attributes:

            - entry_price (float): The price at which the trade was entered.
            - entry_date (datetime): The date at which the trade was entered.
            - exit_price (float): The price at which the trade was exited.
            - exit_date (datetime): The date at which the trade was exited.
            - side (int): The side of the trade (1 for long, -1 for short).
            - units (float): The number of units of the asset traded.

        """

        cols = [self._price_col, SIDE]

        processed_data = processed_data.copy()

        if not self._trade_on_close:
            processed_data[self._price_col] = processed_data[self._price_col].shift(-1)

        trades = processed_data[processed_data.trades != 0][cols]

        trades = trades.reset_index()

        col = list(set(trades.columns).difference(set(cols)))[0]

        trades = trades.rename(columns={self._price_col: "entry_price", col: "entry_date"})
        trades["exit_price"] = trades["entry_price"].shift(-1) * (1 - trading_costs * trades[SIDE])
        trades["entry_price"] = trades["entry_price"] * (1 + trading_costs * trades[SIDE])
        trades["exit_date"] = trades["entry_date"].shift(-1)
        trades = trades[trades.side != 0]

        trades["exit_price"] = np.where(
            np.isnan(trades['exit_price']),
            processed_data.loc[processed_data.index[-1], self._close_col],
            trades['exit_price']
        )

        trades = trades.reset_index(drop=True)
        trades = trades.dropna()

        trades["log_return"] = np.log(trades["exit_price"] / trades["entry_price"]) * trades[SIDE]
        trades["log_cum"] = trades["log_return"].cumsum().apply(np.exp)

        leveraged_amount = self.amount * self.leverage

        if len(trades) > 0:

            trades["pnl"] = None
            trades["equity"] = None
            trades["amount"] = None

            equity = self.amount
            for index, trade in trades.iterrows():
                pnl = (np.exp(trade["log_return"]) - 1) * self.leverage

                equity = equity * (1 + pnl)
                amount = equity * self.leverage

                trades.loc[index, 'pnl'] = pnl
                trades.loc[index, 'equity'] = equity
                trades.loc[index, 'amount'] = amount

            no_funds_left_index = trades["equity"][trades["equity"].le(0)].index
            if len(no_funds_left_index) > 0:
                trades.loc[no_funds_left_index[0]:, "equity"] = 0
                trades.loc[no_funds_left_index[0]:, "amount"] = 0
                trades.loc[no_funds_left_index[0] + 1:, "pnl"] = 0

            trades["units"] = (trades["amount"].shift(1) / trades["entry_price"]).fillna(leveraged_amount / trades["entry_price"][0])
            trades["profit"] = (trades["equity"] - trades["equity"].shift(1)).fillna(trades["equity"][0] - self.amount)

        columns_to_delete = ['log_return', 'log_cum']

        if self.include_margin and len(trades) > 0:
            trades['maintenance_rate'], trades['maintenance_amount'] = get_maintenance_margin(
                self._symbol_bracket,
                trades['units'] * trades['entry_price'],
                self.exchange
            )

            trades['liquidation_price'] = calculate_liquidation_price(
                trades['units'],
                trades['entry_price'],
                trades['side'],
                self.leverage,
                trades['maintenance_rate'],
                trades['maintenance_amount'],
                exchange=self.exchange
            )

        self._trades_df = trades.copy()

        trades.drop(columns_to_delete, axis=1, inplace=True)

        trades_list = [Trade(**row) for _, row in trades.iterrows()]

        return trades_list

    def _calculate_margin_ratio(self, trades_df, processed_data):

        df = processed_data.copy()

        if len(trades_df) == 0:
            df[MARGIN_RATIO] = 0
            return df

        df['entry_price'] = None
        df['units'] = None
        df['maintenance_rate'] = None
        df['maintenance_amount'] = None
        df['mark_price'] = np.where(df['side'].shift(1) == 1, df[self._low_col], df[self._high_col])

        df_filter = (df.trades != 0) & (df.side != 0)

        df.loc[df[df_filter].index, 'entry_price'] = trades_df['entry_price'].values
        df.loc[df[df_filter].index, 'units'] = trades_df['units'].values
        df.loc[df[df_filter].index, 'maintenance_rate'] = trades_df['maintenance_rate'].values
        df.loc[df[df_filter].index, 'maintenance_amount'] = trades_df['maintenance_amount'].values

        df['entry_price'].ffill(inplace=True)
        df['units'].ffill(inplace=True)
        df['maintenance_rate'].ffill(inplace=True)
        df['maintenance_amount'].ffill(inplace=True)

        df[MARGIN_RATIO] = calculate_margin_ratio(
            self.leverage,
            df['units'].shift(1),
            df['side'].shift(1),
            df['entry_price'].shift(1),
            df['mark_price'],
            df['maintenance_rate'].shift(1),
            df['maintenance_amount'].shift(1),
            exchange=self.exchange
        )

        df.drop(
            ['entry_price', 'units', 'mark_price', 'maintenance_rate', 'maintenance_amount'],
            axis=1, inplace=True
        )

        return df

    def _evaluate_backtest(self, processed_data, trades):
        """
       Evaluates the performance of the trading strategy on the backtest run.

       Parameters:
       -----------
       print_results : bool, default True
           Whether to print the results.

       Returns:
       --------
       float
           The performance of the strategy.
       float
           The out-/underperformance of the strategy.
       """

        self.processed_data = processed_data
        self.trades = trades

        nr_trades = self._get_nr_trades(processed_data)

        # absolute performance of the strategy
        perf = processed_data[CUM_SUM_STRATEGY_TC].iloc[-1]

        # out-/underperformance of strategy
        outperf = perf - processed_data[BUY_AND_HOLD].iloc[-1]

        results = self._get_results(self.trades, processed_data.copy())

        return results, nr_trades, perf, outperf

    @staticmethod
    def _get_nr_trades(data):
        return int(data["trades"].sum() / 2) + 1
