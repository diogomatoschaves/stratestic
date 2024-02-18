import logging

import numpy as np

from stratestic.backtesting._mixin import BacktestMixin
from stratestic.backtesting.helpers import Trade
from stratestic.backtesting.helpers.evaluation import CUM_SUM_STRATEGY, CUM_SUM_STRATEGY_TC, BUY_AND_HOLD, \
    STRATEGY_RETURNS_TC
from stratestic.backtesting.helpers.evaluation._constants import MARGIN_RATIO, SIDE
from stratestic.backtesting.helpers.margin import calculate_margin_ratio, get_maintenance_margin, calculate_liquidation_price
from stratestic.trading import Trader


class IterativeBacktester(BacktestMixin, Trader):
    """
    A class for backtesting trading strategies iteratively using historical data.
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
        Initializes the IterativeBacktester object.

        Parameters
        ----------
        strategy : object
            An instance of the trading strategy to be tested.
        symbol : str, optional
            The trading symbol. Default is None.
        amount : float, optional
            The initial amount of currency available for trading. Default is 1000.
        trading_costs : float, optional
            The percentage of trading costs (e.g., spread, commissions). Default is 0.
        include_margin : bool, optional
            Flag indicating whether margin trading is included in the backtest. Default is False.
        leverage : float, optional
            The initial leverage to apply for margin trading. Default is 1.
        margin_threshold : float, optional
            The margin ratio threshold for margin call detection. Default is 0.8.

        Notes
        -----
        The IterativeBacktester is initialized with a specified trading strategy, initial trading parameters,
        and optional margin trading features. It inherits from both BacktestMixin and Trader classes.

        If a trading symbol is provided, it is assigned to the trading strategy. The positions, _equity,
        margin_ratios, returns, strategy_returns, strategy_returns_tc, and positions attributes are initialized.

        Example
        -------
        >>> strategy = MyTradingStrategy()
        >>> backtester = IterativeBacktester(strategy, symbol='BTCUSDT', amount=5000, trading_costs=0.01)
        """

        Trader.__init__(self, amount)
        BacktestMixin.__init__(self, symbol, amount, trading_costs, leverage, margin_threshold)

        self.strategy = strategy

        if symbol is not None:
            self.strategy.symbol = symbol

        self.position = {
            symbol: 0
        }
        self.current_equity = self.amount
        self.initial_balance = amount * self.leverage
        self.current_balance = self.initial_balance
        self.equity = []
        self.margin_ratios = []
        self.returns = []

    def _get_position(self, symbol):
        """
        Gets the side for the given symbol.

        Parameters
        ----------
        symbol : str
            The trading symbol.

        Returns
        -------
        float
            The side value.
        """
        return self.position[symbol]

    def _reset_object(self, symbol):
        """
        Resets the object attributes to their initial values.
        """
        super()._reset_object(symbol)

        self.equity = []
        self.margin_ratios = []
        self.nr_trades = 0
        self.trades = []
        self.initial_balance = self.amount * self.leverage
        self.current_balance = self.initial_balance
        self.current_equity = self.amount
        self.units = 0

    def _get_price(self, _, row):
        """
        Gets the price for the given row.

        Parameters
        ----------
        _ : str
            Not used.
        row : pandas.Series
            The data row.

        Returns
        -------
        float
            The price.
        """
        price = row[self._close_col]

        return price

    def _get_high_low_price(self, side, row):
        """
        Gets the price for the given row.

        Parameters
        ----------
        side : int
            side of trade.
        row : pandas.Series
            The data row.

        Returns
        -------
        float
            The price.
        """
        price = row[self._high_col] if side == -1 else row[self._low_col]

        return price

    def _test_strategy(
        self,
        params=None,
        leverage=None,
        print_results=True,
        plot_results=True,
        show_plot_no_tc=False
    ):
        """
        Run a backtest for the given parameters and assess the performance of the strategy.

        Parameters
        ----------
        params : dict, optional
            Dictionary containing the keywords and respective values of the parameters to be updated.
        print_results: bool, optional
            Flag for whether to print the results of the backtest.
        plot_results : bool, optional
            Flag for whether to plot the results of the backtest.
        show_plot_no_tc: bool, optional
            Whether to plot the equity curve without the trading_costs applied

        Returns
        -------
        dict
            Dictionary containing the performance evaluation of the backtest.

        """
        super()._test_strategy(params, leverage)

        self._reset_object(self.symbol)

        data = self._get_data().dropna().copy()

        # title printout
        if print_results:
            logging.info("-" * 70)
            logging.info(self._get_test_title())
            logging.info("-" * 70)

        if data.empty:
            return 0, 0, None

        processed_data, trades = self._iterative_backtest(data, print_results)

        results, nr_trades, perf, outperf = self._evaluate_backtest(processed_data, trades)

        self.print_results(results, print_results)

        self.plot_results(self.processed_data, plot_results, show_plot_no_tc=show_plot_no_tc)

        return perf, outperf, results

    def _perform_iteration(self, data, print_results):
        equity = self.amount
        amount = self.amount * self.leverage

        for bar, (timestamp, row) in enumerate(data.iterrows()):

            signal = self.get_signal(row)

            previous_position = self._get_position(self.symbol)

            if bar != data.shape[0] - 1:
                self.trade(
                    self.symbol,
                    signal,
                    timestamp,
                    row,
                    amount="all",
                    print_results=print_results,
                    backtesting=True
                )
            else:
                self.close_pos(self.symbol, timestamp, row, print_results=print_results)
                self._set_position(self.symbol, 0)
                signal = 0

            trades = np.abs(signal - previous_position)

            if self.include_margin:
                new_trade = trades >= 1
                self._calculate_margin_ratio(row, new_trade)

            strategy_return = row[self._returns_col] * previous_position - trades * self.tc

            simple_return = np.exp(strategy_return) - 1

            pnl = simple_return * amount

            equity = equity + pnl

            self.equity.append(equity)

            if trades >= 1 and previous_position != 0:
                amount = equity * self.leverage
            else:
                amount = amount + pnl

    def _iterative_backtest(self, data, print_results=True):
        """
        Iterate through the data, trade accordingly, and calculate the strategy's performance.

        Parameters
        ----------
        data : pandas.DataFrame
            Historical data used to backtest the strategy.
        print_results: bool, optional
            Flag for whether to print the results of the backtest.

        """
        self._perform_iteration(data, print_results)

        data[SIDE] = self.positions[self.symbol][1:]
        data.loc[data.index[0], SIDE] = self.positions[self.symbol][1]
        data.loc[data.index[0], self._returns_col] = 0
        data.loc[data.index[0], STRATEGY_RETURNS_TC] = 0

        data["equity"] = self.equity

        if self.include_margin:
            data[MARGIN_RATIO] = self.margin_ratios
            data.loc[data.index[0], MARGIN_RATIO] = 0

            data = self._sanitize_margin_ratio(data)

        data = self._sanitize_equity(data, self.trades)
        data = self._add_trades_rows(data)
        data = self._calculate_strategy_returns(data)
        data = self._calculate_cumulative_returns(data)
        trades = self._sanitize_trades(data, self.trades)

        return data, trades

    @staticmethod
    def _add_trades_rows(data):
        data["trades"] = np.abs(data[SIDE].diff())
        data.loc[data.index[0], "trades"] = np.abs(data.loc[data.index[0], SIDE] - 0)

        return data

    def _calculate_margin_ratio(self, row, new_trade):
        try:
            trade = self.trades[-2 if new_trade else -1]
        except IndexError:
            self.margin_ratios.append(0)
            return

        mark_price = self._get_high_low_price(trade.side, row)

        margin_ratio = calculate_margin_ratio(
            self.leverage,
            trade.units,
            trade.side,
            trade.entry_price,
            mark_price,
            trade.maintenance_rate,
            trade.maintenance_amount,
            exchange=self.exchange
        )

        self.margin_ratios.append(margin_ratio)

    def _evaluate_backtest(self, data, trades):

        self.processed_data = data
        self.trades = trades

        perf = data[CUM_SUM_STRATEGY_TC].iloc[-1]  # Performance with trading_costs

        perf_bh = data[BUY_AND_HOLD].iloc[-1]

        outperf = perf - perf_bh

        results = self._get_results(self.trades, data.copy())

        return results, self.nr_trades, perf, outperf

    def buy_instrument(
        self,
        symbol,
        date=None,
        row=None,
        units=None,
        amount=None,
        open_trade=False,
        header='',
        **kwargs
    ):
        """
        Buys a specified amount of the instrument at the given date or row. If `units` is not specified, it calculates the
        number of units to buy based on the provided `amount` and the price of the instrument. It then calculates the trading cost
        based on the amount or number of units sold, and updates the `current_balance`, `units` and `trades` attributes
        accordingly. If `print_results` is set to True in `**kwargs`, it prints a message showing the date, number of units bought
        and the buying price.

        Parameters
        ----------
        symbol : str
            The symbol of the asset being traded.
        date : str, optional
            The date of the trade.
        row : pandas.Series, optional
            The row of the data being processed.
        units : float, optional
            The number of units to buy.
        amount : float, optional
            The amount of money to spend on the purchase.
        open_trade : boolean, optional
            A trade should be opened if True, and closed if False.
        header : str, optional
            The header of the message printed to the console.
        **kwargs : dict, optional
            Additional keyword arguments.

        """
        print_results = kwargs.get('print_results')
        reducing = kwargs.get("reducing")

        price = self._get_price(date, row)
        price_tc = price * (1 + self.tc)

        if units is None:
            units = amount / price_tc

        if amount is None:
            # The formula below comes from the computation: amount = 2 * prev_amount - new_amount
            # new_amount = prev_price / price * prev_amount, prev_amount = units * prev_price
            amount = self.trades[-1].entry_price * units * (2 - self.trades[-1].entry_price / price_tc)

        self.current_balance -= amount
        self.units += units

        if reducing:
            self._update_equity()

        self._handle_trade(self.trades, open_trade, date, price_tc, units, self.current_equity, self.current_balance, 1)

        if print_results:
            logging.info(f"{date} |  Buying {round(units, 4)} {self.symbol} for {round(price_tc, 5)}")

    def sell_instrument(
        self,
        symbol,
        date=None,
        row=None,
        units=None,
        amount=None,
        open_trade=False,
        header='',
        **kwargs
    ):
        """
        Sells a specified amount of the instrument at the given date or row. If `units` is not specified, it calculates the
        number of units to sell based on the provided `amount` and the price of the instrument. It then calculates the trading cost
        based on the amount or number of units sold, and updates the `current_balance`, `units` and `trades` attributes
        accordingly. If `print_results` is set to True in `**kwargs`, it prints a message showing the date, number of units sold
        and the selling price.

        Parameters
        ----------
        symbol : str
            The symbol of the instrument to sell.
        date : str or None, optional
            The date to sell the instrument at, formatted as 'YYYY-MM-DD'. If None, row must be specified instead.
        row : pandas.Series, optional
            The row of the data being processed.
        units : float or None, optional
            The number of units to sell. If None, amount must be specified instead.
        amount : float or None, optional
            The total amount to use to buy units of the instrument. If None, units must be specified instead.
        open_trade : boolean, optional
            A trade should be opened if True, and closed if False.
        header : str, optional
            A header to print before the results.
        **kwargs : dict, optional
            Additional keyword arguments:
            - print_results : bool, optional
                Whether to print the results.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If both date and row are None or both units and amount are None.
        """
        print_results = kwargs.get('print_results')
        reducing = kwargs.get("reducing")

        price = self._get_price(date, row)
        price_tc = price * (1 - self.tc)

        if units is None:
            units = amount / price_tc

        if amount is None:
            amount = units * price_tc

        self.current_balance += amount
        self.units -= units

        if reducing:
            self._update_equity()

        self._handle_trade(self.trades, open_trade, date, price_tc, units, self.current_equity, self.current_balance, -1)

        if print_results:
            logging.info(f"{date} |  Selling {round(units, 4)} {self.symbol} for {round(price_tc, 5)}")

    def _update_equity(self):
        trade_initial_balance = self.current_equity * self.leverage
        pnl = self.current_balance - trade_initial_balance

        self.current_equity = self.current_equity + pnl
        self.current_balance = self.current_equity * self.leverage

    def close_pos(self, symbol, date=None, row=None, header='', **kwargs):
        """
        Closes the side of the specified instrument at the given date or row. If the number of units is less than or equal to
        zero, it buys the instrument to close the side, otherwise it sells it. It then calculates the performance of the
        trading account, updates the `current_balance`, `trades` and `units` attributes accordingly, and prints a message
        showing the current balance, net performance and number of trades executed.

        Parameters
        ----------
        symbol : str
            The symbol of the instrument to close the side for.
        date : str or None, optional
            The date to close the side at, formatted as 'YYYY-MM-DD'. If None, row must be specified instead.
        row : int or None, optional
            The row index to close the side at. If None, date must be specified instead.
        header : str, optional
            A header to print before the results.
        **kwargs : dict, optional
            Additional keyword arguments:
            - print_results : bool, optional
                Whether to print the results.

        Returns
        -------
        None
        """
        print_results = kwargs.get('print_results')

        if self.units != 0 and print_results:
            logging.info(70 * "-")
            logging.info("{} |  +++ CLOSING FINAL POSITION +++".format(date))

        if self.units < 0:
            self.buy_instrument(symbol, date, row, open_trade=False, reducing=True, units=-self.units)
        elif self.units > 0:
            self.sell_instrument(symbol, date, row, open_trade=False, reducing=True, units=self.units)

        perf = (self.current_balance - self.initial_balance) / self.initial_balance * 100

        if print_results:

            self.print_current_balance(date)

            logging.info("{} |  net performance (%) = {}".format(date, round(perf, 2)))
            logging.info("{} |  number of trades executed = {}".format(date, self.nr_trades))
            logging.info(70 * "-")

    def _handle_trade(self, trades, open_trade, date, price, units, equity, amount, side):
        if open_trade:

            trades.append(Trade(date, None, price, None, units, side))

            if self.include_margin:

                notional_value = units * price

                maintenance_rate, maintenance_amount = get_maintenance_margin(
                    self._symbol_bracket, [notional_value], exchange=self.exchange
                )

                liquidation_price = calculate_liquidation_price(
                    units,
                    price,
                    side,
                    self.leverage,
                    maintenance_rate,
                    maintenance_amount,
                    exchange=self.exchange
                )[0]

                trades[-1].liquidation_price = liquidation_price
                trades[-1].maintenance_rate = maintenance_rate[0]
                trades[-1].maintenance_amount = maintenance_amount[0]
        else:
            trades[-1].exit_date = date
            trades[-1].exit_price = price
            trades[-1].equity = equity
            trades[-1].amount = amount

            trades[-1].calculate_profit(trades[-2].equity if len(trades) >= 2 else self.amount)
            trades[-1].calculate_pnl_pct(self.leverage)

            self.nr_trades += 1
