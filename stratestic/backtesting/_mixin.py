import json
import logging
import math
import os
from typing import Literal

import numpy as np
import pandas as pd
import progressbar
import plotly.io as pio

from stratestic.backtesting.helpers import Trade
from stratestic.backtesting.helpers.evaluation import (
    get_results,
    log_results,
    BUY_AND_HOLD,
    CUM_SUM_STRATEGY,
    CUM_SUM_STRATEGY_TC,
    MARGIN_RATIO,
    SIDE, STRATEGY_RETURNS_TC, STRATEGY_RETURNS
)
from stratestic.backtesting.helpers.plotting import plot_backtest_results
from stratestic.backtesting.optimization import strategy_optimizer, adapt_optimization_input, get_params_mapping, \
    optimization_options_factor, optimization_options
from stratestic.utils.config_parser import get_config
from stratestic.utils.exceptions import SymbolInvalid, LeverageInvalid
from stratestic.utils.logger import configure_logger

config_vars = get_config('general')

configure_logger(config_vars.logger_level)

pio.renderers.default = "browser"


class BacktestMixin:
    """A Mixin class for backtesting trading strategies.

    Attributes:
    -----------
    symbol : str
        The trading symbol used for the backtest.
    tc : float
        The transaction costs (e.g. spread, commissions) as a percentage.
    results : pandas.DataFrame
        A DataFrame containing the results of the backtest.

    Methods:
    --------
    run(params=None, print_results=True, plot_results=True, plot_positions=False):
        Runs the trading strategy and prints and/or plots the results.
    optimize(params, **kwargs):
        Optimizes the trading strategy using brute force.
    _test_strategy(params=None, print_results=True, plot_results=True, plot_positions=False):
        Tests the trading strategy on historical data.
    _assess_strategy(data, title, print_results=True, plot_results=True, plot_positions=True):
        Assesses the performance of the trading strategy on historical data.
    plot_results(title, plot_positions=True):
        Plots the performance of the trading strategy compared to a buy and hold strategy.
    _gen_repeating(s):
        A generator function that groups repeated elements in an iterable.
    plot_func(ax, group):
        A function used for plotting positions.
    """
    def __init__(
        self,
        symbol,
        amount,
        trading_costs,
        leverage=1,
        margin_threshold=0.9,
        exchange='binance'
    ):
        """
        Initialize the BacktestMixin object.

        Parameters:
        -----------
        symbol : str
            The trading symbol to use for the backtest.
        amount : float
            The initial amount of capital to allocate for the backtest.
        trading_costs : float
            The transaction costs (e.g., spread, commissions) as a percentage.
        include_margin : bool, optional
            Flag indicating whether margin trading is included in the backtest. Default is False.
        leverage : float, optional
            The initial leverage to apply for margin trading. Default is 1.
        margin_threshold : float, optional
            The margin ratio threshold for margin call detection. Default is 0.8.
        exchange : str, optional
            The exchange to simulate the backtest on. Default is 'binance'.

        Raises:
        -------
        SymbolInvalid:
            If the specified trading symbol is not found in the leverage brackets.

        Notes:
        ------
        If `include_margin` is set to True, the leverage brackets for the specified symbol
        will be loaded.
        """
        self.exchange = exchange

        self.include_margin = False
        self.leverage_limits = [1, 125]

        self.set_leverage(leverage)
        self.set_margin_threshold(margin_threshold)

        self.amount = amount
        self.symbol = symbol
        self.tc = trading_costs / 100
        self.strategy = None

        self.perf = 0
        self.outperf = 0
        self.results = None

        if self.leverage != 1:
            self.include_margin = True
            self._load_leverage_brackets()

        self.bar = None
        self.optimization_steps = 0

        self._optimizer = None

    def __getattr__(self, attr):
        """
        Overrides the __getattr__ method to get attributes from the trading strategy object.

        Parameters
        ----------
        attr : str
            The attribute to be retrieved.

        Returns
        -------
        object
            The attribute object.
        """
        try:
            method = getattr(self.strategy, attr)
            return method
        except AttributeError:
            return getattr(self, attr)

    def __repr__(self):
        extra_title = (f"<b>Initial Amount</b> = {self.amount} | "
                       f"<b>Trading Costs</b> = {self.tc * 100}% | "
                       f"<b>Leverage</b> = {self.leverage}")
        return extra_title + '<br>' + self.strategy.__repr__()

    def set_leverage(self, leverage):
        if isinstance(leverage, int) and self.leverage_limits[0] <= leverage <= self.leverage_limits[1]:
            self.leverage = leverage
        else:
            raise LeverageInvalid(leverage)

    def set_margin_threshold(self, margin_threshold):
        if isinstance(margin_threshold, (int, float)) and 0 < margin_threshold <= 1:
            self.margin_threshold = margin_threshold
        else:
            raise ValueError('Margin threshold must be between 0 and 1.')

    def load_data(self, data=None, csv_path=None):
        """
        Loads market data into the trading strategy from a pandas DataFrame or a CSV file.
        If no data source is explicitly provided, it attempts to load data from a default CSV
        file path specified in the configuration.

        Parameters
        ----------
        data : pd.DataFrame, optional
            A pandas DataFrame containing market data. Expected to have a 'date' column as
            he index and OHLCV columns. If provided, this data is used directly.
        csv_path : str, optional
            The file path to a CSV file containing market data. The CSV is expected to have
            a 'date' column that will be used as the index, along with OHLCV columns.
            If `csv_path` is provided, it takes precedence over `data`.

        Notes
        -----
        - If both `data` and `csv_path` are None, the method attempts to load the data from
        a default CSV file path specified by `config_vars.ohlc_data_file`.
        - The method ensures there are no duplicated indices in the loaded data by keeping
        the last occurrence of any duplicated 'date' index.
        - This method sets the loaded data as the current dataset for the strategy
        and makes a copy of the original data for internal use.

        Raises
        ------
        FileNotFoundError
            If a `csv_path` is specified but the file cannot be found at the provided path.
        pd.errors.EmptyDataError
            If the CSV file is empty or if it contains only headers without any data rows.

        Examples
        --------
        >>> strategy_instance.load_data(csv_path='path/to/your/data.csv')
        This will load the market data from the specified CSV file and prepare it for the strategy.

        >>> strategy_instance.load_data(data=your_dataframe)
        This will directly use the provided DataFrame as the market data for the strategy.
        """

        if data is None or csv_path:
            default_file_path = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    '..',
                    config_vars.ohlc_data_file
                )
            )
            csv_path = csv_path if csv_path else default_file_path
            data = pd.read_csv(csv_path, index_col='date', parse_dates=True)
            data = data[~data.index.duplicated(keep='last')]  # remove duplicates

        self._original_data = data

        self.set_data(data.copy(), self.strategy)

    def run(self, print_results=True, plot_results=True, leverage=None):
        """
        Executes the trading strategy, evaluates its performance, and optionally prints and plots the results.
        The method supports applying leverage to the trading strategy to assess its impact on performance.

        Parameters
        ----------
        print_results : bool, optional
            If True (default), the performance summary of the trading strategy is printed to the console.
            The summary typically includes key performance indicators such as net profit, Sharpe ratio,
            maximum drawdown, and other relevant metrics.
        plot_results : bool, optional
            If True (default), the method generates plots comparing the trading strategy's performance
            against a benchmark, such as a "buy and hold" strategy. These plots can include equity curves,
            drawdown periods, and other performance metrics visualizations.
        leverage : int or float, optional
            The leverage level to apply to the trading strategy during the backtest. This parameter
            adjusts the size of positions taken by the strategy proportionally. If None (default),
            no leverage is applied. Note that using leverage can significantly increase both potential
            returns and potential risks.

        Returns
        -------
        None

        Notes
        -----
        - The method internally calls a private method `_test_strategy` to perform the backtest,
        which should implement the actual backtesting logic, including applying leverage,
        calculating performance metrics, and handling plotting.
        - The performance metrics (`perf`), outperformance metrics (`outperf`), and any additional
        results (`results`) are stored as attributes of the strategy instance after the method completes.
        This allows for further analysis or external reporting.

        Examples
        --------
        >>> strategy_instance.run(print_results=True, plot_results=True, leverage=2)
        This example runs the strategy with 2x leverage and both prints and plots the results.

        Raises
        ------
        ValueError
            If an invalid `leverage` value is provided (e.g., negative numbers).
        """

        if leverage is not None:
            self.set_leverage(leverage)

        perf, outperf, results = self._test_strategy(print_results=print_results, plot_results=plot_results)

        self.perf = perf
        self.outperf = outperf
        self.results = results

    def optimize(
        self,
        params,
        optimization_metric="Return",
        optimizer: Literal["brute_force", "gen_alg"] = 'brute_force',
        **kwargs
    ):
        """Optimizes the trading strategy using brute force.

        Parameters:
        -----------
        params : dict, list
            A dictionary or list (for strategy combination) containing the parameters to optimize.
            The parameters must be given as the keywords of a dictionary, and the value is an array
            of the lower limit, upper limit and step, respectively.

            Example for single strategy:
                params = dict(window=(10, 20, 1))

            Example for multiple strategies
                params = [dict(window=(10, 20, 1)), dict(ma=(30, 50, 2)]
        optimization_metric : str, optional
            The metric by which to perform the optimization. Option between:
            return_pct, sharpe_ratio, sortino_ratio, calmar_ratio, win_rate,
            profit_factor, sqn, expectancy, volatility_pct_annualized,
            max_drawdown, avg_drawdown, max_drawdown_duration
        optimizer : str, optional
            Choice of algorithm for the optimization.
        **kwargs : dict
            Additional arguments to pass to the `optimizing` function.

        Returns:
        --------
        opt : numpy.ndarray
            The optimal parameter values.
        -self._update_and_run(opt, plot_results=True) : float
            The negative performance of the strategy using the optimal parameter values.
        """

        self._check_metric_input(optimization_metric)
        self._check_optimizer_input(optimizer)

        opt_params, strategy_params_mapping, optimization_steps = adapt_optimization_input(self.strategy, params)

        self.bar = progressbar.ProgressBar(
            max_value=optimization_steps if self._optimizer == 'brute_force' else progressbar.UnknownLength,
            redirect_stdout=True
        )
        self.optimization_steps = 0

        opt = strategy_optimizer(
            self._update_and_run,
            opt_params,
            (False, False, strategy_params_mapping, params),
            optimizer,
            **kwargs
        )

        if not isinstance(opt, (list, tuple, type(np.array([])))):
            opt = np.array([opt])

        self.bar.finish()
        print()

        optimized_params = get_params_mapping(self.strategy, opt, strategy_params_mapping, params)
        optimized_result = self._update_and_run(opt, True, True, strategy_params_mapping, params)
        optimized_result = optimized_result if self._optimizer == 'gen_alg' else -optimized_result

        return optimized_params, optimized_result

    def maximum_leverage(self, margin_threshold=None):
        """
        Find the maximum allowable leverage that keeps the margin ratio below a specified threshold.

        Parameters:
        -----------
        threshold: float, optional
            The desired maximum margin ratio threshold, must be between 0 and 1.

        Raises:
        -----------
        ValueError: If the provided threshold is not within the valid range (0, 1].

        Returns:
        -----------
        int:
            The maximum allowable leverage that satisfies the specified margin ratio threshold.
        """
        initial_leverage = self.leverage
        initial_margin_threshold = self.margin_threshold
        initial_include_margin = self.include_margin

        if margin_threshold is not None:
            self.set_margin_threshold(margin_threshold)

        self.include_margin = True
        self._load_leverage_brackets()

        left_limit, right_limit = self.leverage_limits

        prev_leverage = 0

        print('\tCalculating maximum allowed leverage', end='')

        i = 0
        while True:
            leverage = math.floor((right_limit - left_limit) / 2) + left_limit

            if leverage == prev_leverage:
                break

            self._test_strategy(leverage=leverage, print_results=False, plot_results=False)
            if self.processed_data[MARGIN_RATIO].max() >= self.margin_threshold:
                right_limit = leverage
            else:
                left_limit = leverage

            prev_leverage = leverage
            i += 1
            print('.', end='')

        logging.info('')

        self.set_leverage(initial_leverage)
        self.set_margin_threshold(initial_margin_threshold)
        self.include_margin = initial_include_margin

        return leverage

    def _test_strategy(self, params=None, leverage=None, print_results=True, plot_results=True, show_plot_no_tc=False):
        """Tests the trading strategy on historical data.

        Parameters:
        -----------
        params : dict or None
            The parameters to use for the trading strategy
        """
        if leverage is not None:
            self.set_leverage(leverage)

        self._fix_original_data()

        self._set_index_frequency()

        self.set_parameters(params, data=self._original_data.copy())

    def _check_metric_input(self, optimization_metric):
        if optimization_metric not in optimization_options:
            raise ValueError(f"The chosen metric is not supported. "
                             f"Choose one of: {', '.join(optimization_options.keys())}")
        else:
            self.optimization_metric = optimization_options[optimization_metric]

    def _check_optimizer_input(self, optimizer):
        optimizer_options = ["brute_force", "gen_alg"]
        if optimizer not in optimizer_options:
            raise ValueError(f"The selected optimizer is not supported. "
                             f"Choose one of: {', '.join(optimizer_options)}")
        else:
            self._optimizer = optimizer

    def _set_index_frequency(self):
        self.index_frequency = self._original_data.index.inferred_freq

        if self.index_frequency is None:
            data_index = self._original_data.index
            frequencies = []
            for i in range(7):
                index = np.random.randint(0, len(data_index) - 1)
                frequencies.append(data_index[index + 1] - data_index[index])

            self.index_frequency = max(set(frequencies), key=frequencies.count)

    def _sanitize_equity(self, df, trades):

        if len(trades) == 0:
            return df

        trades_df = pd.DataFrame(trades)

        # Bring equity to 0 if a trade has gotten to zero equity
        no_funds_left_index = trades_df[trades_df["equity"].le(0)]["exit_date"]
        if len(no_funds_left_index) > 0:
            df.loc[no_funds_left_index.iloc[0]:, 'equity'] = 0

        # Bring equity to 0 if a margin call happens
        if self.include_margin:
            no_funds_left_index = df[df[MARGIN_RATIO] >= 1].index

            if len(no_funds_left_index) > 0:
                df.loc[no_funds_left_index[0]:, 'equity'] = 0

        return df

    @staticmethod
    def _sanitize_trades(data, trades):
        no_equity = data[CUM_SUM_STRATEGY_TC][data[CUM_SUM_STRATEGY_TC].le(0)].index

        if len(no_equity) == 0:
            return trades

        trades_df = pd.DataFrame(trades).set_index('entry_date')

        trades_df = trades_df[trades_df.index < no_equity[0]].copy()

        trades_df["pnl"] = np.where(trades_df["pnl"] < -1, -1, trades_df["pnl"])

        return [Trade(**row) for _, row in trades_df.reset_index().iterrows()]

    def _sanitize_margin_ratio(self, df):
        df[MARGIN_RATIO] = np.where(df[MARGIN_RATIO] > 1, 1, df[MARGIN_RATIO])
        df[MARGIN_RATIO] = np.where(df[MARGIN_RATIO] < 0, 1, df[MARGIN_RATIO])
        df[MARGIN_RATIO] = np.where(df[SIDE] == 0, 0, df[MARGIN_RATIO])

        df[MARGIN_RATIO] = df[MARGIN_RATIO].fillna(0)

        greater_than_index = df[df[MARGIN_RATIO].ge(1)].index.shift(1, freq=self.index_frequency)

        if len(greater_than_index) > 0:
            df.loc[greater_than_index[0]:, MARGIN_RATIO] = np.nan

        return df

    def _calculate_strategy_returns(self, df):
        df[STRATEGY_RETURNS_TC] = np.log(df['equity'] / df['equity'].shift(1)).fillna(0)
        df[STRATEGY_RETURNS] = df[STRATEGY_RETURNS_TC] + df["trades"] * self.tc
        df.loc[df.index[0], STRATEGY_RETURNS] = 0

        return df

    def _calculate_cumulative_returns(self, data):
        data[BUY_AND_HOLD] = data[self._returns_col].cumsum().apply(np.exp).fillna(1)
        data[CUM_SUM_STRATEGY_TC] = data[STRATEGY_RETURNS_TC].cumsum().apply(np.exp).fillna(1)

        if STRATEGY_RETURNS in data.columns:
            data[CUM_SUM_STRATEGY] = data[STRATEGY_RETURNS].cumsum().apply(np.exp).fillna(1)

        return data

    def _get_results(self, trades, processed_data):

        processed_data["close_date"] = processed_data.index.shift(1, freq=self.index_frequency)

        return get_results(
            processed_data,
            trades,
            self.leverage,
            self.amount,
            self.tc,
            config_vars.trading_days
        )

    @staticmethod
    def print_results(results, print_results):
        if not print_results:
            return

        log_results(results)

    def plot_results(self, processed_data, plot_results=True, show_plot_no_tc=False):
        """
        Plot the performance of the trading strategy compared to a buy and hold strategy.

        Parameters:
        -----------
        processed_data: pd.DataFrame
            Dataframe containing the results of the backtest to be plotted.
        plot_results: boolean, default True
            Whether to plot the results.
        show_plot_no_tc: boolean, default False
            Whether to show the plot of the equity curve with no trading costs
        """

        columns = [
            BUY_AND_HOLD,
            CUM_SUM_STRATEGY,
            CUM_SUM_STRATEGY_TC,
        ]

        data = processed_data.copy()[columns] * self.amount

        if self.include_margin:
            data[MARGIN_RATIO] = processed_data[MARGIN_RATIO].copy() * 100

        trades_df = pd.DataFrame(self.trades)

        if plot_results:
            nr_strategies = len([col for col in processed_data.columns if SIDE in col])
            offset = max(0, nr_strategies - 2)

            title = self.__repr__()

            plot_backtest_results(
                data,
                trades_df,
                self.margin_threshold,
                self.index_frequency,
                offset,
                plot_margin_ratio=self.include_margin,
                show_plot_no_tc=show_plot_no_tc,
                title=title
            )

    def _update_and_run(self, parameters, *args):
        """
        Update the hyperparameters of the strategy with the given `args`,
        and then run the strategy with the updated parameters.
        The strategy is run by calling the `_test_strategy` method with the
        updated parameters.

        Parameters
        ----------
        parameters : array-like
            A list of hyperparameters to be updated in the strategy.
            The order of the elements in the list should match the order
            of the strategy's hyperparameters, as returned by `self.params`.
        plot_results : bool, optional
            Whether to plot the results of the strategy after running it.

        Returns
        -------
        float
            The negative value of the strategy's score obtained with the
            updated hyperparameters. The negative value is returned to
            convert the maximization problem of the strategy's score into
            a minimization problem, as required by optimization algorithms.

        Raises
        ------
        IndexError
            If the number of elements in `parameters` does not match the number
            of hyperparameters in the strategy.

        Notes
        -----
        This method is intended to be used as the objective function to
        optimize the hyperparameters of the strategy using an optimization
        algorithm. It updates the hyperparameters of the strategy with the
        given `parameters`, then runs the strategy with the updated parameters,
        and returns the negative of the score obtained by the strategy.
        The negative is returned to convert the maximization problem of the
        strategy's score into a minimization problem, as required by many
        optimization algorithms.
        """
        print_results, plot_results, strategy_params_mapping, optimization_params = args

        test_params = get_params_mapping(self.strategy, parameters, strategy_params_mapping, optimization_params)

        results = self._test_strategy(test_params, print_results=print_results, plot_results=plot_results)

        self.optimization_steps += 1

        try:
            self.bar.update(self.optimization_steps)
        except ValueError:
            pass

        result = results[2][self.optimization_metric] if results[2] is not None else -np.inf

        result = result * optimization_options_factor[self.optimization_metric]

        if self._optimizer == 'gen_alg':
            result = -result

        return result

    def _fix_original_data(self):
        if self._original_data is None:
            self._original_data = self.strategy.data.copy()

            position_columns = [col for col in self._original_data if SIDE in col]

            self._original_data = self._original_data.drop(columns=position_columns)

    def _load_leverage_brackets(self):

        filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', config_vars.leverage_brackets_file))

        with open(filepath, 'r') as f:
            data = json.load(f)

        brackets = {symbol["symbol"]: symbol["brackets"] for symbol in data}

        try:
            self._symbol_bracket = pd.DataFrame(brackets[self.symbol])
        except KeyError:
            raise SymbolInvalid(self.symbol)
