import json
import logging
import math
import os

import numpy as np
import pandas as pd
import progressbar
from scipy.optimize import brute
import plotly.io as pio

from stratestic.backtesting.combining import StrategyCombiner
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
from stratestic.utils.config_parser import get_config
from stratestic.utils.exceptions import StrategyRequired, OptimizationParametersInvalid, SymbolInvalid, LeverageInvalid
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
        """Runs the trading strategy and prints and/or plots the results.

        Parameters:
        -----------
        print_results : bool
            If True, print the results of the backtest.
        plot_results : bool
            If True, plot the performance of the trading strategy compared to a buy and hold strategy.
        leverage : int
            the leverage to run the backtest with

        Returns:
        --------
        None
        """
        if leverage is not None:
            self.set_leverage(leverage)

        perf, outperf, results = self._test_strategy(print_results=print_results, plot_results=plot_results)

        self.perf = perf
        self.outperf = outperf
        self.results = results

    def optimize(self, params, **kwargs):
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
        **kwargs : dict
            Additional arguments to pass to the `brute` function.

        Returns:
        --------
        opt : numpy.ndarray
            The optimal parameter values.
        -self._update_and_run(opt, plot_results=True) : float
            The negative performance of the strategy using the optimal parameter values.
        """

        opt_params, strategy_params_mapping, optimization_steps = self._adapt_optimization_input(params)

        self.bar = progressbar.ProgressBar(max_value=optimization_steps, redirect_stdout=True)
        self.optimization_steps = 0

        opt = brute(
            self._update_and_run, opt_params,
            (False, False, strategy_params_mapping, params),
            finish=None,
            **kwargs
        )

        if not isinstance(opt, (list, tuple, type(np.array([])))):
            opt = np.array([opt])

        return (self._get_params_mapping(opt, strategy_params_mapping, params),
                -self._update_and_run(opt, True, True, strategy_params_mapping, params))

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

    def _set_index_frequency(self):
        self.index_frequency = self._original_data.index.inferred_freq

        if self.index_frequency is None:
            data_index = self._original_data.index
            frequencies = []
            for i in range(7):
                index = np.random.randint(0, len(data_index) - 1)
                frequencies.append(data_index[index + 1] - data_index[index])

            self.index_frequency = max(set(frequencies), key=frequencies.count)

    @staticmethod
    def _get_optimization_input(optimization_params, strategy):
        opt_params = []
        optimizations_steps = 1
        for param in strategy.params:

            if param not in optimization_params:
                continue

            param_value = getattr(strategy, f"_{param}")
            is_int = isinstance(param_value, int)
            is_float = isinstance(param_value, float)

            step = 1 if is_int else None

            limits = optimization_params[param] \
                if param in optimization_params \
                else (param_value, param_value + 1) if is_int or is_float \
                else None

            if limits is not None:
                params = (*limits, step) if step is not None else limits
                opt_params.append(params)

                optimizations_steps *= (limits[1] - limits[0])

        return opt_params, optimizations_steps

    def _adapt_optimization_input(self, params):

        if not self.strategy:
            raise StrategyRequired

        if isinstance(self.strategy, StrategyCombiner):
            if not isinstance(params, (list, tuple, type(np.array([])))):
                raise OptimizationParametersInvalid('Optimization parameters must be provided as a list'
                                                    ' of dictionaries with the parameters for each individual strategy')

            if len(params) != len(self.strategy.strategies):
                raise OptimizationParametersInvalid(f'Wrong number of parameters. '
                                                    f'Number of strategies is {len(self.strategy.strategies)}')

            opt_params = []
            strategy_params_mapping = []
            optimization_steps = 1
            for i, strategy in enumerate(self.strategy.strategies):
                strategy_params, opt_steps = self._get_optimization_input(params[i], strategy)
                opt_params.extend(strategy_params)
                strategy_params_mapping.append(len(strategy_params))

                optimization_steps *= opt_steps

            return opt_params, strategy_params_mapping, optimization_steps

        else:
            if not isinstance(params, dict):
                raise OptimizationParametersInvalid('Optimization parameters must be provided as a '
                                                    'dictionary with the parameters the strategy')

            strategy_params, optimization_steps = self._get_optimization_input(params, self.strategy)

            return strategy_params, None, optimization_steps

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

    def _get_params_mapping(self, parameters, strategy_params_mapping, optimization_params):
        if not isinstance(self.strategy, StrategyCombiner):
            strategy_params = [param for param in self.strategy.get_params().keys() if param in optimization_params]
            new_params = {strategy_params[i]: parameter for i, parameter in enumerate(parameters)}
        else:
            new_params = []

            j = -1
            for i, mapping in enumerate(strategy_params_mapping):
                params = {}
                strategy_params = list(self.strategy.get_params(strategy_index=i).keys())
                for k, j in enumerate(range(j + 1, j + 1 + mapping)):
                    params.update({strategy_params[k]: parameters[j]})

                new_params.append(params)

        return new_params

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

        test_params = self._get_params_mapping(parameters, strategy_params_mapping, optimization_params)

        result = self._test_strategy(test_params, print_results=print_results, plot_results=plot_results)

        self.optimization_steps += 1

        try:
            self.bar.update(self.optimization_steps)
        except ValueError:
            pass

        return -result[0]

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
