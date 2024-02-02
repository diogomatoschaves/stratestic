import logging

from stratestic.backtesting.helpers.evaluation.metrics import *
from stratestic.backtesting.helpers.evaluation._constants import results_sections, results_mapping, results_aesthetics

import pandas as pd


CUM_STRATEGY_RETURNS = "accumulated_strategy_returns_tc"
CUM_RETURNS = "accumulated_returns"
STRATEGY_RETURNS = "strategy_returns_tc"
SIDE = "side"
CLOSE_DATE = "close_date"


def get_results(data, trades, leverage, amount, trading_costs=None, trading_days=365):

    results = {}

    results = get_overview_results(results, data, leverage, trading_costs, amount)

    results = get_returns_results(results, data, leverage, amount, trading_days)

    results = get_drawdown_results(results, data)

    results = get_trades_results(results, trades, leverage)

    results = get_ratios_results(results, data, trades, trading_days)

    return pd.Series(results)


def get_overview_results(results, data, leverage, trading_costs, amount):
    results["total_duration"] = get_total_duration(data.index, data[CLOSE_DATE])
    results["start_date"] = get_start_date(data.index)
    results["end_date"] = get_end_date(data[CLOSE_DATE])

    results["leverage"] = leverage

    if trading_costs is not None:
        results["trading_costs"] = trading_costs * 100

    results["initial_equity"] = amount
    results["exposed_capital"] = results["initial_equity"] / leverage

    if "side" in data:
        results["exposure_time"] = exposure_time(data[SIDE])

    return results


def get_returns_results(results, data, leverage, amount, trading_days):
    if CUM_STRATEGY_RETURNS in data:
        results["equity_final"] = equity_final(data[CUM_STRATEGY_RETURNS] * amount)
        results["equity_peak"] = equity_peak(data[CUM_STRATEGY_RETURNS] * amount)
        results["return_pct"] = return_pct(data[CUM_STRATEGY_RETURNS]) * leverage
        results["return_pct_annualized"] = return_pct_annualized(data[CUM_STRATEGY_RETURNS], leverage)

        if STRATEGY_RETURNS in data:
            results["volatility_pct_annualized"] = volatility_pct_annualized(data[STRATEGY_RETURNS], trading_days)

        if "accumulated_returns" in data:
            results["buy_and_hold_return"] = return_buy_and_hold_pct(data[CUM_RETURNS]) * leverage

        return results


def get_drawdown_results(results, data):
    if CUM_STRATEGY_RETURNS in data:
        results["max_drawdown"] = max_drawdown_pct(data[CUM_STRATEGY_RETURNS])
        results["avg_drawdown"] = avg_drawdown_pct(data[CUM_STRATEGY_RETURNS])
        results["max_drawdown_duration"] = max_drawdown_duration(data[CUM_STRATEGY_RETURNS], data[CLOSE_DATE])
        results["avg_drawdown_duration"] = avg_drawdown_duration(data[CUM_STRATEGY_RETURNS], data[CLOSE_DATE])

        return results


def get_trades_results(results, trades, leverage):
    results["nr_trades"] = int(len(trades))
    results["win_rate"] = win_rate_pct(trades)
    results["best_trade"] = best_trade_pct(trades, leverage)
    results["worst_trade"] = worst_trade_pct(trades, leverage)
    results["avg_trade"] = avg_trade_pct(trades, leverage)
    results["max_trade_duration"] = max_trade_duration(trades)
    results["avg_trade_duration"] = avg_trade_duration(trades)
    results["expectancy"] = expectancy_pct(trades, leverage)

    return results


def get_ratios_results(results, data, trades, trading_days):
    if CUM_STRATEGY_RETURNS in data:
        results["calmar_ratio"] = calmar_ratio(data[CUM_STRATEGY_RETURNS])

    if STRATEGY_RETURNS in data:
        results["sharpe_ratio"] = sharpe_ratio(data[STRATEGY_RETURNS], trading_days=trading_days)
        results["sortino_ratio"] = sortino_ratio(data[STRATEGY_RETURNS])

    results["profit_factor"] = profit_factor(trades)
    results["sqn"] = system_quality_number(trades)

    return results


def log_results(results, backtesting=True):

    length = 55

    logging.info("")

    title = 'BACKTESTING' if backtesting else 'TRADING BOT'

    logging.info('*' * length)
    logging.info(f'{title} RESULTS'.center(length))
    logging.info('*' * length)
    logging.info('')

    for section, columns in results_sections.items():

        logging.info(section.center(length))
        logging.info('-' * length)

        for col in columns:

            try:
                value = results[col]
            except KeyError:
                continue

            if callable(results_mapping[col]):
                printed_title = results_mapping[col]('USDT')
            else:
                printed_title = results_mapping[col]

            if col in results_aesthetics:
                value = results_aesthetics[col](value)
            else:
                try:
                    value = str(round(value, 2))
                except TypeError:
                    value = str(value)

            logging.info(f'{printed_title:<30}{value.rjust(25)}')
        logging.info('-' * length)
        logging.info('')
    logging.info('*' * length)
