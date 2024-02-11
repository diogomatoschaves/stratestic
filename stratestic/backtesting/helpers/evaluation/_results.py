import logging

import pandas as pd

from stratestic.backtesting.helpers.evaluation.metrics import *
from stratestic.backtesting.helpers.evaluation._constants import (
    results_sections,
    results_mapping,
    results_aesthetics,
    SIDE,
    CLOSE_DATE,
    CUM_SUM_STRATEGY_TC,
    STRATEGY_RETURNS_TC,
    BUY_AND_HOLD
)


def get_results(data, trades, leverage, amount, trading_costs=None, trading_days=365):

    results = {}

    results = get_overview_results(results, data, leverage, trading_costs, amount)

    results = get_returns_results(results, data, amount, trading_days)

    results = get_drawdown_results(results, data)

    results = get_trades_results(results, trades)

    results = get_ratios_results(results, data, trades, trading_days)

    return pd.Series(results)


def get_overview_results(results, data, leverage, trading_costs, amount):
    results["total_duration"] = get_total_duration(data.index, data[CLOSE_DATE])
    results["start_date"] = get_start_date(data.index)
    results["end_date"] = get_end_date(data[CLOSE_DATE])

    results["leverage"] = leverage

    if trading_costs is not None:
        results["trading_costs"] = trading_costs * 100

    results["equity_initial"] = amount
    results["traded_amount"] = amount * leverage

    if SIDE in data:
        results["exposure_time"] = exposure_time(data[SIDE])

    return results


def get_returns_results(results, data, amount, trading_days):
    if CUM_SUM_STRATEGY_TC in data:
        results["equity_final"] = equity_final(data[CUM_SUM_STRATEGY_TC] * amount)
        results["equity_peak"] = equity_peak(data[CUM_SUM_STRATEGY_TC] * amount)
        results["return_pct"] = return_pct(data[CUM_SUM_STRATEGY_TC])
        results["return_pct_annualized"] = return_pct_annualized(data[CUM_SUM_STRATEGY_TC])

        if STRATEGY_RETURNS_TC in data:
            results["volatility_pct_annualized"] = volatility_pct_annualized(data[STRATEGY_RETURNS_TC], trading_days)

        if "accumulated_returns" in data:
            results["buy_and_hold_return"] = return_buy_and_hold_pct(data[BUY_AND_HOLD])

        return results


def get_drawdown_results(results, data):
    if CUM_SUM_STRATEGY_TC in data:
        results["max_drawdown"] = max_drawdown_pct(data[CUM_SUM_STRATEGY_TC])
        results["avg_drawdown"] = avg_drawdown_pct(data[CUM_SUM_STRATEGY_TC])
        results["max_drawdown_duration"] = max_drawdown_duration(data[CUM_SUM_STRATEGY_TC], data[CLOSE_DATE])
        results["avg_drawdown_duration"] = avg_drawdown_duration(data[CUM_SUM_STRATEGY_TC], data[CLOSE_DATE])

        return results


def get_trades_results(results, trades):
    results["nr_trades"] = int(len(trades))
    results["win_rate"] = win_rate_pct(trades)
    results["best_trade"] = best_trade_pct(trades)
    results["worst_trade"] = worst_trade_pct(trades)
    results["avg_trade"] = avg_trade_pct(trades)
    results["max_trade_duration"] = max_trade_duration(trades)
    results["avg_trade_duration"] = avg_trade_duration(trades)
    results["expectancy"] = expectancy_pct(trades)

    return results


def get_ratios_results(results, data, trades, trading_days):
    if CUM_SUM_STRATEGY_TC in data:
        results["calmar_ratio"] = calmar_ratio(data[CUM_SUM_STRATEGY_TC])

    if STRATEGY_RETURNS_TC in data:
        results["sharpe_ratio"] = sharpe_ratio(data[STRATEGY_RETURNS_TC], trading_days=trading_days)
        results["sortino_ratio"] = sortino_ratio(data[STRATEGY_RETURNS_TC])

    results["profit_factor"] = profit_factor(trades)
    results["sqn"] = system_quality_number(trades)

    return results


def log_results(results, backtesting=True):

    length = 60

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

            logging.info(f'{printed_title:<30}{value.rjust(30)}')
        logging.info('-' * length)
        logging.info('')
    logging.info('*' * length)
