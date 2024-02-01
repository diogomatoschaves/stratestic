import logging

from stratestic.backtesting.helpers.evaluation.metrics import *
from stratestic.backtesting.helpers.evaluation._constants import results_sections, results_mapping, results_aesthetics

import pandas as pd


def get_results(processed_data, trades, leverage, amount, trading_costs=None, trading_days=365):

    results = {}

    results["total_duration"] = get_total_duration(processed_data.index)
    results["start_date"] = get_start_date(processed_data.index)
    results["end_date"] = get_end_date(processed_data.index)

    results["leverage"] = leverage

    if trading_costs is not None:
        results["trading_costs"] = trading_costs * 100

    results["initial_equity"] = amount
    results["exposed_capital"] = results["initial_equity"] / leverage

    if "side" in processed_data:
        results["exposure_time"] = exposure_time(processed_data["side"])

    if "accumulated_returns" in processed_data:
        results["buy_and_hold_return"] = return_buy_and_hold_pct(processed_data["accumulated_returns"]) * leverage

    if "accumulated_strategy_returns_tc" in processed_data:
        results["equity_final"] = equity_final(processed_data["accumulated_strategy_returns_tc"] * amount)
        results["equity_peak"] = equity_peak(processed_data["accumulated_strategy_returns_tc"] * amount)
        results["return_pct"] = return_pct(processed_data["accumulated_strategy_returns_tc"]) * leverage
        results["return_pct_annualized"] = return_pct_annualized(processed_data["accumulated_strategy_returns_tc"], leverage)
        results["calmar_ratio"] = calmar_ratio(processed_data["accumulated_strategy_returns_tc"])
        results["max_drawdown"] = max_drawdown_pct(processed_data["accumulated_strategy_returns_tc"])
        results["avg_drawdown"] = avg_drawdown_pct(processed_data["accumulated_strategy_returns_tc"])
        results["max_drawdown_duration"] = max_drawdown_duration(processed_data["accumulated_strategy_returns_tc"])
        results["avg_drawdown_duration"] = avg_drawdown_duration(processed_data["accumulated_strategy_returns_tc"])

    if "strategy_returns_tc" in processed_data:
        results["volatility_pct_annualized"] = volatility_pct_annualized(processed_data["strategy_returns_tc"], trading_days)
        results["sharpe_ratio"] = sharpe_ratio(processed_data["strategy_returns_tc"], trading_days=trading_days)
        results["sortino_ratio"] = sortino_ratio(processed_data["strategy_returns_tc"])

    results["nr_trades"] = int(len(trades))
    results["win_rate"] = win_rate_pct(trades)
    results["best_trade"] = best_trade_pct(trades, leverage)
    results["worst_trade"] = worst_trade_pct(trades, leverage)
    results["avg_trade"] = avg_trade_pct(trades, leverage)
    results["max_trade_duration"] = max_trade_duration(trades)
    results["avg_trade_duration"] = avg_trade_duration(trades)
    results["profit_factor"] = profit_factor(trades)
    results["expectancy"] = expectancy_pct(trades)
    results["sqn"] = system_quality_number(trades)

    return pd.Series(results)


def log_results(results, backtesting=True):

    length = 50

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

            value = results[col]

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

            logging.info(f'{printed_title:<25}{value.rjust(25)}')
        logging.info('-' * length)
        logging.info('')
    logging.info('*' * length)