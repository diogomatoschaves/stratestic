import logging

from stratestic.backtesting.helpers.evaluation.metrics import *
from stratestic.backtesting.helpers.evaluation._constants import results_sections, results_mapping, results_aesthetics

import pandas as pd


def get_results(processed_data, trades, leverage, tc, amount, trading_days):
    total_duration = get_total_duration(processed_data.index)
    start_date = get_start_date(processed_data.index)
    end_date = get_end_date(processed_data.index)

    leverage = leverage
    trading_costs = tc * 100
    initial_equity = amount
    exposed_capital = initial_equity / leverage

    exposure = exposure_time(processed_data["side"])
    final_equity = equity_final(processed_data["accumulated_strategy_returns_tc"] * amount)
    peak_equity = equity_peak(processed_data["accumulated_strategy_returns_tc"] * amount)
    buy_and_hold_return = return_buy_and_hold_pct(processed_data["accumulated_returns"]) * leverage
    pct_return = return_pct(processed_data["accumulated_strategy_returns_tc"]) * leverage
    annualized_pct_return = return_pct_annualized(processed_data["accumulated_strategy_returns_tc"], leverage)
    annualized_pct_volatility = volatility_pct_annualized(
        processed_data["strategy_returns_tc"],
        trading_days
    )

    sharpe = sharpe_ratio(processed_data["strategy_returns_tc"], trading_days=trading_days)
    sortino = sortino_ratio(processed_data["strategy_returns_tc"])
    calmar = calmar_ratio(processed_data["accumulated_strategy_returns_tc"])
    max_drawdown = max_drawdown_pct(processed_data["accumulated_strategy_returns_tc"])
    avg_drawdown = avg_drawdown_pct(processed_data["accumulated_strategy_returns_tc"])
    max_drawdown_dur = max_drawdown_duration(processed_data["accumulated_strategy_returns_tc"])
    avg_drawdown_dur = avg_drawdown_duration(processed_data["accumulated_strategy_returns_tc"])

    nr_trades = int(len(trades))
    win_rate = win_rate_pct(trades)
    best_trade = best_trade_pct(trades, leverage)
    worst_trade = worst_trade_pct(trades, leverage)
    avg_trade = avg_trade_pct(trades, leverage)
    max_trade_dur = max_trade_duration(trades)
    avg_trade_dur = avg_trade_duration(trades)
    profit_fctor = profit_factor(trades)
    expectancy = expectancy_pct(trades)
    sqn = system_quality_number(trades)

    results = pd.Series(
        dict(
            total_duration=total_duration,
            nr_trades=nr_trades,
            start_date=start_date,
            end_date=end_date,
            trading_costs=trading_costs,
            leverage=leverage,
            initial_equity=initial_equity,
            exposed_capital=exposed_capital,
            exposure_time=exposure,
            buy_and_hold_return=buy_and_hold_return,
            return_pct=pct_return,
            equity_final=final_equity,
            equity_peak=peak_equity,
            return_pct_annualized=annualized_pct_return,
            volatility_pct_annualized=annualized_pct_volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_drawdown,
            avg_drawdown=avg_drawdown,
            max_drawdown_duration=max_drawdown_dur,
            avg_drawdown_duration=avg_drawdown_dur,
            win_rate=win_rate,
            best_trade=best_trade,
            worst_trade=worst_trade,
            avg_trade=avg_trade,
            max_trade_duration=max_trade_dur,
            avg_trade_duration=avg_trade_dur,
            profit_factor=profit_fctor,
            expectancy=expectancy,
            sqn=sqn
        )
    )

    return results


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