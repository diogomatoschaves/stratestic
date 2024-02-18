import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from stratestic.backtesting.helpers.evaluation import CUM_SUM_STRATEGY, CUM_SUM_STRATEGY_TC, BUY_AND_HOLD, MARGIN_RATIO
from stratestic.backtesting.helpers.evaluation.metrics import get_drawdowns, get_dd_durations_limits

pio.renderers.default = "browser"


def plot_backtest_results(
    data,
    trades,
    margin_threshold,
    index_frequency,
    offset=0,
    plot_margin_ratio=False,
    show_plot_no_tc=False,
    title=''
):
    """
    Plots backtesting results for a trading strategy.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the following columns:
        - 'accumulated_strategy_returns_tc': accumulated returns including trading costs
        - 'accumulated_strategy_returns': accumulated returns without trading costs
        - 'accumulated_returns': accumulated returns without trading costs
    trades : pandas.DataFrame
        DataFrame containing trade information, including the following columns:
        - 'entry_date': entry date of the trade
        - 'side': side of the trade (-1 for short, 1 for long)
        - 'profit': profit of the trade
        - 'units': size of the side
    margin_threshold : float
        threshold for maximum allowed margin ratio
    index_frequency : string
        frequency of the index of the data
    offset : int
        Offset for vertical margin of the plot.
    plot_margin_ratio: bool, optional
        Whether to plot the margin ratio curve
    show_plot_no_tc : bool, optional
        Whether to plot equity without trading costs (default is False)
    title : str, optional
        Title to show on the backtesting results

    Returns
    -------
        None (displays plot using Plotly)

    """

    number_rows = 3 if plot_margin_ratio else 2

    fig = make_subplots(rows=number_rows, cols=1, shared_xaxes=True)

    plot_equity_curves(fig, data, show_plot_no_tc, index_frequency)

    plot_trades(fig, trades)

    height = 1000

    if plot_margin_ratio:
        plot_margin_ratios(fig, data, margin_threshold)

        height = height + 350

    variable_offset = 25 * offset

    fig.update_layout(title=title, height=height + variable_offset, showlegend=True, margin=dict(t=105 + variable_offset))

    fig.update_layout(
        title={
            "yref": "container",
            "y": 0.97,
            "yanchor": "top"
        }
    )

    fig.update_yaxes(title_text='Value (USD)', row=1, col=1)
    fig.update_yaxes(title_text='Trade PnL (%)', row=2, col=1)
    fig.update_yaxes(title_text='Margin Ratio (%)', row=3, col=1)

    fig.update_xaxes(row=1, col=1, title_text='Date', showticklabels=True, overwrite=True)
    fig.update_xaxes(row=2, col=1, title_text='Date', showticklabels=True, overwrite=True)
    fig.update_xaxes(row=3, col=1, title_text='Date', showticklabels=True, overwrite=True)

    fig.show()


def plot_margin_ratios(fig, data, margin_threshold):

    fig.add_trace(go.Scatter(
        x=data[MARGIN_RATIO].index,
        y=data[MARGIN_RATIO],
        name='Margin Ratio',
        line=dict(
            width=1.5,
            color='darkcyan'
        )
    ), row=3, col=1)

    threshold = margin_threshold * 100
    start = data[MARGIN_RATIO].index[0]
    end = data[MARGIN_RATIO].index[-1]

    fig.add_trace(go.Scatter(
        x=[start, end],
        y=[threshold, threshold],
        name=f'Margin Ratio Threshold',
        mode='lines',
        line=dict(
            color='red',
            width=1,
            dash='dot'
        )
    ), row=3, col=1)


def plot_equity_curves(fig, data, show_plot_no_tc, index_frequency):

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data[CUM_SUM_STRATEGY_TC],
        name='Equity',
        line=dict(
            width=1.5,
            color='steelblue'
        )
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data[BUY_AND_HOLD],
        name='Buy & Hold',
        line=dict(
            color='Silver',
            width=1.5
        )
    ), row=1, col=1)

    if show_plot_no_tc:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[CUM_SUM_STRATEGY],
            name='Equity (no trading costs)',
            line=dict(
                width=0.8,
                color='Silver'
            )
        ), row=1, col=1)

    # plot drawdowns
    close_date = data.index.shift(1, freq=index_frequency)

    durations, limits = get_dd_durations_limits(data[CUM_SUM_STRATEGY_TC], close_date)

    x = []
    y = []
    for limit in limits:
        x.extend(limit)
        x.append(None)

        value = data[CUM_SUM_STRATEGY_TC][limit[0]]

        y.extend([value, value])
        y.append(None)

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        name='Drawdown',
        mode='lines',
        line=dict(
            color='Gold',
            width=1
        )
    ), row=1, col=1)

    # plot max drawdown duration
    if len(durations) > 0:
        max_duration_index = np.argmax(durations)

        start, end = limits[max_duration_index]
        value = data[CUM_SUM_STRATEGY_TC][start]

        fig.add_trace(go.Scatter(
            x=[start, end],
            y=[value, value],
            name=f'Max Drawdown Duration',
            mode='lines',
            line=dict(
                color='Red',
                width=1
            )
        ), row=1, col=1)

    # plot peak equity point
    peak_index = data[CUM_SUM_STRATEGY_TC].argmax()
    peak_time = data.index[peak_index]
    peak_value = data[CUM_SUM_STRATEGY_TC][peak_index]

    fig.add_trace(go.Scatter(
        x=[peak_time],
        y=[peak_value],
        name='Peak',
        mode='markers',
        marker=dict(
            color='MediumBlue',
            size=8
        )
    ), row=1, col=1)

    # Plot lowest equity point
    low_index = data[CUM_SUM_STRATEGY_TC].argmin()
    low_time = data.index[low_index]
    low_value = data[CUM_SUM_STRATEGY_TC][low_index]

    fig.add_trace(go.Scatter(
        x=[low_time],
        y=[low_value],
        name='Lowest',
        mode='markers',
        marker=dict(
            color='Maroon',
            size=8
        )
    ), row=1, col=1)

    # Plot max drawdown
    drawdowns = get_drawdowns(data[CUM_SUM_STRATEGY_TC])

    max_drawdown_index = drawdowns.argmin()
    max_drawdown_time = drawdowns.index[max_drawdown_index]
    max_drawdown_equity = data[CUM_SUM_STRATEGY_TC][max_drawdown_index]
    max_drawdown_value = drawdowns[max_drawdown_index]

    fig.add_trace(go.Scatter(
        x=[max_drawdown_time],
        y=[max_drawdown_equity],
        name=f'Max Drawdown ({round(max_drawdown_value * 100, 1)} %)',
        mode='markers',
        marker=dict(
            color='Crimson',
            size=7
        )
    ), row=1, col=1)


def size_trade_markers(notional_value, min_marker_size=10, max_marker_size=35):

    min_value = notional_value.min()
    max_value = notional_value.max()

    normalized = (notional_value - min_value) / (max_value - min_value)

    normalized = pd.Series(np.where(np.isnan(normalized), 0.5, normalized))

    return min_marker_size + normalized * (max_marker_size - min_marker_size)


def plot_trades(fig, trades):

    if len(trades) > 0:

        trades["pnl_pct"] = np.round(trades["pnl"] * 100, 2)

        # define a boolean column indicating if each trade is long or short
        trades['is_long'] = trades['side'].apply(lambda x: x > 0)

        # define marker size accoridng to the trade size
        marker_size = size_trade_markers(trades['units'] * trades["entry_price"])

        # create separate traces for long and short trades
        fig.add_trace(go.Scatter(
            x=trades[trades['is_long']]["entry_date"], y=trades.loc[trades['is_long'], 'pnl_pct'],
            name='Long', mode='markers', marker=dict(
                symbol='triangle-up',
                color='limegreen',
                size=marker_size[trades['is_long']],
                line=dict(
                    color='Black',
                    width=1
                )
            )
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=trades[~trades['is_long']]["entry_date"], y=trades.loc[~trades['is_long'], 'pnl_pct'],
            name='Short', mode='markers', marker=dict(
                symbol='triangle-down',
                color='red',
                size=marker_size[~trades['is_long']],
                line=dict(
                    color='White',
                    width=1
                )
            )
        ), row=2, col=1)
