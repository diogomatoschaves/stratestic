from ta.trend import sma_indicator, ema_indicator


def get_moving_average(series, method, window):
    """Compute a simple ('sma') or exponential ('ema') moving average."""
    if method == 'sma':
        return sma_indicator(close=series, window=window)
    elif method == 'ema':
        return ema_indicator(close=series, window=window)

    raise ValueError(f"Method '{method}' is not supported. Choose 'sma' or 'ema'.")
