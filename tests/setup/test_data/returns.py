import pandas as pd
import numpy as np
from pandas import Timestamp

cumulative = np.array([1.25, 1.26, 1.27, 1.23, 1.20, 1.22, 1.25, 1.30, 1.29, 1.26])
log_returns = np.array([0.025, 0.016, -0.10, 0.03, 0.07, 0.04, -0.02, -0.05, 0.01, 0.03])

index = pd.DatetimeIndex([
    Timestamp('2023-09-01 20:00:00+0000', tz='UTC'),
    Timestamp('2023-09-01 21:00:00+0000', tz='UTC'),
    Timestamp('2023-09-01 22:00:00+0000', tz='UTC'),
    Timestamp('2023-09-01 23:00:00+0000', tz='UTC'),
    Timestamp('2023-09-02 00:00:00+0000', tz='UTC'),
    Timestamp('2023-09-02 01:00:00+0000', tz='UTC'),
    Timestamp('2023-09-02 02:00:00+0000', tz='UTC'),
    Timestamp('2023-09-02 03:00:00+0000', tz='UTC'),
    Timestamp('2023-09-02 04:00:00+0000', tz='UTC'),
    Timestamp('2023-09-02 05:00:00+0000', tz='UTC'),
], freq='1h')

cum_returns = pd.Series(cumulative, index=index)

returns = pd.Series(log_returns, index=index)

# Daily dataset spanning ~1.5 years (550 days, straddles a New Year), used for
# the annualization-sensitive metrics (annualized return, Sortino, Calmar).
# A repeating 5-day pattern keeps the expected values hand-computable.
daily_pattern = np.array([0.01, -0.005, 0.002, -0.001, 0.004])

daily_log_returns = np.tile(daily_pattern, 110)

daily_index = pd.date_range('2022-01-01', periods=550, freq='D', tz='UTC')

daily_returns = pd.Series(daily_log_returns, index=daily_index)

daily_cum_returns = pd.Series(np.exp(np.cumsum(daily_log_returns)), index=daily_index)
