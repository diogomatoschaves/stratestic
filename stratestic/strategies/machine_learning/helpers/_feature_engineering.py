import numpy as np


def get_lag_features(df, columns=None, exclude=None, n_in=1, n_out=1, dropnan=True):
    """
    Generates lag and/or lead features from specified columns in a DataFrame for time series forecasting.

    Parameters
    ----------
    df : pd.DataFrame
        The original DataFrame from which to generate lag features.
    columns : list of str, optional
        Specific columns to generate lag features for. If None, all columns are used.
    exclude : list of str, optional
        Columns to exclude from lag feature generation.
    n_in : int, default=1
        The number of lag observations to include as input features. Generates features from t-n_in to t-1 if n_in > 0.
    n_out : int, default=1
        The number of lead observations to include as output features. Generates features from t+1 to t+n_out if n_out > 0.
    dropnan : bool, default=True
        If True, any rows with NaN values resulting from the lag feature generation will be dropped.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the generated lag and lead features. The column names for generated features are suffixed with
        '_lag[n]' or '_fwd[n]' to indicate lagged or lead features, respectively, where [n] is the number of steps lagged or lead.
    """

    if exclude is None:
        exclude = []

    lag_features = set(columns) if columns is not None else set(df.columns)
    lag_features = list(lag_features.difference(exclude))

    original_df = df.copy()

    how = {"how": 'inner' if dropnan else 'outer'}

    df = df[lag_features].copy()

    for i in range(n_in, -n_out, -1):

        if i == 0:
            continue

        df = df.join(original_df[lag_features].shift(i), rsuffix=f"_{'lag' if i > 0 else 'fwd'}{i}", **how)

    df.drop(columns=lag_features, inplace=True)

    if dropnan:
        df.dropna(axis=0, inplace=True)

    return df


def get_rolling_features(
    df,
    windows,
    columns=None,
    exclude=None,
    statistics=('mean',),
    moving_average='sma',
    dropnan=True
):
    """
    Generates rolling window features from specified columns in a DataFrame for time series analysis.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame from which to generate rolling window features.
    windows : int or list of int
        The size(s) of the rolling window(s) to calculate features over.
    columns : list of str, optional
        Specific columns to calculate rolling features for. If None, all columns are used.
    exclude : list of str, optional
        Columns to exclude from rolling feature generation.
    statistics : tuple of str, optional
        The statistical measures to calculate for each rolling window. Default is ('mean',).
    moving_average : str, default='sma'
        The type of moving average to use. Options include 'sma' for simple moving average and 'ema' for exponential moving average.
    dropnan : bool, default=True
        If True, rows with NaN values resulting from the rolling calculation will be dropped.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the generated rolling window features. Column names for generated features are suffixed with
        '_[moving_average]_[window]_[stat]', indicating the type of moving average, window size, and statistical measure, respectively.
    """

    if exclude is None:
        exclude = []

    rolling_features = set(columns) if columns is not None else set(df.columns)
    rolling_features = list(rolling_features.difference(exclude))

    if not isinstance(windows, (list, tuple, type(np.array([])))):
        windows = [windows]

    if not isinstance(statistics, (list, tuple, type(np.array([])))):
        statistics = [statistics]

    df = df[rolling_features].copy()

    for stat in statistics:
        for window in windows:

            if moving_average == 'sma':
                moving_av = df[rolling_features].rolling(window=window)
            elif moving_average == 'ema':
                moving_av = df[rolling_features].ewm(span=window)
            else:
                raise 'Method not supported'

            df = df.join(
                getattr(moving_av, stat)(),
                rsuffix=f'_{moving_average}_{window}_{stat}',
            )

    df.drop(columns=rolling_features, inplace=True)

    if dropnan:
        df.dropna(axis=0, inplace=True)

    return df


def get_labels(data, returns_col):
    """
    Extracts labels (target variable) from a DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the target variable column.
    returns_col : str
        The name of the column in `data` that contains the target variable.

    Returns
    -------
    pd.Series
        A Series containing the extracted labels, renamed to 'y'.
    """

    return data[returns_col].rename('y')


def get_x_y(X_lag, X_roll, y):
    """
    Combines lag features, rolling window features, and labels into a unified DataFrame, preparing it for machine learning models.

    Parameters
    ----------
    X_lag : pd.DataFrame
        The DataFrame containing lag features.
    X_roll : pd.DataFrame
        The DataFrame containing rolling window features.
    y : pd.Series or pd.DataFrame
        The labels or target variable(s).

    Returns
    -------
    X : pd.DataFrame
        The combined DataFrame of input features, including both lag and rolling window features, excluding any overlapping columns.
    y : pd.Series or pd.DataFrame
        The labels or target variable(s), unchanged from input.
    """

    common_cols = set(X_lag.columns).intersection(set(X_roll.columns))

    X = X_lag.join(X_roll.drop(columns=common_cols), how='inner')
    x_y = X.join(y, how='inner')

    return x_y.iloc[:, :-1].copy(), x_y.iloc[:, -1].copy()
