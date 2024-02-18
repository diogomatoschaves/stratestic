from collections import OrderedDict
from typing import Literal, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from stratestic.backtesting.helpers.evaluation import SIDE
from stratestic.strategies._mixin import StrategyMixin
from stratestic.strategies.machine_learning.helpers import (
    get_rolling_features,
    get_labels,
    get_x_y,
    get_lag_features,
    train_model,
    estimator_mapping, plot_learning_curve, estimator_params,
)
from stratestic.utils.exceptions import ModelNotFitted


class MachineLearning(StrategyMixin):
    """
    Implements a machine learning-based trading strategy using a
    specified estimator and feature engineering techniques
    such as rolling and lag features.

    Parameters
    ----------
    estimator : Literal["Linear", "Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
                        "Decision Tree", "Random Forest", "Neural Net", "AdaBoost", "Naive Bayes", "QDA"]
        The machine learning algorithm to use for the trading strategy.
    nr_lags : int, optional
        The number of lag features to generate, by default 5.
    lag_features : List[str], optional
        List of column names from the data to be used for generating lag features, by default all columns.
    rolling_features : List[str], optional
        List of column names from the data to be used for generating rolling window features, by default all columns.
    excluded_features : List[str], optional
        List of column names to be excluded from feature generation and model training, by default an empty list.
    windows : Tuple[int]
        Tuple containing sizes of the rolling windows for feature generation. Must be supplied if using rolling features.
    test_size : float, optional
        Proportion of the dataset to include in the test split, by default 0.2.
    polynomial_degree : int, optional
        The degree of the polynomial features to generate, by default 1.
    verbose : bool, optional
        If True, displays detailed results; otherwise, a summary is displayed, by default False.
    data : pd.DataFrame, optional
        The dataset to be used for the strategy, by default None.

    Attributes
    ----------
    model : Pipeline
        The machine learning pipeline constructed during training.
    X_train : pd.DataFrame
        Training feature dataset.
    y_train : pd.Series
        Training target dataset.
    X_test : pd.DataFrame
        Test feature dataset.
    y_test : pd.Series
        Test target dataset.

    Methods
    -------
    update_data(data: pd.DataFrame) -> None
        Updates the strategy's dataset and trains the model with the new data.
    calculate_positions(data: pd.DataFrame) -> pd.DataFrame
        Calculates and assigns trading positions based on the strategy's model predictions.
    get_signal(row: Optional[pd.Series]=None) -> np.ndarray
        Predicts the trading signal for a single row of data or the last row of the dataset if none is provided.
    """

    def __init__(
        self,
        estimator: Literal[
            "Logistic Regression",
            "Nearest Neighbors",
            "Linear SVM",
            "RBF SVM",
            "Gaussian Process",
            "Decision Tree",
            "Random Forest",
            "Neural Net",
            "AdaBoost",
            "Naive Bayes",
            "QDA",
        ],
        model_type: Literal["classification", "regression"] = "classification",
        nr_lags: int = 5,
        lag_features: List[str] = None,
        rolling_features: List[str] = None,
        excluded_features: List[str] = None,
        windows: Union[Tuple[int], int] = 5,
        test_size: float = 0.2,
        polynomial_degree: int = 1,
        moving_average: Literal["sma", "ema"] = "sma",
        verbose=False,
        data: pd.DataFrame = None,
        **kwargs
    ):
        """
        Constructs a MachineLearning strategy object with specified
        parameters for model training and feature generation.
        """

        self._check_estimator(estimator, model_type)
        self._model_type = model_type
        self._nr_lags = nr_lags
        self._rolling_windows = windows
        self._polynomial_degree = polynomial_degree
        self._lag_features = set(lag_features) \
            if lag_features is not None else []
        self._rolling_features = set(rolling_features) \
            if rolling_features is not None else []
        self._excluded_features = set(excluded_features) \
            if excluded_features is not None else []
        self._test_size = test_size
        self._moving_average = moving_average
        self._verbose = verbose

        self.params = OrderedDict(
            estimator=lambda x: x,
            model_type=lambda x: x,
            nr_lags=lambda x: int(x),
            lag_features=lambda x: x,
            rolling_features=lambda x: x,
            excluded_features=lambda x: x,
            rolling_windows=lambda x: x,
            test_size=lambda x: float(x),
            polynomial_degree=lambda x: int(x),
            moving_average=lambda x: x
        )

        self.model = None
        self.results = None
        self.X_test = None
        self.X_train = None
        self.y_test = None
        self.y_train = None

        StrategyMixin.__init__(self, data, **kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}(symbol = {self.symbol}, estimator = {self._estimator}, nr_lags = {self._nr_lags})"

    def update_data(self, data):
        """
        Prepares the dataset by generating features and labels, and then trains the model.

        Parameters
        ----------
        data : pd.DataFrame
            The new dataset to be used for the strategy.

        Returns
        -------
        None
        """
        data = super().update_data(data)

        X_lag = get_lag_features(data, columns=self._lag_features, exclude=self._excluded_features,
                                 n_in=self._nr_lags, n_out=1)

        X_roll = get_rolling_features(data, self._rolling_windows, columns=self._rolling_features,
                                      exclude=self._excluded_features, moving_average=self._moving_average)

        y = get_labels(data, returns_col=self._returns_col)

        X, y = get_x_y(X_lag, X_roll, y)

        model, results, X_train, X_test, y_train, y_test = train_model(
            self._estimator, X, y,
            model_type=self._model_type,
            test_size=self._test_size,
            polynomial_degree=self._polynomial_degree,
            verbose=self._verbose
        )

        self.model = model
        self.results = results
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        return self._get_data()

    def calculate_positions(self, data):
        """
        Calculates and assigns trading positions based on the model's predictions.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset for which positions are to be calculated.

        Returns
        -------
        pd.DataFrame
            The input dataset with an additional column for positions based on the strategy's predictions.
        """
        data[SIDE] = np.sign(self.model.predict(data))

        return data

    def get_signal(self, row=None):
        """
        Predicts the trading signal for a given row of data or the last row of the dataset if none is provided.

        Parameters
        ----------
        row : pd.Series, optional
            A single row of data for which the trading signal is to be predicted, by default None.

        Returns
        -------
        np.ndarray
            The predicted trading signal.
        """

        if row is None:
            row = self.data.iloc[-1]

        transformed_row = pd.DataFrame(row[self.X_test.columns]).T.astype(self.X_test.dtypes)

        return np.sign(self.model.predict(transformed_row)[0])

    def _check_estimator(self, estimator, model_type):

        if model_type not in estimator_mapping:
            raise ValueError(f"model_type must be one of 'classification' or 'regression'")

        if estimator not in estimator_mapping[model_type]:
            raise ValueError(f"{estimator} is not currently supported. Choose an estimator from "
                             f"{', '.join(estimator_mapping[model_type])}")

        self._estimator = estimator

    def _get_data(self):
        if self.X_test is not None:
            return self._original_data.join(self.X_test, how='inner')
        else:
            raise ModelNotFitted

    def learning_curve(self, metric='neg_mean_squared_error', n_splits=2):

        if self.model is None:
            raise ModelNotFitted

        title = f"Learning Curves ({self._estimator}) \n {estimator_params[self._model_type][self._estimator]})"

        tscv = TimeSeriesSplit(n_splits=n_splits)

        plot_learning_curve(
            self.model, title, self.X_train, self.y_train, metric=metric, cv=tscv
        )
