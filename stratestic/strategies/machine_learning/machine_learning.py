import logging
import os
from collections import OrderedDict
from typing import Literal, List

import dill
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
    estimator_mapping, plot_learning_curve, estimator_params, get_filename,
)
from stratestic.utils.exceptions import ModelNotFitted


class MachineLearning(StrategyMixin):
    """
    A class to implement machine learning-based trading strategies using specified estimators and feature engineering techniques. It supports both classification and regression models and includes functionalities for training the model, making predictions, and evaluating performance.

    Parameters
    ----------
    estimator : Literal
        The machine learning algorithm to use for the trading strategy. Supported algorithms
        include "Linear", "Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
        "Decision Tree", "Random Forest", "Neural Net", "AdaBoost", "Naive Bayes", and "QDA".
    model_type : Literal["classification", "regression"], optional
        The type of machine learning model to be used, either "classification" for
        classification tasks or "regression" for regression tasks. Default is "classification".
    nr_lags : int, optional
        The number of lag observations to generate as features. Default is 5.
    lag_features : List[str], optional
        The specific columns from the data to be used for generating lag features.
        If None, all columns are used. Default is None.
    rolling_features : List[str], optional
        The specific columns from the data to be used for generating rolling window features.
        If None, all columns are used. Default is None.
    excluded_features : List[str], optional
        The columns to be excluded from feature generation and model training. Default is an empty list.
    window : int
        The size of the rolling windows for feature generation. Must be specified if using rolling features.
    test_size : float, optional
        The proportion of the dataset to include in the test split. Default is 0.2.
    polynomial_degree : int, optional
        The degree of the polynomial features to generate. Default is 1.
    moving_average : Literal["sma", "ema"], optional
        The type of moving average to use for rolling features. Options are "sma" for
        simple moving average and "ema" for exponential moving average. Default is "sma".
    verbose : bool, optional
        If True, detailed results will be displayed. Otherwise, a summary is displayed. Default is False.
    save_model : bool, optional
        If True, the trained model is saved to a file after each (re)training,
        including during parameter optimization. Default is False.
    load_model : str, optional
        The filename of a previously saved model to load. If specified, the model will
        be loaded instead of trained. Default is None. Note that model files are
        deserialized with dill and must come from a trusted source.
    models_dir : str, optional
        The directory where models are saved or loaded from. Default is 'models'.
    data : pd.DataFrame, optional
        The dataset to be used for the strategy. Default is None.

    Attributes
    ----------
    model : Pipeline
        The machine learning model pipeline constructed during training.
    X_train, X_test : pd.DataFrame
        Training and test feature datasets, respectively.
    y_train, y_test : pd.Series
        Training and test target datasets, respectively.
    results : dict
        The performance metrics of the model.

    Methods
    -------
    update_data(data: pd.DataFrame) -> None
        Prepares the dataset by generating features and labels, then trains the model with the new data.
    calculate_positions(data: pd.DataFrame) -> pd.DataFrame
        Calculates and assigns trading positions based on the model's predictions.
    get_signal(row: Optional[pd.Series]=None) -> np.ndarray
        Predicts the trading signal for a given row of data or the last row of the dataset if none is provided.
    learning_curve(metric='neg_mean_squared_error', n_splits=2) -> None
        Plots the learning curve of the model using the specified metric and number of splits for cross-validation.

    Raises
    ------
    ValueError
        If an unsupported estimator is specified or if the estimator is not suitable for the specified model type.
    ModelNotFitted
        If attempting to predict or plot learning curves without a trained model.
    """

    def __init__(
        self,
        estimator: Literal[
            "Linear",
            "Nearest Neighbors",
            "Linear SVM",
            "RBF SVM",
            "Gaussian Process",
            "Decision Tree",
            "Random Forest",
            "Neural Net",
            "AdaBoost",
        ] = None,
        model_type: Literal["classification", "regression"] = "classification",
        nr_lags: int = 5,
        lag_features: List[str] = None,
        rolling_features: List[str] = None,
        excluded_features: List[str] = None,
        window: int = 5,
        test_size: float = 0.2,
        polynomial_degree: int = 1,
        moving_average: Literal["sma", "ema"] = "sma",
        verbose=False,
        load_model=None,
        save_model=False,
        models_dir='stratestic/strategies/machine_learning/models',
        data: pd.DataFrame = None,
        **kwargs
    ):
        """
        Constructs a MachineLearning strategy object with specified
        parameters for model training and feature generation.
        """
        StrategyMixin.__init__(self, **kwargs)

        self._check_estimator(estimator, model_type, load_model)
        self._model_type = model_type
        self._nr_lags = nr_lags
        self._window = window
        self._polynomial_degree = polynomial_degree
        self._lag_features = set(lag_features) \
            if lag_features is not None else [self._returns_col]
        self._rolling_features = set(rolling_features) \
            if rolling_features is not None else []
        self._excluded_features = set(excluded_features) \
            if excluded_features is not None else []
        self._test_size = test_size
        self._moving_average = moving_average
        self._verbose = verbose
        self._load_model = load_model
        self._save_model = save_model
        self._models_dir = models_dir

        self.params = OrderedDict(
            estimator=lambda x: x,
            model_type=lambda x: x,
            nr_lags=lambda x: int(x),
            lag_features=lambda x: x,
            rolling_features=lambda x: x,
            excluded_features=lambda x: x,
            window=lambda x: int(x),
            test_size=lambda x: float(x),
            polynomial_degree=lambda x: int(x),
            moving_average=lambda x: x
        )

        if self._load_model:
            self.load_model()
            if data is not None:
                self.data = self.update_data(data.copy())
        else:
            self.model = None
            self.results = None
            self.X_test = None
            self.X_train = None
            self.y_test = None
            self.y_train = None

            if data is not None:
                self.data = self.update_data(data.copy())

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

        X_roll = get_rolling_features(data, self._window, columns=self._rolling_features,
                                      exclude=self._excluded_features, moving_average=self._moving_average)

        y = get_labels(data, returns_col=self._returns_col)

        X, y = get_x_y(X_lag, X_roll, y)

        if self._load_model:
            if self.X_train is not None and len(X.index.intersection(self.X_train.index)) > 0:
                logging.warning(
                    "The provided data overlaps the loaded model's training period; "
                    "backtest results on this data will be partially in-sample."
                )

            self.X_test = X
            return self._get_data()

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

        if self._save_model:
            self.save_model()

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
            transformed_row = self.X_test.iloc[-1:]
        else:
            transformed_row = pd.DataFrame(row[self.X_test.columns]).T.astype(self.X_test.dtypes)

        return np.sign(self.model.predict(transformed_row)[0])

    def _check_estimator(self, estimator, model_type, load_model):

        if load_model:
            return

        if model_type not in estimator_mapping:
            raise ValueError("model_type must be one of 'classification' or 'regression'")

        if estimator not in estimator_mapping[model_type]:
            raise ValueError(f"{estimator} is not currently supported. Choose an estimator from "
                             f"{', '.join(estimator_mapping[model_type])}")

        self._estimator = estimator

    def _get_data(self):
        if self.X_test is not None:
            return self._original_data.join(self.X_test, how='inner')
        else:
            raise ModelNotFitted

    def load_model(self):
        """
        Loads a previously saved strategy (model, configuration, and
        training/test datasets) from `self._models_dir / self._load_model`.

        Deserialization uses `dill`, which can execute arbitrary code while
        unpickling - only load model files from a trusted source.

        Returns
        -------
        pd.DataFrame
            The original dataset associated with the loaded model.

        Raises
        ------
        FileNotFoundError
            If the specified model file does not exist in `self._models_dir`.
        """

        original_file_name = self._load_model

        file_path = os.path.abspath(
            os.path.join(
                self._models_dir,
                self._load_model or ''
            )
        )

        ml = dill.load(open(file_path, 'rb'))

        self.__dict__.update(ml.__dict__)
        self._load_model = original_file_name

        return self._get_data()

    def get_model_filename(self):
        """
        Builds the filename (without extension) used to persist this strategy.

        Includes the estimator parameters, the strategy's own feature/training
        parameters, and a fingerprint of the training data, so that two runs
        with different configurations or datasets don't overwrite each other.
        """
        parameters = {
            **estimator_params[self._model_type][self._estimator],
            "nr_lags": self._nr_lags,
            "window": self._window,
            "test_size": self._test_size,
            "polynomial_degree": self._polynomial_degree,
            "moving_average": self._moving_average,
        }

        filename = get_filename(self._estimator, self._model_type, parameters)

        if self.X_train is not None and len(self.X_train) > 0:
            start = pd.Timestamp(self.X_train.index[0]).strftime('%Y%m%d%H%M')
            end = pd.Timestamp(self.X_train.index[-1]).strftime('%Y%m%d%H%M')
            filename = f"{filename}-{start}-{end}-{len(self.X_train)}"

        return filename

    def save_model(self):
        """
        Serializes the strategy (model, configuration, and datasets) with
        `dill` into `self._models_dir`, under the name returned by
        `get_model_filename()`. Overwrites an existing file of the same name.
        """

        if self._verbose:
            logging.info("\tsaving model...")

        file_path = os.path.abspath(
            os.path.join(
                self._models_dir,
                self.get_model_filename() + '.pkl'
            )
        )

        dill.dump(self, open(file_path, 'wb'))

    def learning_curve(self, metric='neg_mean_squared_error', n_splits=2):
        """
        Plots the learning curve of the machine learning model to evaluate its performance over
        varying amounts of training data. This method helps in diagnosing whether the model suffers
        from high variance or high bias, indicating potential overfitting or underfitting respectively.

        Parameters
        ----------
        metric : str, optional
            The performance metric to use for evaluating the model. Default is 'neg_mean_squared_error',
             which is suitable for regression tasks. For classification, consider metrics like 'accuracy',
              'roc_auc', etc.
        n_splits : int, optional
            The number of splits to use for generating the learning curve, determining the size of
            each training set increment. Default is 2, which means the training data is split into
            two portions to calculate the model's performance at two points of training set size.

        Raises
        ------
        ModelNotFitted
            If the method is called before the model has been fitted. This ensures that there is
            a trained model available to evaluate.

        Notes
        -----
        - The learning curve is plotted using cross-validation with a `TimeSeriesSplit` strategy
        to maintain the order of observations, which is important for time series data.
        - The method automatically adjusts to the specified `metric` for plotting, allowing for
        flexible evaluation based on the task at hand (regression or classification).
        - This method is particularly useful for assessing how much benefit additional training
        data might bring to the model's predictive performance.

        Examples
        --------
        >>> strategy_instance.learning_curve(metric='accuracy', n_splits=5)
        This will plot the learning curve of the strategy's model using accuracy as the evaluation metric and 5 splits for the training data size.

        """

        if self.model is None:
            raise ModelNotFitted

        title = f"Learning Curves ({self._estimator}) \n {estimator_params[self._model_type][self._estimator]})"

        tscv = TimeSeriesSplit(n_splits=n_splits)

        plot_learning_curve(
            self.model, title, self.X_train, self.y_train, metric=metric, cv=tscv
        )
