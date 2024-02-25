import logging
import joblib

import numpy as np
from sklearn.base import is_classifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from stratestic.strategies.machine_learning.helpers._defaults import estimator_mapping, estimator_params
from stratestic.strategies.machine_learning.helpers._helpers import train_test_split_ts
from stratestic.strategies.machine_learning.helpers._evaluation import model_evaluation
from stratestic.utils.logger import configure_logger
from stratestic.strategies.machine_learning.helpers._pipeline_custom_classes import (
    FeatureSelector,
    CustomOneHotEncoder,
)

grid_search_params_defaults = {
    "reg__n_estimators": [250, 300, 350],
    "reg__min_samples_split": [2, 4, 5],
    "reg__max_features": ["sqrt", "log2", "auto"],
    "reg__max_depth": [2, 3, 5, 6],
}

configure_logger()


def build_pipeline(estimator, polynomial_degree=1, interaction_only=False):
    """
    Constructs a machine learning pipeline with optional grid search capability.

    Parameters
    ----------
    estimator : estimator object instance
        The estimator (machine learning algorithm) to use.
    polynomial_degree : int, optional
        Degree of polynomial features to generate. Default is 1.
    interaction_only : bool, default=False
        If `True`, only interaction features are produced: features that are
        products of at most `degree` *distinct* input features, i.e. terms with
        power of 2 or higher of the same input feature are excluded.

    Returns
    -------
    pipeline : Pipeline or GridSearchCV
        The constructed machine learning pipeline or GridSearchCV object if grid_search is True.
    is_clf : bool
        Indicates whether the estimator is a classifier.
    """

    is_clf = is_classifier(estimator)

    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('num_features', Pipeline([
                ('selector', FeatureSelector('num')),
                ('feature_polynomials', PolynomialFeatures(
                    degree=polynomial_degree,
                    include_bias=False,
                    interaction_only=interaction_only
                )),
                ('scaling', StandardScaler())
            ])),
            ('cat_features', Pipeline([
                ('selector', FeatureSelector('cat')),
                ('one-hot-encoder', CustomOneHotEncoder(drop='first'))
            ]))
        ])),
        ('estimator', estimator),
    ])

    return pipeline, is_clf


def train_model(
    estimator,
    X,
    y,
    model_type="classification",
    test_size=0.2,
    polynomial_degree=1,
    verbose=True
):
    """
    Trains a machine learning model with the given dataset, optionally performing grid search.

    Parameters
    ----------
    estimator : str
        Name of the estimator to use for training.
    X : pd.DataFrame or np.ndarray
        The feature set to train the model on.
    y : pd.Series or np.ndarray
        The target variable.
    model_type : str, optional
        the type of model to be trained
    evaluation_metric : str or callable, optional
        The metric to use for evaluating model performance. Default is None.
    test_size : float, optional
        The proportion of the dataset to include in the test split. Default is 0.2.
    polynomial_degree : int, optional
        The degree of polynomial features to generate. Default is 1.
    verbose : bool, optional
        Whether to print detailed results. Default is True.

    Returns
    -------
    model : Pipeline or GridSearchCV
        The trained machine learning model or GridSearchCV object if grid search was performed.
    results : dict
        The evaluation results for both training and test datasets.
    X_train, X_test : pd.DataFrame or np.ndarray
        Training and testing feature sets.
    y_train, y_test : pd.Series or np.ndarray
        Training and testing target variables.
    """
    
    if verbose:
        logging.info("\tbuilding model...")

    estimator_ = estimator_mapping[model_type][estimator](**estimator_params[model_type][estimator])

    model, is_clf = build_pipeline(estimator_, polynomial_degree)

    if is_clf:
        y = np.sign(y)

    X_train, X_test, y_train, y_test = train_test_split_ts(X, y, test_size=test_size)
    
    if verbose:
        logging.info("\tfitting data...")

    model.fit(X_train, y_train)
    
    if verbose:
        logging.info("\tgetting model results...")

    results = model_evaluation(
        model,
        X_test,
        y_test,
        X_train,
        y_train,
        is_clf=is_clf,
        verbose=verbose
    )

    return model, results, X_train, X_test, y_train, y_test
