import logging

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sb
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)


base_color = sb.color_palette()[0]


def plot_predictions(y_test, y_pred, is_clf):
    """
    Plots the predictions against the actual target values for both classification and regression models.

    Parameters
    ----------
    y_test : array-like
        The actual target values from the test dataset.
    y_pred : array-like
        The predicted target values by the model.
    is_clf : bool
        A flag indicating whether the model is a classifier. If True, the function will plot predictions
         for a classification model; if False, for a regression model.

    Notes
    -----
    - For classification, correct and incorrect predictions are shown in different colors.
    - For regression, actual and predicted values are plotted for comparison.

    Returns
    -------
    None
    """

    if is_clf:

        equal = y_test[y_pred == y_test]
        not_equal = y_test[y_pred != y_test]

        plt.figure(figsize=(15, 10))
        plt.bar(x=equal.index, height=equal, width=0.1, color='limegreen', label='Correct Predictions')
        plt.bar(x=not_equal.index, height=not_equal, width=0.1, color='r', label='Wrong Predictions')
        plt.yticks((-1, 1), ('Negative returns', 'Positive returns'))
        plt.title(f'{y_test.name.replace("_", " ")}: Predictions')
        plt.legend()

    else:
        plt.figure(figsize=(15, 10))
        plt.bar(x=y_test.index, height=y_test, width=0.3, color='deepskyblue', label='Real')
        plt.bar(x=y_test.index, height=y_pred, width=0.3, color='r', label='Prediction')
        plt.title(f'{y_test.name.replace("_", " ")}: Real vs Predicted')
        plt.legend()

    plt.show()


def model_evaluation(
    model,
    X_test,
    y_test,
    X_train,
    y_train,
    is_clf,
    verbose=False
):
    """
    Evaluates the performance of a fitted machine learning model using appropriate metrics
    for classification or regression and optionally plots the predictions.

    Parameters
    ----------
    model : estimator instance
        The fitted model to evaluate.
    X_test : array-like or pd.DataFrame
        The input features from the test dataset.
    y_test : array-like
        The actual target values from the test dataset.
    X_train : array-like or pd.DataFrame
        The input features from the training dataset (unused, can be omitted).
    y_train : array-like
        The actual target values from the training dataset (unused, can be omitted).
    is_clf : bool
        A flag indicating whether the model is a classifier. Determines the set of evaluation metrics to use.
    verbose : bool, default=False
        If True, logs the performance metrics and plots the predictions.

    Returns
    -------
    dict
        A dictionary containing the computed performance metrics. For classification models, it
        includes accuracy, F1 score, recall, and precision. For regression models, it includes
        R^2 score, mean absolute error (MAE), and mean squared error (MSE).

    Notes
    -----
    - The function automatically determines the average method for F1 score, recall, and precision based
      on the uniqueness of `y_test` and `y_pred` values.
    - The function supports binary and multi-class classification tasks, as well as regression tasks.

    """

    y_pred = model.predict(X_test)

    if is_clf:
        if len(np.unique(y_test)) == 2 and len(np.unique(y_pred)) == 2:
            average = 'binary'
        else:
            average = 'weighted'

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average=average)
        recall = recall_score(y_test, y_pred, average=average)
        precision = precision_score(y_test, y_pred, average=average)

        results = {
            "accuracy": accuracy,
            "f1": f1,
            "recall": recall,
            "precision": precision
        }

        def log_results():
            logging.info(f"\t\tAccuracy: {accuracy}")
            logging.info(f"\t\tF1 score: {f1}")
            logging.info(f"\t\tRecall: {recall}")
            logging.info(f"\t\tPrecision: {precision}")

    else:
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        results = {
            "r2": r2,
            "mae": mae,
            "mse": mse
        }

        def log_results():
            logging.info(f"\t\tR2 score: {r2}")
            logging.info(f"\t\tMean absolute error: {mae}")
            logging.info(f"\t\tMean squared error: {mse}")

    if verbose:
        log_results()

        plot_predictions(y_test, y_pred, is_clf)

    return results
