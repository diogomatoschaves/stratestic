import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve


def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    metric="neg_mean_squared_error",
):
    """
    Generates plots for analyzing the learning curve of an estimator.

    Parameters
    ----------
    estimator : object
        An estimator instance implementing 'fit' and 'predict' methods.
    title : str
        The title for the chart.
    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and n_features is the number of features.
    y : array-like, shape (n_samples,) or (n_samples, n_features)
        Target relative to X for classification or regression.
    axes : array of 3 axes, optional
        Axes to use for plotting the curves. If None, new axes are created.
    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum y values plotted.
    cv : int, cross-validation generator, or an iterable, optional
        Determines the cross-validation splitting strategy.
    n_jobs : int or None, optional
        Number of jobs to run in parallel.
    train_sizes : array-like, shape (n_ticks,), dtype float or int, optional
        Relative or absolute numbers of training examples that will be used to generate the learning curve.
    metric : str, optional
        The metric to use for scoring the model performance.

    Returns
    -------
    train_sizes : array, shape (n_unique_ticks,)
        Numbers of training examples that have been used to generate the learning curve.
    train_scores : array, shape (n_ticks, n_cv_folds)
        Scores on training sets.
    test_scores : array, shape (n_ticks, n_cv_folds)
        Scores on test set.
    fit_times : array, shape (n_ticks,)
        Times spent for fitting the models on the training sets.
    """
    if axes is None:
        _, axes = plt.subplots(3, 1, figsize=(8, 18))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel(metric)

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
        scoring=metric,
    )

    train_scores = np.abs(train_scores)
    test_scores = np.abs(test_scores)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training error"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation error"
    )
    axes[0].set_ylim(bottom=0)
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, "o-")
    axes[2].fill_between(
        fit_times_mean,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Error")
    axes[2].set_title("Performance of the model")

    plt.show()

    return train_sizes, train_scores, test_scores, fit_times


def train_test_split_ts(X, y, test_size=0.2):
    """
    Splits time series data into training and test sets.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray, shape (n_samples, n_features)
        Feature dataset.
    y : pd.Series or np.ndarray, shape (n_samples,)
        Target dataset.
    test_size : float, optional
        Proportion of the dataset to include in the test split.

    Returns
    -------
    X_train : pd.DataFrame or np.ndarray
        Training set of features.
    X_test : pd.DataFrame or np.ndarray
        Test set of features.
    y_train : pd.Series or np.ndarray
        Training set of targets.
    y_test : pd.Series or np.ndarray
        Test set of targets.
    """

    train_indices = [0, int(X.shape[0] * (1 - test_size))]
    test_indices = [train_indices[1], X.shape[0]]

    X_train, y_train = X.iloc[train_indices[0]:train_indices[1]], y.iloc[train_indices[0]:train_indices[1]]
    X_test, y_test = X.iloc[test_indices[0]:test_indices[1]], y.iloc[test_indices[0]:test_indices[1]]

    return X_train, X_test, y_train, y_test
