import numpy as np
import pandas as pd
import pytest

from stratestic.backtesting import VectorizedBacktester
from stratestic.strategies import MachineLearning
from stratestic.strategies.machine_learning.helpers import estimator_mapping
from stratestic.strategies.machine_learning.helpers._pipeline_custom_classes import CustomOneHotEncoder
from tests.setup.fixtures.external_modules import mocked_plotly_figure_show


def make_price_data(returns):
    index = pd.date_range("2023-01-01", periods=len(returns) + 1, freq="1h", tz="UTC")
    prices = pd.Series(100 * np.exp(np.concatenate([[0], np.cumsum(returns)])), index=index)
    return pd.DataFrame(
        {"open": prices, "high": prices, "low": prices, "close": prices, "volume": 1.0}
    )


@pytest.fixture
def random_walk_data():
    # large enough for every default estimator (Nearest Neighbors needs
    # at least n_neighbors=100 training samples)
    rng = np.random.RandomState(42)
    return make_price_data(rng.normal(0, 0.01, size=300))


class TestEstimators:
    """Every advertised estimator must instantiate and train (regression
    'Linear'/'Linear SVM'/'RBF SVM' used to crash on invalid kwargs)."""

    @pytest.mark.parametrize("model_type", ["classification", "regression"])
    @pytest.mark.parametrize("estimator", sorted(estimator_mapping["classification"]))
    def test_estimator_trains(self, estimator, model_type, random_walk_data):
        ml = MachineLearning(
            estimator,
            model_type=model_type,
            nr_lags=2,
            lag_features=["returns"],
            data=random_walk_data,
        )

        assert ml.model is not None
        assert ml.results is not None


class TestCustomOneHotEncoder:

    def test_multi_category_feature(self):
        # used to fail with a shape error for any feature with more than
        # two categories, and reset the index to a RangeIndex
        index = pd.date_range("2023-01-01", periods=4, freq="1h", tz="UTC")
        X = pd.DataFrame({"cat": ["a", "b", "c", "a"]}, index=index)

        encoder = CustomOneHotEncoder(drop="first")
        encoder.fit(X)
        out = encoder.transform(X)

        assert out.shape == (4, 2)  # 3 categories, first dropped
        assert list(out.index) == list(index)


def test_label_alignment_captures_predictable_next_bar_return(mocked_plotly_figure_show):
    """The label is the next bar's return - exactly what the backtest pays.

    With perfectly alternating returns the next bar is fully predictable
    from the lags; under the old contemporaneous label the deployed signal
    was one bar stale and this strategy would have *lost* money.
    """
    returns = np.tile([0.01, -0.01], 150)
    data = make_price_data(returns)

    ml = MachineLearning(
        "Decision Tree",
        nr_lags=2,
        lag_features=["returns"],
        data=data,
    )

    assert ml.results["accuracy"] > 0.95

    vect = VectorizedBacktester(ml)
    vect.run(print_results=False, plot_results=False)

    # the out-of-sample fold gains ~1% per bar when correctly aligned
    assert vect.results["return_pct"] > 30


def test_model_filename_unique_across_configs(random_walk_data):
    ml_a = MachineLearning("Decision Tree", nr_lags=2, lag_features=["returns"], data=random_walk_data)
    ml_b = MachineLearning("Decision Tree", nr_lags=3, lag_features=["returns"], data=random_walk_data)
    ml_c = MachineLearning(
        "Decision Tree", nr_lags=2, lag_features=["returns"], data=random_walk_data[:40]
    )

    filenames = {ml.get_model_filename() for ml in (ml_a, ml_b, ml_c)}

    assert len(filenames) == 3  # params and data fingerprint both disambiguate
