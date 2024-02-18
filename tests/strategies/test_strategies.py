import os

import pytest
import pandas as pd

from stratestic.strategies import MachineLearning
from stratestic.utils.exceptions import ModelNotFitted
from tests.setup.test_data.sample_data import data
from tests.setup.test_setup import get_fixtures
from tests.setup.fixtures.external_modules import mocked_matplotlib_show, spy_matplotlib_show

current_path = os.path.dirname(os.path.realpath(__file__))

fixtures = get_fixtures(current_path)


class TestStrategy:

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "fixture",
        [
            pytest.param(fixture, id=fixture_name)
            for fixture_name, fixture in fixtures.items()
        ],
    )
    def test_strategy_data(self, fixture, spy_matplotlib_show, mocked_matplotlib_show):
        """
        GIVEN some params
        WHEN the method get_signal is called
        THEN the return value is equal to the expected response

        """

        params = fixture["in"]["params"]

        strategy = fixture["in"]["strategy"]

        instance = strategy(**params, data=data)

        instance.__repr__()

        print(instance.data.to_dict(orient='records'))

        pd.testing.assert_frame_equal(instance.data, fixture["out"]["expected_data"], check_exact=False)

    @pytest.mark.parametrize(
        "fixture",
        [
            pytest.param(fixture, id=fixture_name)
            for fixture_name, fixture in fixtures.items()
        ],
    )
    def test_strategy_set_parameters(self, fixture, mocked_matplotlib_show):
        """
        GIVEN some params
        WHEN the method get_signal is called
        THEN the return value is equal to the expected response

        """

        params = fixture["in"]["params"]

        strategy = fixture["in"]["strategy"]
        new_parameters = fixture["in"]["new_parameters"]

        instance = strategy(**params, data=data)

        instance.set_parameters(new_parameters)

        print(instance.data.to_dict(orient='records'))

        pd.testing.assert_frame_equal(
            instance.data.sort_index(axis=1),
            fixture["out"]["expected_data_set_parameters"].sort_index(axis=1),
            check_exact=False
        )

        for param, value in new_parameters.items():
            assert getattr(instance, f"_{param}") == value

    @pytest.mark.parametrize(
        "fixture",
        [
            pytest.param(fixture, id=fixture_name)
            for fixture_name, fixture in fixtures.items()
        ],
    )
    def test_strategy_get_signal(self, fixture, mocked_matplotlib_show):
        """
        GIVEN some params
        WHEN the method get_signal is called
        THEN the return value is equal to the expected response
        """

        params = fixture["in"]["params"]

        strategy = fixture["in"]["strategy"]

        instance = strategy(**params, data=data)

        assert instance.get_signal() == fixture["out"]["expected_signal"]

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "strategy,params,exception",
        [
            pytest.param(
                MachineLearning,
                {"estimator": "Invalid Estimator"},
                ValueError,
                id="MachineLearning-invalid_estimator",
            ),
            pytest.param(
                MachineLearning,
                {"estimator": "Random Forest", "model_type": "invalid"},
                ValueError,
                id="MachineLearning-invalid_model_type",
            ),
            pytest.param(
                MachineLearning,
                {"estimator": "Linear"},
                ModelNotFitted,
                id="MachineLearning-model_not_fitted",
            )
        ],
    )
    def test_strategy_exceptions(self, strategy, params, exception, mocked_matplotlib_show):
        """
        GIVEN some params
        WHEN the method get_signal is called
        THEN the return value is equal to the expected response
        """

        with pytest.raises(Exception) as excinfo:

            instance = strategy(**params)
            instance._get_data()

        assert excinfo.type == exception

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "strategy,params,method,method_params",
        [
            pytest.param(
                MachineLearning,
                {"estimator": "Random Forest", "lag_features": ["returns"]},
                "learning_curve",
                {},
                id="MachineLearning-learning_curve",
            ),
        ],
    )
    def test_strategy_specific_methods(self, strategy, params, method, method_params, mocked_matplotlib_show):
        """
        GIVEN some params
        WHEN the method get_signal is called
        THEN the return value is equal to the expected response
        """

        instance = strategy(**params, data=data, trade_on_close=False)
        getattr(instance, method)(**method_params)
