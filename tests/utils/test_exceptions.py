import pytest

from stratestic.utils.exceptions import (
    LeverageInvalid,
    NoConfigFile,
    NoSuchPipeline,
    OptimizationParametersInvalid,
    StrategyInvalid,
    StrategyRequired,
    SymbolInvalid, ModelNotFitted
)

API_PREFIX = '/api'


arg = 'abc'


class TestExceptions:

    @pytest.mark.parametrize(
        "exception,message",
        [
            pytest.param(
                LeverageInvalid(),
                f"Leverage Invalid.",
                id="LeverageInvalid",
            ),
            pytest.param(
                NoConfigFile(),
                "A config file is required.",
                id="NoConfigFile",
            ),
            pytest.param(
                NoSuchPipeline(),
                "Pipeline was not found.",
                id="NoSuchPipeline",
            ),
            pytest.param(
                OptimizationParametersInvalid(),
                f"Optimization parameters are not valid.",
                id="OptimizationParametersInvalid",
            ),
            pytest.param(
                StrategyInvalid(),
                f"Strategy is not valid.",
                id="StrategyInvalid",
            ),
            pytest.param(
                StrategyRequired(),
                f"A strategy must be included in the request.",
                id="StrategyRequired",
            ),
            pytest.param(
                SymbolInvalid(),
                f"Symbol is not valid.",
                id="SymbolInvalid",
            ),
            pytest.param(
                ModelNotFitted(),
                "The model has not been fitted yet.",
                id="ModelNotFitted",
            ),
        ],
    )
    def test_exceptions_without_args(self, exception, message):

        assert exception.message == message
        assert exception.__str__() == message
        assert exception.__repr__() == exception.__class__.__name__

    @pytest.mark.parametrize(
        "exception,message",
        [
            pytest.param(
                LeverageInvalid(arg),
                f"{arg} is not a valid leverage.",
                id="LeverageInvalid",
            ),
            pytest.param(
                NoConfigFile(arg),
                "A config file is required.",
                id="NoConfigFile",
            ),
            pytest.param(
                NoSuchPipeline(arg),
                f"Pipeline {arg} was not found.",
                id="NoSuchPipeline",
            ),
            pytest.param(
                OptimizationParametersInvalid(arg),
                f"{arg} optimization parameters are not valid.",
                id="OptimizationParametersInvalid",
            ),
            pytest.param(
                StrategyInvalid(arg),
                f"{arg} is not a valid strategy.",
                id="StrategyInvalid",
            ),
            pytest.param(
                StrategyRequired(arg),
                f"A strategy must be included in the request.",
                id="StrategyRequired",
            ),
            pytest.param(
                SymbolInvalid(arg),
                f"{arg} is not a valid symbol.",
                id="SymbolInvalid",
            ),
        ],
    )
    def test_exceptions_with_args(self, exception, message):
        assert exception.message == message
        assert exception.__str__() == message
        assert exception.__repr__() == exception.__class__.__name__
