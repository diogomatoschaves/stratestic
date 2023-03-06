import os

import pytest

from stratestic.backtesting import IterativeBacktester
from stratestic.tests.setup.fixtures.external_modules import mocked_matplotlib_show
from stratestic.tests.setup.test_data.sample_data import data
from stratestic.utils.test_setup import get_fixtures

current_path = os.path.dirname(os.path.realpath(__file__))

fixtures = get_fixtures(current_path)


class TestIterativeBacktester:
    @pytest.mark.parametrize(
        "fixture",
        [
            pytest.param(fixture, id=fixture_name)
            for fixture_name, fixture in fixtures.items()
        ],
    )
    def test_run(self, fixture, mocked_matplotlib_show):

        strategy = fixture["in"]["strategy"]
        params = fixture["in"]["params"]
        trading_costs = fixture["in"]["trading_costs"]

        strategy_instance = strategy(**params, data=data)

        ite = IterativeBacktester(strategy_instance, trading_costs=trading_costs)

        ite.run()

        print(ite.results.to_dict(orient="records"))

        for i, d in enumerate(ite.results.to_dict(orient="records")):
            for key in d:
                assert d[key] == pytest.approx(fixture["out"]["expected_results"][i][key], 0.2)
