import pytest

from stratestic.trading import Trader
from tests.setup.fixtures.external_modules import spy_logging_info


class TestTrader:

    @pytest.mark.parametrize(
        "method,extra_input",
        [
            pytest.param(
                '_get_position',
                ['BTCUSDT'],
                id="_get_position",
            ),
            pytest.param(
                'buy_instrument',
                ['BTCUSDT'],
                id="buy_instrument",
            ),
            pytest.param(
                'sell_instrument',
                ['BTCUSDT'],
                id="sell_instrument",
            ),
            pytest.param(
                'close_pos',
                ['BTCUSDT'],
                id="close_pos",
            ),
        ],
    )
    def test_exceptions_without_args(self, method, extra_input):

        trader = Trader(1000)

        with pytest.raises(NotImplementedError):
            getattr(trader, method)(*extra_input)

    @pytest.mark.parametrize(
        "method",
        [
            pytest.param(
                'print_current_position_value',
                id="_set_position",
            ),
            pytest.param(
                'print_current_nav',
                id="_get_position",
            ),
        ],
    )
    def test_print_functions(self, method, spy_logging_info):

        trader = Trader(1000)

        getattr(trader, method)('', 10000)

        assert spy_logging_info.call_count == 1
