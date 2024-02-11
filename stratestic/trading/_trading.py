import logging
from collections import defaultdict
from datetime import datetime


class Trader:

    def __init__(self, amount, units=0):

        self.initial_balance = amount
        self.current_balance = amount
        self.units = units
        self.leverage = 1
        self.nr_trades = 0
        self.trades = []
        self.date_col = 'date'
        self.position = defaultdict(int)
        self.positions = defaultdict(list)
        self.strategy_returns = defaultdict(list)
        self.strategy_returns_tc = defaultdict(list)

    def _set_position(self, symbol, value, **kwargs):
        """
        Sets the side for the given symbol.

        Parameters
        ----------
        symbol : str
            The trading symbol.
        value : int
            The side value.
        """
        self.position[symbol] = value
        self.positions[symbol].append(value)

    def _reset_object(self, symbol):
        """
        Resets the object attributes to their initial values.
        """
        self.positions[symbol] = []
        self._set_position(symbol, 0)
        self.strategy_returns[symbol] = []
        self.strategy_returns_tc[symbol] = []

    def _get_position(self, symbol):
        raise NotImplementedError

    def buy_instrument(
        self,
        symbol,
        date=None,
        row=None,
        units=None,
        amount=None,
        header='',
        open_trade=False,
        **kwargs
    ):
        raise NotImplementedError

    def sell_instrument(
        self,
        symbol,
        date=None,
        row=None,
        units=None,
        amount=None,
        header='',
        open_trade=False,
        **kwargs
    ):
        raise NotImplementedError

    def close_pos(self, symbol, date=None, row=None, header='', **kwargs):
        raise NotImplementedError

    def _get_balances(self, symbol):

        initial_balance = self.initial_balance[symbol] if isinstance(self.initial_balance, dict) else self.initial_balance
        current_balance = self.current_balance[symbol] if isinstance(self.current_balance, dict) else self.current_balance

        return initial_balance, current_balance

    def _get_units(self, symbol):
        return self.units[symbol] if isinstance(self.units, dict) else self.units

    def go_long(self, symbol, position, date, row, units=None, amount=None, header='', **kwargs):

        existing_units = -self._get_units(symbol)

        if position == -1:
            self.buy_instrument(symbol, date, row, units=existing_units, header=header, reducing=True, **kwargs)

        if units:
            self.buy_instrument(symbol, date, row, units=units, header=header, open_trade=True, **kwargs)
        elif amount:
            if amount == "all":
                amount = self._get_balances(symbol)[1]
            self.buy_instrument(symbol, date, row, amount=amount, header=header, open_trade=True, **kwargs)  # go long

    # helper method
    def go_short(self, symbol, position, date, row, units=None, amount=None, header='', **kwargs):

        existing_units = self._get_units(symbol)

        if position == 1:
            self.sell_instrument(symbol, date, row, units=existing_units, header=header, reducing=True, **kwargs)

        if units:
            self.sell_instrument(symbol, date, row, units=units, header=header, open_trade=True, **kwargs)
        elif amount:
            if amount == "all":
                amount = self._get_balances(symbol)[1]
            self.sell_instrument(symbol, date, row, amount=amount, header=header, open_trade=True, **kwargs)  # go short

    def trade(self, symbol, signal, date=None, row=None, amount=None, units=None, header='', **kwargs):

        position = self._get_position(symbol)
        print_results = kwargs.get('print_results')
        backtesting = kwargs.get('backtesting')

        if signal == 1:  # signal to go long
            if position in [0, -1]:
                # go long with full amount
                self.go_long(symbol, position, date, row, amount=amount, units=units, header=header, **kwargs)
        elif signal == -1:  # signal to go short
            if position in [0, 1]:
                # go short with full amount
                self.go_short(symbol, position, date, row, amount=amount, units=units, header=header, **kwargs)
        elif signal == 0:

            units = self.units[symbol] if isinstance(self.units, dict) else self.units

            if position == -1:
                self.buy_instrument(symbol, date, row, units=-units, header=header, reducing=True, **kwargs)
            elif position == 1:
                self.sell_instrument(symbol, date, row, units=units, header=header, reducing=True, **kwargs)

        if position == signal:
            if print_results and not backtesting:

                verbose_position = "LONG" if position == 1 else "SHORT" if position == -1 else "NEUTRAL"

                logging.info(header + f"Maintaining {verbose_position} position.")

                self.print_current_balance(date, header, symbol=symbol)

        self._set_position(symbol, signal, previous_position=position, **kwargs)

    def print_current_position_value(self, date, price, header='', **kwargs):
        cpv = self.get_current_position_value(price, **kwargs)
        logging.info(header + f"| {date} | Current Position Value = {round(cpv, 2)}")

    def print_current_nav(self, date, price, header='', **kwargs):
        nav = self.get_current_nav(price, **kwargs)
        logging.info(header + f"| {date} | Net Asset Value = {round(nav, 2)}")

    def print_current_balance(self, date, header='', **kwargs):
        symbol = kwargs.get("symbol", "")
        current_balance = self._get_balances(symbol)[1]

        logging.info(header + f"| {date if date else datetime.now()} | Current Balance: {round(current_balance, 2)}")

    def get_current_position_value(self, price, **kwargs):
        units = self._get_units(kwargs.get("symbol", ""))

        return units * price

    def get_current_nav(self, price, **kwargs):
        symbol = kwargs.get("symbol", "")
        current_balance = self._get_balances(symbol)[1]
        current_position_value = self.get_current_position_value(price, **kwargs)

        return current_balance + current_position_value
