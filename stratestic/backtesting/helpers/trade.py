import numpy as np

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Trade:
    """ Class for storing information about a trade.

    Parameters
    ----------
    entry_date : datetime
        Entry date in datetime format.
    exit_date : datetime
        Exit date in datetime format.
    entry_price : float, None
        Price at which the trade was entered.
    exit_price : float, None
        Price at which the trade was exited.
    units : float, None
        Number of shares or contracts traded.
    side : int
        Trade side, either 1 for long or -1 for short.
    profit : float
        Trade net profit
    """

    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    units: float
    side: int
    equity: float = None
    amount: float = None
    profit: float = None
    pnl: float = None
    liquidation_price: float = None
    maintenance_rate: float = None
    maintenance_amount: float = None
    # must stay the LAST field: Trade is constructed positionally in the
    # iterative engine for all fields up to maintenance_amount
    symbol: str = None

    def calculate_profit(self, prev_equity):
        if self.amount is None:
            return

        self.profit = self.equity - prev_equity

    def calculate_pnl_pct(self, leverage, short_model="static", prev_equity=None):
        # a leveraged loss is capped at -100%: the account is wiped out
        if short_model == "static" and prev_equity:
            # real cash accounting: the pnl is the profit on the pre-trade equity
            self.pnl = max(self.profit / prev_equity, -1.0)
        else:
            self.pnl = max((np.exp(np.log(self.exit_price / self.entry_price) * self.side) - 1) * leverage, -1.0)
