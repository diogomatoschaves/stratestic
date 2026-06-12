from stratestic.strategies.moving_average import MovingAverage

strategy = MovingAverage
params = {"ma": 4}
trading_costs = 0
# Volatility is undefined (NaN) on this 1-hour dataset (a single daily bin
# with sample std), so a trade-based metric is used instead.
optimization_params = [{"ma": (1, 10)}, "Win Rate"]
