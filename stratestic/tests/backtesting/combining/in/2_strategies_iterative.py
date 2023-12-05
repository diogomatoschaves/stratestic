from stratestic.backtesting import VectorizedBacktester
from stratestic.strategies import MovingAverageCrossover, Momentum

strategy_1 = MovingAverageCrossover(2, 6)
strategy_2 = Momentum(5)

strategies = [strategy_1, strategy_2]

method = "Unanimous"

backtester = VectorizedBacktester