from stratestic.backtesting import VectorizedBacktester
from stratestic.strategies import MovingAverageCrossover

strategy = MovingAverageCrossover(2, 6)

strategies = [strategy]

method = "Unanimous"

backtester = VectorizedBacktester
