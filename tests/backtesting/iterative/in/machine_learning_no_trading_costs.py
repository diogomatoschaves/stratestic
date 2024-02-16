from stratestic.strategies.machine_learning import MachineLearning

strategy = MachineLearning
params = {"estimator": "Linear", "lag_features": ["returns"], "test_size": 0.4}
trading_costs = 0
optimization_params = [{"nr_lags": (2, 5)}, "System Quality Number"]
