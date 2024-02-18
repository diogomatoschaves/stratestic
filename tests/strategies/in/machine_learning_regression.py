from stratestic.strategies.machine_learning import MachineLearning

strategy = MachineLearning
params = {
    "estimator": "Random Forest",
    "model_type": "regression",
    "lag_features": ["returns"],
    "rolling_features": ["returns"],
    "test_size": 0.5,
    "verbose": True
}
new_parameters = {
    "estimator": "AdaBoost",
    "model_type": "regression",
    "lag_features": ["returns", "volume"],
    "rolling_features": ["returns"],
    "moving_average": "ema"
}
