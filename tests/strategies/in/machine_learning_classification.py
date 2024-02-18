from stratestic.strategies.machine_learning import MachineLearning

strategy = MachineLearning
params = {
    "estimator": "Linear",
    "model_type": "classification",
    "lag_features": ["returns"],
    "rolling_features": ["returns"],
    "test_size": 0.5,
    "verbose": True
}
new_parameters = {
    "estimator": "Linear SVM",
    "lag_features": ["returns", "volume"],
    "rolling_features": ["returns"]
}
