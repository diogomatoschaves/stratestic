from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


estimator_mapping = {
    "classification": {
        "Linear": lambda *args, **kwargs: LogisticRegression(*args, **kwargs),
        "Nearest Neighbors": lambda *args, **kwargs: KNeighborsClassifier(*args, **kwargs),
        "Linear SVM": lambda *args, **kwargs: SVC(*args, **kwargs),
        "RBF SVM": lambda *args, **kwargs: SVC(*args, **kwargs),
        "Gaussian Process": lambda *args, **kwargs:  GaussianProcessClassifier(*args, **kwargs),
        "Decision Tree": lambda *args, **kwargs: DecisionTreeClassifier(*args, **kwargs),
        "Random Forest": lambda *args, **kwargs: RandomForestClassifier(*args, **kwargs),
        "Neural Net": lambda *args, **kwargs: MLPClassifier(*args, **kwargs),
        "AdaBoost": lambda *args, **kwargs: AdaBoostClassifier(*args, **kwargs),
        "Gradient Boosting": lambda *args, **kwargs: GradientBoostingClassifier(*args, **kwargs),
    },
    "regression": {
        "Linear": lambda *args, **kwargs: Lasso(*args, **kwargs),
        "Nearest Neighbors": lambda *args, **kwargs: KNeighborsRegressor(*args, **kwargs),
        "Linear SVM": lambda *args, **kwargs: SVR(*args, **kwargs),
        "RBF SVM": lambda *args, **kwargs: SVR(*args, **kwargs),
        "Gaussian Process": lambda *args, **kwargs:  GaussianProcessRegressor(*args, **kwargs),
        "Decision Tree": lambda *args, **kwargs: DecisionTreeRegressor(*args, **kwargs),
        "Random Forest": lambda *args, **kwargs: RandomForestRegressor(*args, **kwargs),
        "Neural Net": lambda *args, **kwargs: MLPRegressor(*args, **kwargs),
        "AdaBoost": lambda *args, **kwargs: AdaBoostRegressor(*args, **kwargs),
        "Gradient Boosting": lambda *args, **kwargs: GradientBoostingRegressor(*args, **kwargs),
    }
}


estimator_params = {
    "classification": {
        "Linear": dict(C=1e6, max_iter=100000, multi_class="ovr", random_state=42),
        "Nearest Neighbors": dict(n_neighbors=100, weights='distance'),
        "Linear SVM": dict(kernel="linear", C=0.025, random_state=42),
        "RBF SVM": dict(gamma=0.01, C=0.55, degree=4, random_state=42, kernel='rbf'),
        "Gaussian Process": dict(kernel=1.0 * RBF(1.0), random_state=42),
        "Decision Tree": dict(max_depth=5, random_state=42),
        "Random Forest": dict(max_depth=5, n_estimators=10, max_features=1, random_state=42),
        "Neural Net": dict(alpha=1, max_iter=1000, random_state=42),
        "AdaBoost": dict(n_estimators=200, algorithm="SAMME", random_state=42),
        "Gradient Boosting": dict(max_depth=1, max_features='sqrt'),
    },
    "regression": {
        "Linear": dict(max_iter=100000, multi_class="ovr"),
        "Nearest Neighbors": dict(n_neighbors=100, weights='distance'),
        "Linear SVM": dict(kernel="linear", C=0.025, random_state=42),
        "RBF SVM": dict(gamma=0.01, C=0.55, degree=4, random_state=42, kernel='rbf'),
        "Gaussian Process": dict(kernel=1.0 * RBF(1.0), random_state=42),
        "Decision Tree": dict(max_depth=5, random_state=42),
        "Random Forest": dict(max_depth=5, n_estimators=200, max_features=1, random_state=42),
        "Neural Net": dict(alpha=1, max_iter=1000, random_state=42),
        "AdaBoost": dict(n_estimators=200, random_state=42),
        "Gradient Boosting": dict(max_depth=1, max_features='sqrt'),
    }
}


def get_filename(estimator, model_type, parameters):

    params_str = ','.join([f'{k}={v}'.replace(" ", '') for k, v in parameters.items()])

    return f"{estimator.replace(' ', '_').lower()}-{model_type}-{params_str}"
