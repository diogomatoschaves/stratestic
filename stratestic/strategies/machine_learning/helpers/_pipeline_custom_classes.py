import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Transformer for selecting features of a specific data type or specified columns.

    Parameters
    ----------
    data_type : str
        The type of the data to select ('num' for numerical or 'cat' for categorical).
    columns : list of str, default=None
        Specific columns to be included in the output. If None, selects all columns of the specified data_type.

    Attributes
    ----------
    None

    Methods
    -------
    fit(X, y=None, **fit_params)
        Fits the transformer to X.
    transform(X, y=None, **transform_params)
        Transforms X by selecting specified features.
    """

    def __init__(self, data_type, columns=None):
        self.data_type = data_type
        self.columns = columns

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **transform_params):

        if self.columns is None:
            self.columns = X.columns

        if self.data_type == 'num':

            num_features = list(X.dtypes[(X.dtypes == 'int64') | (X.dtypes == 'float64')].index)
            columns = [col for col in X.columns if col in num_features and col in self.columns]

        elif self.data_type == 'cat':

            cat_features = list(X.dtypes[(X.dtypes != 'int64') & (X.dtypes != 'float64')].index)
            columns = [col for col in X.columns if col in cat_features and col in self.columns]

        else:
            columns = self.columns

        return X.copy()[columns]


class CustomOneHotEncoder(OneHotEncoder):
    """
    A custom OneHotEncoder that retains feature names.

    Parameters
    ----------
    drop : str, default=None
        Specifies a category to be dropped for each feature. If None, no category is dropped.

    Attributes
    ----------
    columns : list of str
        Feature names.

    Methods
    -------
    fit(X, y=None, **fit_params)
        Fits the encoder to X and stores feature names.
    transform(X, y=None, **transform_params)
        Transforms X using one-hot encoding and returns a DataFrame with feature names.
    """

    def __init__(self, drop=None):
        OneHotEncoder.__init__(self, drop=drop)

    def fit(self, X, y=None, **fit_params):
        self.columns = X.columns
        return super(CustomOneHotEncoder, self).fit(X, y, **fit_params)

    def transform(self, X, y=None, **transform_params):

        transformed_data = super(CustomOneHotEncoder, self).transform(X, **transform_params).toarray()

        return pd.DataFrame(data=transformed_data, columns=self.columns)
