import numpy as np
import polars as pl
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_percentile=0.01, upper_percentile=99.99):
        """
        Initialize the Winsorizer with percentiles for lower and upper bounds.

        Args:
            lower_percentile (float): The lower percentile for winsorization (e.g., 5 for 5th percentile).
            upper_percentile (float): The upper percentile for winsorization (e.g., 95 for 95th percentile).
        """
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.bounds_ = None  # To store computed bounds during fitting


    def get_feature_names_out(self, input_features=None):
        """Return feature names unchanged."""
        return input_features  # ✅ Fix: Make sure feature names pass through

    def fit(self, X, y=None):
        """
        Compute the lower and upper bounds for each feature in the data.

        Args:
            X (array-like): The input data (2D array).
            y (ignored): Not used, for compatibility with Scikit-learn API.
        """
        X = np.asarray(X)
        # Add debugging to check the shape of the input
        if X.ndim != 2:
            raise ValueError(f"Expected 2D input, got {X.ndim}D input with shape {X.shape}")

        # Compute bounds
        self.bounds_ = {}
        for col in range(X.shape[1]):
            self.bounds_[col] = {
                "lower": np.percentile(X[:, col], self.lower_percentile),
                "upper": np.percentile(X[:, col], self.upper_percentile),
            }
        return self

    def transform(self, X):
        """
        Apply winsorization to the data using the computed bounds.

        Args:
            X (array-like): The input data (2D array).

        Returns:
            np.ndarray: The winsorized data.
        """
        if self.bounds_ is None:
            raise ValueError("Winsorizer has not been fitted yet.")

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D input, got {X.ndim}D input with shape {X.shape}")

        # Apply winsorization
        X_winsorized = X.copy()
        for col, bounds in self.bounds_.items():
            X_winsorized[:, col] = np.clip(X[:, col], bounds["lower"], bounds["upper"])
        return X_winsorized


class SupervisedPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, polars_df, primary_key="customer_id", label_col="label"):
        self.primary_key = primary_key
        self.label_col = label_col
        self.feature_columns = [col for col in polars_df.columns if col not in {self.primary_key, self.label_col}]

        self.numeric_features = [col for col in polars_df.columns if polars_df[col].dtype in [pl.Int64, pl.Float64] and col not in {self.primary_key, self.label_col}]
        self.categorical_features = [col for col in polars_df.columns if polars_df[col].dtype == pl.Utf8 and col not in {self.primary_key, self.label_col}]


        # Get column indices for numeric and categorical features (since NumPy has no column names)
        self.num_indices = [self.feature_columns.index(col) for col in self.numeric_features]
        self.cat_indices = [self.feature_columns.index(col) for col in self.categorical_features]

        # Define transformers using column indices
        numeric_transformer = Pipeline(steps=[
            ("winsorizer", Winsorizer(lower_percentile=0.001, upper_percentile=99.999)),
            ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values
            ("ln_transform", FunctionTransformer(SupervisedPreprocessor.log1p_signed, validate=True)),
            ('scaler', StandardScaler())  # Scale numeric features
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing categorical values
            ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
        ])


        # Apply transformations using ColumnTransformer (column indices instead of names)
        self.preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, self.num_indices),
            ('cat', categorical_transformer, self.cat_indices)
        ])

    @classmethod
    def log1p_signed(cls, x):
        return np.sign(x) * np.log1p(np.abs(x))

    def fit(self, X: pl.DataFrame, y=None):

        if type(X) == pl.DataFrame:
            y_np = X[self.label_col].to_numpy() if y is None else y
            X_np = X.select(self.feature_columns).to_numpy()  # Keep as DataFrame
        else:
            X_np = X
            y_np = y

        return self.preprocessor.fit(X_np, y_np)

    def transform(self, X: pl.DataFrame):
        """Transforms the data using the fitted preprocessing pipeline."""
        if type(X) == pl.DataFrame:
            X_np = X.select(self.feature_columns).to_numpy()  # Keep as DataFrame
        else:
            X_np = X

        return self.preprocessor.transform(X_np)

    def fit_transform(self, X: pl.DataFrame, y=None):
        """Fits and transforms the data, handling train-test split if needed."""
        if type(X) == pl.DataFrame:
            y_np = X[self.label_col].to_numpy() if y is None else y
            X_np = X.select(self.feature_columns).to_numpy()  # Keep as DataFrame

            return self.preprocessor.fit_transform(X_np, y_np)
