import numpy as np
import polars as pl
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

# what can/should preprocessor do
# https://www.kaggle.com/code/nnjjpp/pipelines-for-preprocessing-a-tutorial



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


# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
# from sklearn.compose import ColumnTransformer
# from sklearn.model_selection import train_test_split
# import polars as pl
#
# class DataPreprocessor(BaseEstimator, TransformerMixin):
#     def __init__(self, pk_col="customer_id", label_col="label", cat_cols=None, split=False, test_size=0.1, random_state=42, stratify=True, use_ln_transform=True):
#         self.pk_col = pk_col
#         self.label_col = label_col
#         self.cat_cols = cat_cols  # List of categorical columns (if provided)
#         self.split = split
#         self.test_size = test_size
#         self.random_state = random_state
#         self.stratify = stratify
#         self.use_ln_transform = use_ln_transform
#         self.feature_columns = None
#         self.num_cols = None
#         self.pipeline = None  # Will be initialized in `set_feature_columns()`
#
#     def set_feature_columns(self, aligned_df):
#
#         """Automatically determines numerical and categorical feature columns."""
#         all_columns = aligned_df.columns
#         self.feature_columns = [col for col in all_columns if col not in [self.pk_col, self.label_col]]
#
#         # Auto-detect categorical columns if not provided
#         if self.cat_cols is None:
#             self.cat_cols = [col for col in self.feature_columns if aligned_df[col].dtype == pl.Utf8]
#             # this intermittently prints twice
#             print(f"Automatically detected categorical columns: {self.cat_cols}\n")
#
#         # Validate provided categorical columns
#         elif any(col for col in self.cat_cols if col not in self.feature_columns):
#             invalid_cats = [col for col in self.cat_cols if col not in self.feature_columns]
#             print(f"Warning: The following provided categorical columns are not valid and will be ignored: {invalid_cats}")
#             self.cat_cols = [col for col in self.cat_cols if col in self.feature_columns]
#
#         # Detect extra text columns not in provided cat_cols
#         extra_text_cols = [col for col in self.feature_columns if aligned_df[col].dtype == pl.Utf8 and col not in self.cat_cols]
#         if extra_text_cols:
#             print(f"Warning: The following text columns are not in the provided categorical column list and will be ignored: {extra_text_cols}")
#
#         # Determine numerical columns (anything not in categorical list)
#         self.num_cols = [col for col in self.feature_columns if col not in self.cat_cols]
#
#         # Define pipelines
#         num_pipeline_steps = [("winsorizer", Winsorizer(lower_percentile=0.001, upper_percentile=99.999))]
#         # num_pipeline_steps = []
#
#         if self.use_ln_transform:
#
#             def log1p_signed(x):
#                 return np.sign(x) * np.log1p(np.abs(x))
#
#             num_pipeline_steps.append(("ln_transform", FunctionTransformer(log1p_signed, validate=True)))  # NEW: PowerTransformer
#
#         num_pipeline_steps.append(("scaler", StandardScaler()))
#
#         num_pipeline = Pipeline(num_pipeline_steps) if num_pipeline_steps else "passthrough"
#
#         cat_pipeline = Pipeline([
#             ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
#         ])
#
#         # Create ColumnTransformer with correct column names
#         self.pipeline = ColumnTransformer([
#             ("num", num_pipeline, self.num_cols),  # Numerical columns
#             ("cat", cat_pipeline, self.cat_cols)  # Categorical columns
#         ])
#
#     def _split_data(self, X, y):
#         """Splits the data into training and testing sets."""
#         stratify_labels = y if self.stratify else None
#         return train_test_split(
#             X, y, test_size=self.test_size, random_state=self.random_state, stratify=stratify_labels
#         )
#
#     def fit(self, X: pl.DataFrame, y=None):
#         """Fits the preprocessing pipeline."""
#         y_np = X[self.label_col].to_numpy() if y is None else y
#         X_df = X.select(self.feature_columns)  # Keep as DataFrame
#
#         if self.split:
#             X_train, X_test, y_train, y_test = self._split_data(X_df, y_np)
#             self.pipeline.fit(X_train, y_train)
#
#         else:
#             self.pipeline.fit(X_df, y_np)
#
#         return self
#
#     def transform(self, X: pl.DataFrame):
#         """Transforms the data using the fitted preprocessing pipeline."""
#         X_df = X.select(self.feature_columns)  # Keep as DataFrame
#         return self.pipeline.transform(X_df)
#
#     def fit_transform(self, X: pl.DataFrame, y=None):
#         """Fits and transforms the data, handling train-test split if needed."""
#         y_np = X[self.label_col].to_numpy() if y is None else y
#         X_df = X.select(self.feature_columns)  # Keep as DataFrame
#
#         if self.split:
#             X_train, X_test, y_train, y_test = self._split_data(X_df, y_np)
#             X_train_transformed = self.pipeline.fit_transform(X_train, y_train)
#             X_test_transformed = self.pipeline.transform(X_test)
#             return X_train_transformed, X_test_transformed, y_train, y_test  # ✅ Returns 4 values
#         else:
#             X_transformed = self.pipeline.fit_transform(X_df, y_np)
#             return X_transformed, None, y_np, None  # ✅ Returns `None` for test data




