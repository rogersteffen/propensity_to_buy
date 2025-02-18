import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize


class FeatureReport:
    def __init__(self, use_winsorize=True, use_log1p_signed=True, use_standardize=True):
        self.use_winsorize = use_winsorize
        self.use_log1p_signed = use_log1p_signed
        self.use_standardize = use_standardize
        self.winsorizer_params = {}

    @staticmethod
    def log1p_signed(x):
        """Custom transformation: sign(x) * ln(1 + |x|)."""
        return np.sign(x) * np.log1p(np.abs(x))

    def generate_report(self, df: pl.DataFrame, num_cols):
        """Generates a report showing the effect of transformations on numerical features."""
        report = []
        transformed_data = {}

        for col in num_cols:
            original_data = np.array(df[col].to_numpy(), dtype=float)

            # Step 1: Compute initial statistics
            stats_before = self.compute_statistics(original_data)

            # Step 2: Apply Winsorization
            if self.use_winsorize:
                lower_bound = np.percentile(original_data, 0.001)
                upper_bound = np.percentile(original_data, 99.999)
                self.winsorizer_params[col] = (lower_bound, upper_bound)
                winsorized_data = winsorize(original_data, limits=(0.001, 0.001))
            else:
                winsorized_data = original_data

            stats_after_winsorization = self.compute_statistics(winsorized_data)

            # Step 3: Apply log1p_signed Transformation
            if self.use_log1p_signed:
                log_transformed_data = self.log1p_signed(winsorized_data)
            else:
                log_transformed_data = winsorized_data

            stats_after_log_transform = self.compute_statistics(log_transformed_data)

            # Step 4: Standardize (if enabled)
            if self.use_standardize:
                standardized_data = (log_transformed_data - np.mean(log_transformed_data)) / np.std(log_transformed_data)
            else:
                standardized_data = log_transformed_data

            stats_after_standardization = self.compute_statistics(standardized_data)

            # Store transformed data for visualization
            transformed_data[col] = standardized_data

            # Add statistics to the report
            report.append({
                "Feature": col,
                "Mean (Before)": stats_before["mean"],
                "Std (Before)": stats_before["std"],
                "Min (Before)": stats_before["min"],
                "Max (Before)": stats_before["max"],
                "Mean (Winsorized)": stats_after_winsorization["mean"],
                "Std (Winsorized)": stats_after_winsorization["std"],
                "Min (Winsorized)": stats_after_winsorization["min"],
                "Max (Winsorized)": stats_after_winsorization["max"],
                "Mean (Log1pSigned)": stats_after_log_transform["mean"],
                "Std (Log1pSigned)": stats_after_log_transform["std"],
                "Min (Log1pSigned)": stats_after_log_transform["min"],
                "Max (Log1pSigned)": stats_after_log_transform["max"],
                "Mean (Standardized)": stats_after_standardization["mean"],
                "Std (Standardized)": stats_after_standardization["std"],
                "Min (Standardized)": stats_after_standardization["min"],
                "Max (Standardized)": stats_after_standardization["max"],
            })

        # Visualize the transformations
        self.visualize_distributions(df, num_cols, transformed_data)
        return pl.DataFrame(report)

    def compute_statistics(self, data):
        """Computes basic statistics for a given dataset."""
        return {
            "mean": np.mean(data),
            "std": np.std(data),
            "min": np.min(data),
            "max": np.max(data),
        }

    def visualize_distributions(self, df, num_cols, transformed_data):
        """Plots histograms for before and after transformations."""
        num_features = len(num_cols)
        fig, axes = plt.subplots(num_features, 3, figsize=(12, num_features * 3))

        for i, col in enumerate(num_cols):
            original_data = np.array(df[col].to_numpy(), dtype=float)
            winsorized_data = winsorize(original_data, limits=(0.01, 0.01)) if self.use_winsorize else original_data
            log_transformed_data = self.log1p_signed(winsorized_data)
            standardized_data = (log_transformed_data - np.mean(log_transformed_data)) / np.std(log_transformed_data)

            # Plot original data
            axes[i, 0].hist(original_data, bins=30, alpha=0.7, color="blue")
            axes[i, 0].set_title(f"{col} - Before")

            # Plot Winsorized data
            axes[i, 1].hist(winsorized_data, bins=30, alpha=0.7, color="green")
            axes[i, 1].set_title(f"{col} - Winsorized")

            # Plot Standardized data (after Log1pSigned)
            axes[i, 2].hist(standardized_data, bins=30, alpha=0.7, color="red")
            axes[i, 2].set_title(f"{col} - Log1pSigned + Standardized")

        plt.tight_layout()
        plt.show()



