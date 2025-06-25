import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from config.config import IP_CLASSIFICATION_CONFIG
from util.map_risk_level import map_risk_level


class FeatureNormalizer:
    def __init__(self, config_path=None):
        """Initialize FeatureNormalizer with configuration.

        Args:
            config_path (str, optional): Path to JSON config file for backward compatibility.
                                       If None, uses CONFIG from config.py
        """
        if config_path and os.path.exists(config_path):

            self.config = self.load_config(config_path)
        else:

            self.config = IP_CLASSIFICATION_CONFIG

        self.feature_ranges = self.config.get("feature_ranges", {})

    @staticmethod
    def load_config(config_path):
        """Load configuration from JSON file (backward compatibility).

        Args:
            config_path (str): Path to the configuration file.

        Returns:
            dict: Configuration dictionary.

        Raises:
            FileNotFoundError: If the configuration file is not found.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
        with open(config_path, "r") as file:
            return json.load(file)

    def normalize_with_defined_ranges(self, df):
        """Normalize DataFrame features using predefined ranges.

        Args:
            df (pd.DataFrame): DataFrame to normalize.

        Returns:
            pd.DataFrame: Normalized DataFrame.
        """
        df_normalized = pd.DataFrame()
        for feature, (min_val, max_val) in self.feature_ranges.items():
            if feature in df.columns:
                clipped_values = df[feature].clip(lower=min_val, upper=max_val)

                range_size = max_val - min_val
                if range_size > 0:
                    normalized_values = (clipped_values - min_val) / range_size
                else:
                    normalized_values = clipped_values * 0

                df_normalized[feature] = normalized_values
        return df_normalized

    def load_saved_weights(self, folder="weights", filename="final_weights.json"):
        """Load saved weights from JSON file.

        Args:
            folder (str): Folder containing the weights file.
            filename (str): Name of the weights file.

        Returns:
            pd.Series: Series containing the weights.

        Raises:
            FileNotFoundError: If the weights file is not found.
        """
        file_path = os.path.join(folder, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Weights file '{file_path}' not found.")
        with open(file_path, "r") as file:
            weights = json.load(file)
        return pd.Series(weights)

    def apply_weights_to_new_dataset(self, dataset_path=None, output_path=None):
        """Apply normalization and weights to a new dataset.

        Args:
            dataset_path (str, optional): Path to the dataset. Uses config if None.
            output_path (str, optional): Output path. Uses config if None.

        Returns:
            pd.DataFrame: DataFrame with weighted scores.
        """
        if dataset_path is None:
            dataset_path = self.config["dataset_path"]
        if output_path is None:
            output_path = self.config["output_path"]

        df = pd.read_csv(dataset_path)

        if "risk_recommended_pulsedive" in df.columns:
            df["risk_recommended_pulsedive"] = df["risk_recommended_pulsedive"].apply(
                map_risk_level
            )

        selected_features = list(self.feature_ranges.keys())
        df_selected = df[selected_features].copy()

        df_norm = self.normalize_with_defined_ranges(df_selected)

        # base_name = os.path.splitext(os.path.basename(dataset_path))[0]
        # normalized_path = f"{base_name}_normalized.csv"

        df_norm_to_save = df_norm.copy()
        if "ip_address" in df.columns:
            df_norm_to_save.insert(0, "ip_address", df["ip_address"])

        # df_norm_to_save.to_csv(normalized_path, index=False)

        inverse_features = self.config.get("inverse_features", [])
        for feature in inverse_features:
            if feature in df_norm.columns:
                df_norm[feature] = 1 - df_norm[feature]

        weights = self.load_saved_weights()
        scores = pd.DataFrame()

        for feature in df_norm.columns:
            if feature in weights:
                scores[feature] = df_norm[feature] * weights[feature]

        if "ip_address" in df.columns:
            scores.insert(0, "ip_address", df["ip_address"])

        scores.to_csv(output_path, index=False)

        return scores


if __name__ == "__main__":
    normalizer = FeatureNormalizer()
    normalizer.apply_weights_to_new_dataset()
