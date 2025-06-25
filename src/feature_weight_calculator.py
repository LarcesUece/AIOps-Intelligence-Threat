import json
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent))

from config.config import (
    FIGURE_SIZE,
    FONT_SIZE,
    IP_CLASSIFICATION_CONFIG,
    LABEL_SIZE,
    OUTPUT_DIR,
    TITLE_SIZE,
)
from util.map_risk_level import map_risk_level


def setup_logging(log_file=None):
    """Configure the logging system.

    Args:
        log_file (str): Path to the log file. Defaults to "outputs/weight_calculation.log".

    Returns:
        logging.Logger: Configured logger instance.
    """
    if log_file is None:
        log_file = os.path.join(OUTPUT_DIR, "weight_calculation.log")

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def calculate_final_weights(
    config_path=None, output_dir="weights", output_file="final_weights.json"
):
    """Calculate the final feature weights and save them.

    This function follows the same logic as feature_analyzer.py but focuses solely on weight calculation.
    The original calculation method is preserved (without normalizing variance), but with enhanced
    visualizations and logs for better understanding of the process.

    Args:
        config_path (str): Path to the configuration file. If None, uses config.py. Defaults to None.
        output_dir (str): Directory where the weights will be saved. Defaults to "weights".
        output_file (str): Output file name for the weights. Defaults to "final_weights.json".

    Returns:
        dict: A dictionary containing the calculated weights.

    Raises:
        FileNotFoundError: If the configuration file or dataset is not found.
    """
    logger = setup_logging()

    logger.info("=== STARTING WEIGHT CALCULATION ===")

    logger.info("1. Loading configuration...")

    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as file:
            config = json.load(file)
    else:
        config = IP_CLASSIFICATION_CONFIG

    logger.info("2. Loading dataset...")
    file_path = config["input_file"]
    if not os.path.exists(file_path):
        logger.error(f"Dataset '{file_path}' not found.")
        raise FileNotFoundError(f"Dataset '{file_path}' not found.")

    df = pd.read_csv(file_path)

    logger.info("3. Categorizing risk_recommended_pulsedive...")
    if "risk_recommended_pulsedive" in df.columns:
        df["risk_recommended_pulsedive"] = df["risk_recommended_pulsedive"].apply(
            map_risk_level
        )

    logger.info("4. Getting feature information...")
    feature_ranges = config.get("feature_ranges", {})
    selected_features = list(feature_ranges.keys())
    inverted_features = config.get("inverse_features", [])

    logger.info("5. Selecting features...")
    df_selected = df[selected_features].copy()

    logger.info("6. Normalizing features...")
    df_norm = pd.DataFrame()
    for feature, (min_val, max_val) in feature_ranges.items():
        if feature in df_selected.columns:
            clipped_values = df_selected[feature].clip(lower=min_val, upper=max_val)

            range_size = max_val - min_val
            if range_size > 0:
                normalized_values = (clipped_values - min_val) / range_size
            else:
                normalized_values = clipped_values * 0

            df_norm[feature] = normalized_values

    logger.info("7. Inverting specific features...")
    for feature in inverted_features:
        if feature in df_norm.columns:
            df_norm[feature] = 1 - df_norm[feature]

    logger.info("8. Calculating variances...")
    variances = pd.DataFrame(
        {
            "Feature": df_norm.columns,
            "Variance": np.var(df_norm, axis=0),
            "Type": [
                ("Denylist" if col not in inverted_features else "Inverted (Allowlist)")
                for col in df_norm.columns
            ],
        }
    ).sort_values("Variance", ascending=False)

    logger.info("\nVariance Analysis:\n%s", variances.to_string(index=False))

    logger.info(
        "   Variance statistics: min=%.6f, max=%.6f, mean=%.6f",
        variances["Variance"].min(),
        variances["Variance"].max(),
        variances["Variance"].mean(),
    )

    logger.info("9. Calculating correlations...")
    corr_matrix = df_norm.corr()
    mean_corr = corr_matrix.abs().mean()
    correlations = pd.DataFrame(
        {
            "Feature": df_norm.columns,
            "Mean_Correlation": mean_corr,
            "Type": [
                ("Denylist" if col not in inverted_features else "Inverted (Allowlist)")
                for col in df_norm.columns
            ],
        }
    ).sort_values("Mean_Correlation", ascending=False)

    logger.info("\nMean Correlation Analysis:\n%s", correlations.to_string(index=False))

    logger.info(
        "   Correlation statistics: min=%.6f, max=%.6f, mean=%.6f",
        correlations["Mean_Correlation"].min(),
        correlations["Mean_Correlation"].max(),
        correlations["Mean_Correlation"].mean(),
    )

    logger.info("10. Calculating composite score...")
    composite_score = pd.DataFrame(
        {
            "Feature": variances["Feature"],
            "Variance": variances["Variance"],
            "Correlation": correlations.set_index("Feature").loc[
                variances["Feature"], "Mean_Correlation"
            ],
        }
    )

    composite_score["Importance_Score"] = (
        composite_score["Variance"] + composite_score["Correlation"]
    ) / 2

    composite_score["Variance_Contribution"] = (
        composite_score["Variance"]
        / (composite_score["Variance"] + composite_score["Correlation"])
        * 100
    )

    composite_score["Correlation_Contribution"] = (
        composite_score["Correlation"]
        / (composite_score["Variance"] + composite_score["Correlation"])
        * 100
    )

    composite_score["Type"] = [
        "Denylist" if col not in inverted_features else "Inverted (Allowlist)"
        for col in composite_score["Feature"]
    ]

    composite_score = composite_score.sort_values("Importance_Score", ascending=False)

    logger.info(
        "\nImportance Score Calculations:\n%s",
        composite_score[
            ["Feature", "Variance", "Correlation", "Importance_Score", "Type"]
        ].to_string(index=False),
    )

    logger.info(
        "\nContribution Analysis (how much each metric contributes to the final score):\n%s",
        composite_score[
            [
                "Feature",
                "Variance_Contribution",
                "Correlation_Contribution",
                "Importance_Score",
            ]
        ].to_string(index=False),
    )

    logger.info(
        "\nScale Analysis (checking if variance and correlation are on comparable scales):"
    )
    var_range = variances["Variance"].max() - variances["Variance"].min()
    corr_range = (
        correlations["Mean_Correlation"].max() - correlations["Mean_Correlation"].min()
    )

    logger.info(
        f"   Variance range: [{variances['Variance'].min():.6f}, {variances['Variance'].max():.6f}], span: {var_range:.6f}"
    )
    logger.info(
        f"   Correlation range: [{correlations['Mean_Correlation'].min():.6f}, {correlations['Mean_Correlation'].max():.6f}], span: {corr_range:.6f}"
    )

    if variances["Variance"].max() <= 0.25:
        logger.info(
            "   NOTE: Variance values are naturally low (max ≤ 0.25) because data was normalized to [0,1]"
        )
        logger.info(
            "   This explains why no additional variance normalization is needed - scales are already comparable"
        )

    ratio = variances["Variance"].max() / correlations["Mean_Correlation"].max()
    logger.info(f"   Max variance / max correlation ratio: {ratio:.6f}")

    if ratio < 0.5:
        logger.info(
            "   Correlation dominates the importance score (variance is relatively small)"
        )
    elif ratio > 2.0:
        logger.info(
            "   Variance dominates the importance score (correlation is relatively small)"
        )
    else:
        logger.info(
            "   Variance and correlation have comparable influence on the importance score"
        )

    logger.info("11. Calculating normalized weights...")
    normalized_score = (
        composite_score["Importance_Score"] / composite_score["Importance_Score"].sum()
    )

    combined_weights = pd.DataFrame(
        {
            "Feature": composite_score["Feature"],
            "Final_Weight": normalized_score,
            "Type": composite_score["Type"],
        }
    ).sort_values("Final_Weight", ascending=False)

    logger.info(
        "\nFinal Calculated Weights:\n%s", combined_weights.to_string(index=False)
    )

    logger.info("12. Saving weights in JSON...")
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, output_file)
    weights_dict = combined_weights.set_index("Feature")["Final_Weight"].to_dict()

    with open(file_path, "w") as file:
        json.dump(weights_dict, file, indent=4)

    logger.info(f"Final weights saved at: {file_path}")

    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "axes.titlesize": TITLE_SIZE,
        }
    )

    logger.info("13. Generating correlation matrix plot...")
    plt.figure(figsize=FIGURE_SIZE)
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Feature Correlation Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "correlation_matrix.png")
    plt.savefig(plot_path)
    logger.info(f"Correlation matrix plot saved at: {plot_path}")
    plt.close()

    logger.info("14. Generating importance visualization plots...")

    plt.figure(figsize=(14, 8))
    bar_width = 0.35
    features = composite_score["Feature"]
    x = np.arange(len(features))

    plt.bar(
        x - bar_width / 2,
        composite_score["Variance"],
        bar_width,
        label="Variance",
        color="#1f77b4",
    )
    plt.bar(
        x + bar_width / 2,
        composite_score["Correlation"],
        bar_width,
        label="Mean Correlation",
        color="#ff7f0e",
    )

    plt.xlabel("Features")
    plt.ylabel("Value")
    plt.title("Comparison between Variance and Mean Correlation")
    plt.xticks(x, features, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()

    comparison_plot_path = os.path.join(
        output_dir, "variance_correlation_comparison.png"
    )
    plt.savefig(comparison_plot_path)
    logger.info(
        f"Variance and correlation comparison plot saved at: {comparison_plot_path}"
    )
    plt.close()

    plt.figure(figsize=(14, 8))

    var_contrib = composite_score["Variance"] / 2
    corr_contrib = composite_score["Correlation"] / 2

    plt.bar(features, var_contrib, label="Variance Contribution", color="#1f77b4")
    plt.bar(
        features,
        corr_contrib,
        bottom=var_contrib,
        label="Correlation Contribution",
        color="#ff7f0e",
    )

    plt.xlabel("Features")
    plt.ylabel("Contribution to Importance Score")
    plt.title("Importance Score Composition")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()

    contribution_plot_path = os.path.join(output_dir, "importance_composition.png")
    plt.savefig(contribution_plot_path)
    logger.info(f"Importance composition plot saved at: {contribution_plot_path}")
    plt.close()

    plt.figure(figsize=(14, 8))
    combined_weights_sorted = combined_weights.sort_values(
        "Final_Weight", ascending=True
    )

    plt.barh(
        combined_weights_sorted["Feature"],
        combined_weights_sorted["Final_Weight"],
        color="#2ca02c",
    )
    plt.xlabel("Normalized Final Weight")
    plt.ylabel("Features")
    plt.title("Final Feature Weights")
    plt.tight_layout()

    weights_plot_path = os.path.join(output_dir, "final_weights.png")
    plt.savefig(weights_plot_path)
    logger.info(f"Final weights plot saved at: {weights_plot_path}")
    plt.close()

    plt.figure(figsize=(12, 10))
    plt.pie(
        combined_weights["Final_Weight"],
        labels=combined_weights["Feature"],
        autopct="%1.1f%%",
        startangle=90,
        shadow=True,
    )
    plt.axis("equal")
    plt.title("Percentage Distribution of Final Weights")

    pie_plot_path = os.path.join(output_dir, "weights_distribution_pie.png")
    plt.savefig(pie_plot_path)
    logger.info(f"Weights distribution pie chart saved at: {pie_plot_path}")
    plt.close()

    logger.info("=== WEIGHT CALCULATION COMPLETED ===")

    return weights_dict


if __name__ == "__main__":
    weights = calculate_final_weights()
    logger = logging.getLogger(__name__)
    logger.info("\nWeights calculated successfully!")
