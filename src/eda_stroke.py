"""Exploratory Data Analysis for the healthcare stroke dataset."""

from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, f1_score, recall_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "healthcare-dataset-stroke-data.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda"


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid", palette="Set2")

    df = pd.read_csv(DATA_FILE, na_values=["N/A"])

    # Remove the identifier from analysis; it has no meaningful predictive value.
    analysis_df = df.drop(columns=["id"]).copy()

    # Basic data-quality report.
    quality = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing_count": df.isna().sum(),
            "missing_percent": (df.isna().mean() * 100).round(2),
            "n_unique": df.nunique(dropna=False),
        }
    )
    quality.to_csv(OUTPUT_DIR / "data_quality.csv", encoding="utf-8-sig")

    # Numerical and categorical summaries.
    analysis_df.describe(include="all").T.to_csv(
        OUTPUT_DIR / "summary_statistics.csv", encoding="utf-8-sig"
    )
    df.groupby("stroke").mean(numeric_only=True).round(3).to_csv(
        OUTPUT_DIR / "numeric_by_stroke.csv", encoding="utf-8-sig"
    )

    # Imbalance baseline: always predicting the majority class.
    positive_rate = df["stroke"].mean()
    y_pred_baseline = pd.Series(0, index=df.index)
    y_score_baseline = pd.Series(positive_rate, index=df.index)
    baseline_metrics = pd.Series(
        {
            "positive_rate_percent": positive_rate * 100,
            "negative_rate_percent": (1 - positive_rate) * 100,
            "majority_accuracy_percent": (1 - positive_rate) * 100,
            "recall": recall_score(df["stroke"], y_pred_baseline, zero_division=0),
            "f1": f1_score(df["stroke"], y_pred_baseline, zero_division=0),
            "auc_pr_average_precision": average_precision_score(
                df["stroke"], y_score_baseline
            ),
        }
    )
    baseline_metrics.to_frame("value").to_csv(
        OUTPUT_DIR / "imbalance_baseline_metrics.csv", encoding="utf-8-sig"
    )

    categorical_cols = analysis_df.select_dtypes(include="object").columns
    category_rates = []
    for col in categorical_cols:
        grouped = (
            df.groupby(col, dropna=False)["stroke"]
            .agg(count="size", strokes="sum", stroke_rate="mean")
            .reset_index()
        )
        grouped["feature"] = col
        grouped["stroke_rate_percent"] = (grouped["stroke_rate"] * 100).round(2)
        category_rates.append(grouped)
    pd.concat(category_rates, ignore_index=True).to_csv(
        OUTPUT_DIR / "categorical_stroke_rates.csv", index=False, encoding="utf-8-sig"
    )

    # 1. Target distribution.
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=df, x="stroke")
    ax.set(title="Stroke target distribution", xlabel="Stroke", ylabel="Count")
    for container in ax.containers:
        ax.bar_label(container)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_target_distribution.png", dpi=150)
    plt.close()

    # 2. Numerical distributions by target.
    numerical_cols = ["age", "avg_glucose_level", "bmi"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, col in zip(axes, numerical_cols):
        sns.histplot(data=df, x=col, hue="stroke", kde=True, stat="density",
                     common_norm=False, element="step", ax=ax)
        ax.set_title(f"{col} by stroke")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_numeric_distributions.png", dpi=150)
    plt.close()

    # 3. Boxplots for outlier and group comparison.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, numerical_cols):
        sns.boxplot(data=df, x="stroke", y=col, ax=ax)
        ax.set_title(f"{col} vs stroke")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_boxplots_by_stroke.png", dpi=150)
    plt.close()

    # 4. Stroke rate by categorical features.
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for ax, col in zip(axes.flat, categorical_cols):
        rates = df.groupby(col, dropna=False)["stroke"].mean().mul(100).sort_values(ascending=False)
        sns.barplot(x=rates.values, y=rates.index, ax=ax)
        ax.set(title=f"Stroke rate by {col}", xlabel="Stroke rate (%)", ylabel="")
    axes.flat[-1].axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_categorical_stroke_rates.png", dpi=150)
    plt.close()

    # 5. Correlation heatmap for numeric variables.
    plt.figure(figsize=(9, 7))
    sns.heatmap(analysis_df.select_dtypes(exclude="object").corr(), annot=True,
                fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Numeric feature correlation")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_correlation_heatmap.png", dpi=150)
    plt.close()

    print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Missing BMI values: {df['bmi'].isna().sum():,}")
    print(f"Duplicate rows: {df.duplicated().sum():,}")
    print("Stroke distribution (%):")
    print((df["stroke"].value_counts(normalize=True) * 100).round(2).to_string())
    print(f"\nEDA outputs saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
