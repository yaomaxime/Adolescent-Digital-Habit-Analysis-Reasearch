"""
Adolescent Digital Habit Analysis
=================================

Re-analysis of the raw dataset from Pérez-Chada et al. (2023, PLOS ONE),
testing whether *total* screen time or *device type* better explains
adolescent sleep debt and classroom attention problems.

Analyses
--------
1. Spearman correlation between each device type and attention problems.
2. Spearman correlation between total screen time and weekend sleep debt.
3. K-means clustering of students into behavioural profiles, followed by a
   Kruskal-Wallis H-test and Dunn's post-hoc test (Bonferroni-corrected)
   on attention scores across profiles.

Usage
-----
    python analysis.py --data data/repositorio_translated.csv --out figures
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scikit_posthocs as sp
import seaborn as sns
from scipy.stats import kruskal, spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

COL_ATTENTION = "AttentionProblems_RECODED"  # ordinal: 1 = best, 4 = worst
COL_TOTAL_SCREEN = "TotalScreenTime_hrs"
COL_WEEKDAY_SLEEP = "WeekdaySleepDuration"
COL_WEEKEND_SLEEP = "WeekendSleepDuration"

COL_VIDEO_GAMES = "VideoGameTime_hrs"
COL_SOCIAL_MEDIA = "SocialMediaTime_hrs"
SCREEN_TYPES = [
    COL_VIDEO_GAMES,
    "TabletPhoneGameTime_hrs",
    COL_SOCIAL_MEDIA,
    "TVOnlineTime_hrs",
]

N_CLUSTERS = 3
K_CANDIDATES = [2, 3, 4]
RANDOM_STATE = 42
ALPHA = 0.05


def section(title: str) -> None:
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_and_clean(path: Path) -> pd.DataFrame:
    """Load the dataset, remove duplicates and incomplete rows."""
    df = pd.read_csv(path)
    n_raw = len(df)

    n_duplicates = df.duplicated().sum()
    df = df.drop_duplicates()

    required = [COL_ATTENTION, COL_TOTAL_SCREEN,
                COL_WEEKDAY_SLEEP, COL_WEEKEND_SLEEP] + SCREEN_TYPES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in dataset: {missing}")

    df = df[required].apply(pd.to_numeric, errors="coerce").dropna()

    print(f"Raw rows:            {n_raw}")
    print(f"Duplicates removed:  {n_duplicates}")
    print(f"Rows analysed (N):   {len(df)}")
    return df


# --------------------------------------------------------------------------
# Analysis 1: device type vs attention
# --------------------------------------------------------------------------

def analyse_device_types(df: pd.DataFrame, out: Path) -> None:
    section("1. DEVICE TYPE vs ATTENTION PROBLEMS (Spearman)")

    for screen in SCREEN_TYPES:
        r, p = spearmanr(df[screen], df[COL_ATTENTION])
        sig = "significant" if p < ALPHA else "not significant"
        print(f"{screen:<26} r = {r:+.3f}  R² = {r**2:.4f}  p = {p:.4g} ({sig})")

    corr = df[SCREEN_TYPES + [COL_ATTENTION]].corr(method="spearman")

    # Full correlation matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Reds")
    plt.title("Spearman Correlation: Screen Types vs Attention Problems", pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out / "fig1_screen_types_heatmap.png", dpi=200)
    plt.close()

    # Targeted 1-D heatmap: each device type against attention only
    targeted = corr[[COL_ATTENTION]].drop(COL_ATTENTION)
    plt.figure(figsize=(4, 6))
    sns.heatmap(targeted, annot=True, fmt=".3f", cmap="Reds", vmin=-0.5, vmax=0.5)
    plt.title("Correlation with Attention Problems", pad=15)
    plt.ylabel("Screen time type")
    plt.tight_layout()
    plt.savefig(out / "fig2_attention_targeted_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------
# Analysis 2: total screen time vs sleep debt
# --------------------------------------------------------------------------

def analyse_sleep_debt(df: pd.DataFrame, out: Path) -> None:
    section("2. TOTAL SCREEN TIME vs WEEKEND SLEEP DEBT (Spearman)")

    # Sleep debt proxy: extra sleep taken at the weekend to catch up
    df["SleepDebt"] = df[COL_WEEKEND_SLEEP] - df[COL_WEEKDAY_SLEEP]

    r, p = spearmanr(df[COL_TOTAL_SCREEN], df["SleepDebt"])
    sig = "significant" if p < ALPHA else "not significant"
    print(f"r = {r:+.3f}  R² = {r**2:.4f}  p = {p:.4g} ({sig})")

    plt.figure(figsize=(8, 5))
    sns.regplot(data=df, x=COL_TOTAL_SCREEN, y="SleepDebt",
                scatter_kws={"alpha": 0.5}, line_kws={"color": "purple"})
    plt.axhline(0, color="black", linestyle="--")
    plt.title("Total Screen Time vs Weekend Sleep Debt")
    plt.xlabel("Total daily screen time (hours)")
    plt.ylabel("Sleep debt (weekend − weekday sleep, hours)")
    plt.tight_layout()
    plt.savefig(out / "fig3_sleep_debt_scatter.png", dpi=200)
    plt.close()


# --------------------------------------------------------------------------
# Analysis 3: behavioural profiles
# --------------------------------------------------------------------------

def name_clusters(df: pd.DataFrame, labels) -> dict:
    """
    Name clusters from their centroids rather than hard-coding label numbers,
    since K-means assigns cluster IDs arbitrarily.

    - Highest mean social-media time         -> "Heavy Social Media Users"
    - Highest mean video-game time (of rest) -> "Heavy Gamers"
    - Remaining cluster                      -> "Balanced Users"
    """
    means = df[SCREEN_TYPES].groupby(labels).mean()

    social_id = means[COL_SOCIAL_MEDIA].idxmax()
    gamer_id = means.drop(index=social_id)[COL_VIDEO_GAMES].idxmax()

    names = {}
    for cluster_id in means.index:
        if cluster_id == social_id:
            names[cluster_id] = "Heavy Social Media Users"
        elif cluster_id == gamer_id:
            names[cluster_id] = "Heavy Gamers"
        else:
            names[cluster_id] = "Balanced Users"
    return names


def analyse_profiles(df: pd.DataFrame, out: Path) -> None:
    section("3. BEHAVIOURAL PROFILES (K-means + Kruskal-Wallis + Dunn)")

    # Standardise so no single device type dominates the Euclidean distance
    X = StandardScaler().fit_transform(df[SCREEN_TYPES])

    print("Silhouette scores:")
    for k in K_CANDIDATES:
        labels_k = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(X)
        print(f"  k = {k}: {silhouette_score(X, labels_k):.3f}")
    print(f"Using k = {N_CLUSTERS} to separate active gamers from passive social media users.")

    labels = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10).fit_predict(X)
    names = name_clusters(df, labels)
    df["Profile"] = pd.Series(labels, index=df.index).map(names)

    print("\nMean daily hours per profile:")
    print(df.groupby("Profile")[SCREEN_TYPES].mean().round(2))
    print("\nProfile sizes:")
    print(df["Profile"].value_counts())

    # Attention is ordinal and non-normal -> non-parametric tests
    groups = [g[COL_ATTENTION].values for _, g in df.groupby("Profile")]
    h, p = kruskal(*groups)
    sig = "significant difference" if p < ALPHA else "no significant difference"
    print(f"\nKruskal-Wallis: H = {h:.3f}, p = {p:.4g} ({sig})")

    print("\nDunn's post-hoc test (Bonferroni-adjusted p-values):")
    print(sp.posthoc_dunn(df, val_col=COL_ATTENTION, group_col="Profile",
                          p_adjust="bonferroni").round(4))

    order = ["Balanced Users", "Heavy Social Media Users", "Heavy Gamers"]
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Profile", y=COL_ATTENTION, hue="Profile",
                order=order, palette="Set2", legend=False)
    plt.xlabel("Behavioural profile")
    plt.ylabel("Attention problem score (1 = best, 4 = worst)")
    plt.title("Attention Problems by Digital Profile")
    plt.tight_layout()
    plt.savefig(out / "fig4_attention_by_profile.png", dpi=200)
    plt.close()


# --------------------------------------------------------------------------
# Summary statistics
# --------------------------------------------------------------------------

def summary_statistics(df: pd.DataFrame) -> None:
    section("SUMMARY STATISTICS")
    print(df[SCREEN_TYPES + [COL_ATTENTION]].agg(["mean", "median", "std"]).round(2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/repositorio_translated.csv"),
                        help="Path to the dataset CSV")
    parser.add_argument("--out", type=Path, default=Path("figures"),
                        help="Directory for output figures")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)

    df = load_and_clean(args.data)
    summary_statistics(df)
    analyse_device_types(df, args.out)
    analyse_sleep_debt(df, args.out)
    analyse_profiles(df, args.out)

    print(f"\nFigures saved to: {args.out.resolve()}")


if __name__ == "__main__":
    main()
