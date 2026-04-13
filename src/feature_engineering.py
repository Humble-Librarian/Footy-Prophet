import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

PROCESSED = Path("data/processed")
WINDOW = 5   # rolling window for form features

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").copy()

    # Per-team rolling stats (last N games, regardless of H/A)
    for team_col, goal_for_col, goal_against_col, xg_col in [
        ("Home", "HomeGoals", "AwayGoals", "HomeXG_Final"),
        ("Away", "AwayGoals", "HomeGoals", "AwayXG_Final"),
    ]:
        prefix = "H" if team_col == "Home" else "A"
        
        # Build a per-team time series
        teams = df[team_col].unique()
        rolling_gf, rolling_ga, rolling_xg = {}, {}, {}

        for team in teams:
            mask_h = df["Home"] == team
            mask_a = df["Away"] == team

            # Combine home and away appearances in chronological order
            home_stats = df[mask_h][["Date", "HomeGoals", "AwayGoals", "HomeXG_Final"]].copy()
            home_stats.columns = ["Date", "GF", "GA", "XG"]
            away_stats = df[mask_a][["Date", "AwayGoals", "HomeGoals", "AwayXG_Final"]].copy()
            away_stats.columns = ["Date", "GF", "GA", "XG"]

            all_stats = pd.concat([home_stats, away_stats]).sort_values("Date")
            all_stats["roll_gf"] = all_stats["GF"].shift(1).rolling(WINDOW, min_periods=1).mean()
            all_stats["roll_ga"] = all_stats["GA"].shift(1).rolling(WINDOW, min_periods=1).mean()
            all_stats["roll_xg"] = all_stats["XG"].shift(1).rolling(WINDOW, min_periods=1).mean()

            for _, row in all_stats.iterrows():
                rolling_gf[(team, row["Date"])] = row["roll_gf"]
                rolling_ga[(team, row["Date"])] = row["roll_ga"]
                rolling_xg[(team, row["Date"])] = row["roll_xg"]

        df[f"{prefix}_roll_gf"] = df.apply(
            lambda r: rolling_gf.get((r[team_col], r["Date"]), np.nan), axis=1)
        df[f"{prefix}_roll_ga"] = df.apply(
            lambda r: rolling_ga.get((r[team_col], r["Date"]), np.nan), axis=1)
        df[f"{prefix}_roll_xg"] = df.apply(
            lambda r: rolling_xg.get((r[team_col], r["Date"]), np.nan), axis=1)

    return df

def add_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").copy()
    h2h_home_goals, h2h_away_goals, h2h_count = [], [], []

    for idx, row in df.iterrows():
        home, away, date = row["Home"], row["Away"], row["Date"]
        past = df[
            (((df["Home"] == home) & (df["Away"] == away)) |
             ((df["Home"] == away) & (df["Away"] == home))) &
            (df["Date"] < date)
        ]
        if len(past) == 0:
            h2h_home_goals.append(np.nan)
            h2h_away_goals.append(np.nan)
            h2h_count.append(0)
        else:
            last5 = past.tail(5)
            # Normalize: always from the perspective of current home team
            hg = last5.apply(
                lambda r: r["HomeGoals"] if r["Home"] == home else r["AwayGoals"], axis=1
            ).mean()
            ag = last5.apply(
                lambda r: r["AwayGoals"] if r["Home"] == home else r["HomeGoals"], axis=1
            ).mean()
            h2h_home_goals.append(hg)
            h2h_away_goals.append(ag)
            h2h_count.append(len(past))

    df["H2H_HomeGoals_Avg"] = h2h_home_goals
    df["H2H_AwayGoals_Avg"] = h2h_away_goals
    df["H2H_Count"] = h2h_count
    return df

def add_result_target(df: pd.DataFrame) -> pd.DataFrame:
    # Targets
    df["TotalGoals"] = df["HomeGoals"] + df["AwayGoals"]
    df["Result"] = df.apply(
        lambda r: 1 if r["HomeGoals"] > r["AwayGoals"]
                  else (-1 if r["HomeGoals"] < r["AwayGoals"] else 0), axis=1
    )
    return df

FEATURE_COLS = [
    "H_roll_gf", "H_roll_ga", "H_roll_xg",
    "A_roll_gf", "A_roll_ga", "A_roll_xg",
    "H2H_HomeGoals_Avg", "H2H_AwayGoals_Avg", "H2H_Count",
    "HomePPDA", "AwayPPDA"
    # Note: HomePoss, AwayPoss, HomeSoT, AwaySoT are dropped as soccerdata read_schedule doesn't fetch them reliably by default.
]

def build_features():
    matches = pd.read_csv(PROCESSED / "matches.csv", parse_dates=["Date"])
    print("Base matches shape:", matches.shape)
    
    print("Adding rolling features...")
    matches = add_rolling_features(matches)
    
    print("Adding H2H features...")
    matches = add_h2h_features(matches)
    
    print("Generating Targets...")
    matches = add_result_target(matches)
    
    # Save standard feature engineered dataset
    matches.to_csv(PROCESSED / "features.csv", index=False)
    
    # Save a clean ML-ready slice with drops
    ml_ready = matches.dropna(subset=FEATURE_COLS + ["HomeGoals", "AwayGoals"])
    ml_ready.to_csv(PROCESSED / "features_clean.csv", index=False)
    
    print(f"Feature set saved → data/processed/features.csv")
    print(f"Shape with NAs: {matches.shape}")
    print(f"ML Ready Clean Shape: {ml_ready.shape}")
    return matches

if __name__ == "__main__":
    build_features()
