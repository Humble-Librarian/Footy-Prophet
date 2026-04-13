import pandas as pd
import numpy as np
import soccerdata as sd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

LEAGUES = ["ENG-Premier League", "ESP-La Liga"]
SEASONS = ["20/21", "21/22", "22/23", "23/24", "24/25"]

def fetch_and_merge():
    # 1. Fetch FBref
    print(f"Fetching FBref data for {LEAGUES} ({SEASONS})...")
    fbref = sd.FBref(leagues=LEAGUES, seasons=SEASONS)
    fbref_matches = fbref.read_schedule()
    if isinstance(fbref_matches.index, pd.MultiIndex):
        fbref_matches = fbref_matches.reset_index()
    
    print("FBref fetched columns:", fbref_matches.columns.tolist())
    
    # 2. Fetch Understat
    print(f"\nFetching Understat data for {LEAGUES} ({SEASONS})...")
    understat = sd.Understat(leagues=LEAGUES, seasons=SEASONS)
    us_matches = understat.read_team_match_stats()
    if isinstance(us_matches.index, pd.MultiIndex):
        us_matches = us_matches.reset_index()
        
    print("Understat fetched columns:", us_matches.columns.tolist())

    # Safely print a raw sample
    print("\nFBref sample head:\n", fbref_matches.head(2))
    print("\nUnderstat sample head:\n", us_matches.head(2))
    
    # Let's do some basic mapping depending on actual columns returned
    # Usually: fbref returns ['league', 'season', 'game', 'date', 'home_team', 'away_team', 'home_xg', 'away_xg', 'home_score', 'away_score', ...]
    # Understat returns ['league', 'season', 'game', 'date', 'home_team', 'away_team', 'home_goals', 'away_goals', 'home_xg', 'away_xg', 'home_ppda', 'away_ppda', ...]
    
    # To keep this robust during the test phase, let's rename known columns
    fb_map = {
        "date": "Date", "home_team": "Home", "away_team": "Away",
        "home_score": "HomeGoals", "away_score": "AwayGoals",
        "home_xg": "HomeXG", "away_xg": "AwayXG",
        "league": "League", "season": "Season"
    }
    fbref_matches = fbref_matches.rename(columns=lambda x: fb_map.get(x, x))
    
    us_map = {
        "date": "Date", "home_team": "Home", "away_team": "Away",
        "home_xg": "HomeXG_US", "away_xg": "AwayXG_US",
        "home_ppda": "HomePPDA", "away_ppda": "AwayPPDA"
    }
    us_matches = us_matches.rename(columns=lambda x: us_map.get(x, x))
    
    # Normalize team names between FBref and Understat
    # FBref -> Understat mapping
    TEAM_MAPPING = {
        "Alavés": "Alaves", "Almería": "Almeria", "Atlético Madrid": "Atletico Madrid",
        "Cádiz": "Cadiz", "Huesca": "SD Huesca", "Ipswich Town": "Ipswich",
        "Leeds United": "Leeds", "Leganés": "Leganes", "Leicester City": "Leicester",
        "Luton Town": "Luton", "Manchester Utd": "Manchester United", "Norwich City": "Norwich",
        "Tottenham Hotspur": "Tottenham", "Valladolid": "Real Valladolid", 
        "West Brom": "West Bromwich Albion", "West Ham United": "West Ham", 
        "Wolves": "Wolverhampton Wanderers"
    }
    fbref_matches['Home'] = fbref_matches['Home'].replace(TEAM_MAPPING)
    fbref_matches['Away'] = fbref_matches['Away'].replace(TEAM_MAPPING)

    # Parse 'score' string (e.g., '2–1') into HomeGoals and AwayGoals
    if 'score' in fbref_matches.columns:
        fbref_matches = fbref_matches.dropna(subset=['score'])
        # Handle different dash characters
        score_split = fbref_matches['score'].astype(str).str.replace('–', '-').str.split('-', expand=True)
        fbref_matches['HomeGoals'] = pd.to_numeric(score_split[0], errors='coerce')
        fbref_matches['AwayGoals'] = pd.to_numeric(score_split[1], errors='coerce')

    # Ensure Date format
    if "Date" in fbref_matches.columns:
        fbref_matches['Date'] = pd.to_datetime(fbref_matches['Date']).dt.strftime('%Y-%m-%d')
    if "Date" in us_matches.columns:
        us_matches['Date'] = pd.to_datetime(us_matches['Date']).dt.strftime('%Y-%m-%d')
    
    # Let's temporarily save raw to see structure if merge breaks
    fbref_matches.to_csv(PROCESSED / "raw_fbref.csv", index=False)
    us_matches.to_csv(PROCESSED / "raw_understat.csv", index=False)
    
    print("\nMerging datasets...")
    # Base merge on Date, Home
    merged = pd.merge(
        fbref_matches,
        us_matches[['Date', 'Home', 'HomeXG_US', 'AwayXG_US', 'HomePPDA', 'AwayPPDA']],
        on=['Date', 'Home'],
        how='left'
    )
    
    # Fill defaults if missing
    for col in ["HomeXG", "AwayXG", "HomeXG_US", "AwayXG_US"]:
        if col not in merged.columns:
            merged[col] = np.nan
            
    # Calculate Final xG
    merged["HomeXG_Final"] = merged[["HomeXG", "HomeXG_US"]].mean(axis=1)
    merged["AwayXG_Final"] = merged[["AwayXG", "AwayXG_US"]].mean(axis=1)
    
    # Clean up
    if "HomeGoals" in merged.columns and "AwayGoals" in merged.columns:
        merged = merged.dropna(subset=["HomeGoals", "AwayGoals"])
    else:
        print("Warning: Missing goals columns!")
        
    merged = merged.sort_values("Date").reset_index(drop=True)
    return merged

def run_pipeline():
    matches = fetch_and_merge()
    matches.to_csv(PROCESSED / "matches.csv", index=False)
    print(f"\nFinal processing complete! Saved {len(matches)} matches to data/processed/matches.csv")

if __name__ == "__main__":
    run_pipeline()
