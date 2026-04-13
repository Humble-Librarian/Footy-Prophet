import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import poisson

PROCESSED = Path("data/processed")
MODELS = Path("models")
MODELS.mkdir(exist_ok=True)

# Dixon-Coles parameters
RHO = 0  # Correlation parameter between home and away goals

def dixon_coles_likelihood(params, df, teams):
    n_teams = len(teams)
    # params: [home_adv] + [attack_0 ... attack_N] + [defend_0 ... defend_N]
    if len(params) != 2 * n_teams + 1:
        return 1e9

    home_adv = params[0]
    attacks = {team: params[1 + i] for i, team in enumerate(teams)}
    defends = {team: params[1 + n_teams + i] for i, team in enumerate(teams)}

    # Vectorized computation for speed
    home_attack = df['Home'].map(attacks).values
    home_defend = df['Home'].map(defends).values
    away_attack = df['Away'].map(attacks).values
    away_defend = df['Away'].map(defends).values

    # Expected goals
    lambda_home = np.exp(home_attack + away_defend + home_adv)
    lambda_away = np.exp(away_attack + home_defend)

    hg = df['HomeGoals'].values
    ag = df['AwayGoals'].values

    # Log Likelihood of independent Poisson distributions
    # We ignore the low-scoring dependency term (rho) for stable MLE estimation here
    llik = poisson.logpmf(hg, lambda_home) + poisson.logpmf(ag, lambda_away)
    
    return -np.sum(llik)

def train_dixon_coles():
    print("Loading clean dataset for Dixon-Coles Probability Engine...")
    df = pd.read_csv(PROCESSED / "features_clean.csv", parse_dates=["Date"])
    # Filter to relatively recent seasons if needed, but since we only have 5 seasons, all is fine.
    # DC models are often weighted, but simple MLE works well enough for baselines.
    df = df.dropna(subset=['HomeGoals', 'AwayGoals'])
    df['HomeGoals'] = df['HomeGoals'].astype(int)
    df['AwayGoals'] = df['AwayGoals'].astype(int)

    teams = sorted(list(set(df['Home'].unique()) | set(df['Away'].unique())))
    n_teams = len(teams)
    
    print(f"Fitting team strengths for {n_teams} teams...")

    # Initial guess
    # home_adv = 0.2, attack_str = 1.0, defend_str = 0.0
    init_params = np.concatenate([[0.2], np.ones(n_teams), np.zeros(n_teams)])

    # We must constrain the average attack strength to 1.0 (or sum to N) to allow identifiability
    def constraint_func(params):
        return np.mean(params[1:n_teams+1]) - 1.0

    constraints = [{'type': 'eq', 'fun': constraint_func}]
    
    # Simple bounds: Attack > 0
    bounds = [(None, None)] + [(0.01, 5.0)] * n_teams + [(None, None)] * n_teams

    opt = minimize(
        dixon_coles_likelihood,
        init_params,
        args=(df, teams),
        constraints=constraints,
        bounds=bounds,
        method='SLSQP',
        options={'maxiter': 200, 'disp': False}
    )

    if opt.success:
        print("Dixon-Coles MLE Converged Successfully!")
        home_adv = opt.x[0]
        attack_params = {team: opt.x[1 + i] for i, team in enumerate(teams)}
        defend_params = {team: opt.x[1 + n_teams + i] for i, team in enumerate(teams)}

        dc_model = {
            "home_adv": home_adv,
            "attack": attack_params,
            "defend": defend_params,
            "teams": teams
        }
        joblib.dump(dc_model, MODELS / "dixon_coles.pkl")
        print("Saved -> models/dixon_coles.pkl")
    else:
        print("Warning: Optimization failed.")
        print(opt.message)

if __name__ == "__main__":
    train_dixon_coles()
