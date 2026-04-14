import pandas as pd
import numpy as np
import json
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

def predict_outcome(dc_model, home, away, max_goals=10):
    """Predict W/D/L probabilities for a single match using fitted DC params."""
    if home not in dc_model["attack"] or away not in dc_model["attack"]:
        return None
    mu_h = np.exp(dc_model["attack"][home] - dc_model["defend"][away] + dc_model["home_adv"])
    mu_a = np.exp(dc_model["attack"][away] - dc_model["defend"][home])

    win = draw = loss = 0.0
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            prob = poisson.pmf(hg, mu_h) * poisson.pmf(ag, mu_a)
            if hg > ag:
                win += prob
            elif hg == ag:
                draw += prob
            else:
                loss += prob
    total = win + draw + loss
    if total == 0:
        return None
    return win / total, draw / total, loss / total

def train_dixon_coles():
    print("Loading clean dataset for Dixon-Coles Probability Engine...")
    df = pd.read_csv(PROCESSED / "features_clean.csv", parse_dates=["Date"])
    df = df.sort_values("Date")
    # Filter to relatively recent seasons if needed, but since we only have 5 seasons, all is fine.
    # DC models are often weighted, but simple MLE works well enough for baselines.
    df = df.dropna(subset=['HomeGoals', 'AwayGoals'])
    df['HomeGoals'] = df['HomeGoals'].astype(int)
    df['AwayGoals'] = df['AwayGoals'].astype(int)

    # Hold out last 15% for evaluation
    split_idx = int(len(df) * 0.85)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]

    teams = sorted(list(set(df_train['Home'].unique()) | set(df_train['Away'].unique())))
    n_teams = len(teams)
    
    print(f"Fitting team strengths for {n_teams} teams on {len(df_train)} training matches...")

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
        args=(df_train, teams),
        constraints=constraints,
        bounds=bounds,
        method='SLSQP',
        options={'maxiter': 200, 'disp': False}
    )

    if opt.success:
        print("Dixon-Coles MLE Converged Successfully!")
    else:
        print(f"Warning: Optimization may not have fully converged: {opt.message}")

    home_adv = opt.x[0]
    attack_params = {team: opt.x[1 + i] for i, team in enumerate(teams)}
    defend_params = {team: opt.x[1 + n_teams + i] for i, team in enumerate(teams)}

    dc_model = {
        "home_adv": home_adv,
        "attack": attack_params,
        "defend": defend_params,
        "teams": teams
    }

    # ── Evaluate on held-out test set ────────────────────────────────────────
    print(f"Evaluating on {len(df_test)} test matches...")
    correct = 0
    total_ll = 0.0
    n_evaluated = 0

    for _, row in df_test.iterrows():
        home, away = row["Home"], row["Away"]
        probs = predict_outcome(dc_model, home, away)
        if probs is None:
            continue

        win_p, draw_p, loss_p = probs
        actual_hg, actual_ag = int(row["HomeGoals"]), int(row["AwayGoals"])

        # Actual outcome
        if actual_hg > actual_ag:
            actual_label = "W"
            actual_prob = win_p
        elif actual_hg == actual_ag:
            actual_label = "D"
            actual_prob = draw_p
        else:
            actual_label = "L"
            actual_prob = loss_p

        # Predicted outcome (highest probability)
        pred_label = ["W", "D", "L"][np.argmax([win_p, draw_p, loss_p])]
        if pred_label == actual_label:
            correct += 1

        # Log-loss contribution (clamp to avoid log(0))
        total_ll -= np.log(max(actual_prob, 1e-10))
        n_evaluated += 1

    if n_evaluated > 0:
        accuracy = correct / n_evaluated
        avg_log_loss = total_ll / n_evaluated
        print(f"Outcome Accuracy: {accuracy:.4f} ({correct}/{n_evaluated})")
        print(f"Average Log-Loss: {avg_log_loss:.4f}")
    else:
        accuracy = 0.0
        avg_log_loss = 0.0
        print("Warning: No test matches could be evaluated (teams not in training set)")

    # ── Refit on ALL data for production model ───────────────────────────────
    print("Refitting on full dataset for production model...")
    all_teams = sorted(list(set(df['Home'].unique()) | set(df['Away'].unique())))
    n_all = len(all_teams)
    init_all = np.concatenate([[0.2], np.ones(n_all), np.zeros(n_all)])
    def constraint_all(params):
        return np.mean(params[1:n_all+1]) - 1.0
    bounds_all = [(None, None)] + [(0.01, 5.0)] * n_all + [(None, None)] * n_all

    opt_full = minimize(
        dixon_coles_likelihood, init_all, args=(df, all_teams),
        constraints=[{'type': 'eq', 'fun': constraint_all}],
        bounds=bounds_all, method='SLSQP', options={'maxiter': 200, 'disp': False}
    )

    dc_prod = {
        "home_adv": opt_full.x[0],
        "attack": {team: opt_full.x[1 + i] for i, team in enumerate(all_teams)},
        "defend": {team: opt_full.x[1 + n_all + i] for i, team in enumerate(all_teams)},
        "teams": all_teams
    }
    joblib.dump(dc_prod, MODELS / "dixon_coles.pkl")
    print("Saved -> models/dixon_coles.pkl")

    # ── Save metrics ─────────────────────────────────────────────────────────
    metrics = {
        "accuracy": round(float(accuracy), 4),
        "log_loss": round(float(avg_log_loss), 4),
        "eval_date": str(pd.Timestamp.today().date()),
        "n_test_matches": int(n_evaluated)
    }
    with open(MODELS / "dixon_coles_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved -> models/dixon_coles_metrics.json")

if __name__ == "__main__":
    train_dixon_coles()
