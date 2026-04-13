import numpy as np
import pandas as pd
import joblib
import torch
from pathlib import Path
from scipy.stats import poisson
from src.train_xg_mlp import XGModel, XG_FEATURES, DEVICE
from src.train_lgbm import FEATURE_COLS

MODELS = Path("models")

class MatchPredictor:
    def __init__(self):
        # We wrap in a class/lazy logic so it doesn't crash on import if models aren't trained yet
        self._lgbm_home = None
        self._lgbm_away = None
        self._xg_scaler = None
        self._dc_params  = None
        self._xg_model = None
        self.is_loaded = False

    def load_models(self):
        if self.is_loaded:
            return True
            
        try:
            self._lgbm_home = joblib.load(MODELS / "lgbm_home.pkl")
            self._lgbm_away = joblib.load(MODELS / "lgbm_away.pkl")
            self._xg_scaler = joblib.load(MODELS / "xg_scaler.pkl")
            self._dc_params  = joblib.load(MODELS / "dixon_coles.pkl")

            self._xg_model = XGModel(input_dim=len(XG_FEATURES)).to(DEVICE)
            self._xg_model.load_state_dict(torch.load(MODELS / "xg_mlp.pt", map_location=DEVICE, weights_only=True))
            self._xg_model.eval()
            self.is_loaded = True
            return True
        except FileNotFoundError:
            return False

    def predict(self, features: dict) -> dict:
        if not self.load_models():
            raise FileNotFoundError("Models not found. Run training first.")
            
        # ── LightGBM score prediction ────────────────────────────────────────────
        X_lgbm = pd.DataFrame([{k: features.get(k, 0.0) for k in FEATURE_COLS}])
        home_goals = max(0.0, float(self._lgbm_home.predict(X_lgbm)[0]))
        away_goals = max(0.0, float(self._lgbm_away.predict(X_lgbm)[0]))

        # ── PyTorch xG prediction ────────────────────────────────────────────────
        X_xg = np.array([[features.get(k, 0.0) for k in XG_FEATURES]], dtype=np.float32)
        X_xg = self._xg_scaler.transform(X_xg)
        with torch.no_grad():
            xg_out = self._xg_model(torch.tensor(X_xg).to(DEVICE)).cpu().numpy()[0]
        home_xg = max(0.0, float(xg_out[0]))
        away_xg = max(0.0, float(xg_out[1]))

        # ── Dixon-Coles W/D/L probabilities ─────────────────────────────────────
        home_team = features.get("HomeTeam", "")
        away_team = features.get("AwayTeam", "")
        win_prob, draw_prob, loss_prob = self.dixon_coles_probs(home_team, away_team)

        return {
            "home_goals": round(home_goals, 2),
            "away_goals": round(away_goals, 2),
            "home_xg": round(home_xg, 2),
            "away_xg": round(away_xg, 2),
            "win_prob": round(win_prob, 3),
            "draw_prob": round(draw_prob, 3),
            "loss_prob": round(loss_prob, 3),
        }

    def dixon_coles_probs(self, home_team: str, away_team: str, max_goals: int = 10):
        p = self._dc_params
        if home_team not in p["attack"] or away_team not in p["attack"]:
            # Fallback: league average (basic 45-27-28 split)
            return 0.45, 0.27, 0.28

        # Recalculate lambda values without rho
        # Rho (dependency factor) requires rho optimization in training, which we omitted for stability
        # We will use simple poisson expectations
        mu_h = np.exp(p["attack"][home_team] - p["defend"][away_team] + p["home_adv"])
        mu_a = np.exp(p["attack"][away_team] - p["defend"][home_team])

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
            return 0.45, 0.27, 0.28
        return win / total, draw / total, loss / total

predictor = MatchPredictor()
