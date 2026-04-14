import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import joblib
from pathlib import Path

PROCESSED = Path("data/processed")
MODELS = Path("models")
MODELS.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

XG_FEATURES = [
    "H_roll_gf", "H_roll_ga", "H_roll_xg",
    "A_roll_gf", "A_roll_ga", "A_roll_xg",
    "HomePPDA", "AwayPPDA"
]

class XGModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 2)   # [HomeXG, AwayXG]
        )

    def forward(self, x):
        return self.net(x)

def train_xg_mlp():
    print(f"Training xG MLP on: {DEVICE}")
    df = pd.read_csv(PROCESSED / "features_clean.csv", parse_dates=["Date"])

    X = df[XG_FEATURES].values.astype(np.float32)
    y = df[["HomeXG_Final", "AwayXG_Final"]].values.astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    joblib.dump(scaler, MODELS / "xg_scaler.pkl")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, shuffle=False)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=64, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=64
    )

    model = XGModel(input_dim=X.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    MAX_EPOCHS = 100
    EARLY_STOP_PATIENCE = 15

    print("Training PyTorch xG Neural Network...")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(Xb), yb).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODELS / "xg_mlp.pt")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch}. Best Val MSE: {best_val_loss:.4f}")
                break

    print(f"Saved -> models/xg_mlp.pt")

    # ── Compute and save test metrics ────────────────────────────────────────
    model.load_state_dict(torch.load(MODELS / "xg_mlp.pt", map_location=DEVICE, weights_only=True))
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_val).to(DEVICE)).cpu().numpy()

    home_xg_mae = mean_absolute_error(y_val[:, 0], preds[:, 0])
    away_xg_mae = mean_absolute_error(y_val[:, 1], preds[:, 1])
    print(f"Test Home xG MAE: {home_xg_mae:.4f} | Test Away xG MAE: {away_xg_mae:.4f}")

    metrics = {
        "home_xg_mae": round(float(home_xg_mae), 4),
        "away_xg_mae": round(float(away_xg_mae), 4),
        "best_val_mse": round(float(best_val_loss), 4),
        "eval_date": str(pd.Timestamp.today().date()),
        "n_test_matches": len(y_val)
    }
    with open(MODELS / "xg_mlp_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved -> models/xg_mlp_metrics.json")

if __name__ == "__main__":
    train_xg_mlp()
