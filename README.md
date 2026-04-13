# FootProphet ⚽🔮

**FootProphet** is an automated, multi-model football prediction engine designed to forecast match outcomes, expected goals (xG), and final results for the English Premier League and Spanish La Liga. 

Built with a modular Python CLI, it combines state-of-the-art machine learning (LightGBM & PyTorch) with classical statistical modeling (Dixon-Coles) to provide a data-driven edge.

---

## 🏗️ Architecture

FootProphet follows a 4-layer modular architecture:

1.  **Data Layer (`soccerdata` integration)**: Orchestrates automated scraping from FBref and Understat for historical match stats, xG data, and upcoming schedules across 5+ seasons.
2.  **Feature Layer (Engineering)**: Transforms raw match data into "form" metrics using rolling windows (last 5 games), Head-to-Head (H2H) historical averages, and defensive pressure metrics (PPDA).
3.  **Inference Layer (Model Ensemble)**:
    *   **LightGBM Regressors**: Predicts the most likely integer score for Home and Away goals.
    *   **PyTorch NN**: A deep neural network that predicts granular **Expected Goals (xG)** based on current team form.
    *   **Dixon-Coles Solver**: A statistical distribution model that calculates the discrete probability of a Win, Draw, or Loss.
4.  **CLI Layer (`main.py`)**: A unified terminal interface using `rich` for beautiful, formatted tables and `argparse` for command routing.

---

## 📁 File Structure

```bash
FootProphet/
├── data/
│   └── processed/          # Standardized CSVs used for training/inference
├── models/                 # Pre-trained .pkl and .pt binary files
├── src/
│   ├── data_pipeline.py    # Scraping & team-name normalization
│   ├── feature_engineering.py # Rolling form & H2H calculations
│   ├── train_lgbm.py       # Optuna-tuned LightGBM training
│   ├── train_xg_mlp.py     # PyTorch Neural Network training
│   ├── train_dixon_coles.py # Statistical MLE distribution solver
│   └── predict.py          # Unified inference wrapper
├── main.py                 # CLI Entry Point
├── requirements.txt        # Project dependencies
└── README.md
```

---

## 🚀 How to Run

### 1. Setup Environment
First, create and activate a virtual environment, then install the dependencies:
```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Retrain the Engine (Optional)
If you want to fetch the latest scores from the previous weekend and re-train all models:
```bash
python main.py retrain
```
*Note: This will scrape ~4,000 matches and optimize hyperparameters using Optuna.*

### 3. Predict a Match
Run a prediction for any team pairing within the supported leagues:
```bash
python main.py predict --home "Chelsea" --away "Arsenal"
```

---

## 📊 Output Type

FootProphet provides three distinct data points for every prediction:

1.  **Predicted Score (Integer)**: The most likely final scoreline (e.g., `2 - 1`) based on regression.
2.  **Expected Goals (xG)**: The volume and quality of chances expected for each team (e.g., `1.85 - 1.30`) based on deep learning.
3.  **Outcome Probabilities**: A formatted table showing the calculated percentage chance for a **Home Win**, **Draw**, or **Away Win**.

Example CLI Output:
```text
Match Prediction: Chelsea vs Arsenal
Predicted Score: 2 - 1
Expected Goals (xG): 1.46 - 1.64

+---------------------------+
| Outcome     | Probability |
|-------------+-------------|
| Chelsea Win | 39.6%       |
| Draw        | 28.1%       |
| Arsenal Win | 32.3%       |
+---------------------------+
```

---

## 🛠️ Requirements
- **Python**: 3.9+
- **Hardware**: CPU-friendly (though PyTorch will use CUDA if a GPU is detected).
- **Core Stacks**: `pandas`, `lightgbm`, `torch`, `optuna`, `soccerdata`, `scipy`.

---
*Disclaimer: This tool is for informational/entertainment purposes only. Sports involve high variance, and prediction models should not be used as the sole basis for gambling.*
