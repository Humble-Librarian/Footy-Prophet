import streamlit as st
import pandas as pd
from src.predict import predictor

st.set_page_config(page_title="Footy Prophet", page_icon="⚽", layout="centered")
st.title("⚽ Footy Prophet")
st.caption("Multi-model football match prediction — LightGBM + PyTorch xG + Dixon-Coles")

df = pd.read_csv("data/processed/features_clean.csv", parse_dates=["Date"])
all_teams = sorted(set(df["Home"].unique()) | set(df["Away"].unique()))

col1, col2 = st.columns(2)
with col1:
    home = st.selectbox("Home Team", all_teams)
with col2:
    away = st.selectbox("Away Team", all_teams, index=1)

if st.button("Predict Match", type="primary"):
    if home == away:
        st.error("Home and Away teams must be different.")
    else:
        with st.spinner("Running prediction engine..."):
            home_matches = df[df["Home"] == home].sort_values("Date")
            away_matches = df[df["Away"] == away].sort_values("Date")
            h2h = df[
                ((df["Home"] == home) & (df["Away"] == away)) |
                ((df["Home"] == away) & (df["Away"] == home))
            ].sort_values("Date")

            last_h   = home_matches.iloc[-1] if not home_matches.empty else None
            last_a   = away_matches.iloc[-1] if not away_matches.empty else None
            last_h2h = h2h.iloc[-1]          if not h2h.empty          else None

            features = {
                "HomeTeam": home, "AwayTeam": away,
                "H_roll_gf":          last_h["H_roll_gf"]          if last_h  is not None else 1.5,
                "H_roll_ga":          last_h["H_roll_ga"]          if last_h  is not None else 1.0,
                "H_roll_xg":          last_h["H_roll_xg"]          if last_h  is not None else 1.4,
                "A_roll_gf":          last_a["A_roll_gf"]          if last_a  is not None else 1.2,
                "A_roll_ga":          last_a["A_roll_ga"]          if last_a  is not None else 1.1,
                "A_roll_xg":          last_a["A_roll_xg"]          if last_a  is not None else 1.3,
                "H2H_HomeGoals_Avg":  last_h2h["H2H_HomeGoals_Avg"] if last_h2h is not None else 1.3,
                "H2H_AwayGoals_Avg":  last_h2h["H2H_AwayGoals_Avg"] if last_h2h is not None else 1.1,
                "H2H_Count":          last_h2h["H2H_Count"]         if last_h2h is not None else 5,
                "HomePPDA":           last_h["HomePPDA"]            if last_h  is not None else 9.0,
                "AwayPPDA":           last_a["AwayPPDA"]            if last_a  is not None else 10.5,
            }
            result = predictor.predict(features)

        st.subheader(f"{home} vs {away}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Score",
                  f"{int(round(result['home_goals']))} — {int(round(result['away_goals']))}")
        c2.metric(f"{home} xG", f"{result['home_xg']:.2f}")
        c3.metric(f"{away} xG", f"{result['away_xg']:.2f}")

        st.divider()
        prob_df = pd.DataFrame({
            "Outcome": [f"{home} Win", "Draw", f"{away} Win"],
            "Probability": [
                f"{result['win_prob']  * 100:.1f}%",
                f"{result['draw_prob'] * 100:.1f}%",
                f"{result['loss_prob'] * 100:.1f}%",
            ]
        })
        st.dataframe(prob_df, hide_index=True, use_container_width=True)
