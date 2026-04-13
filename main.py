import argparse
import sys
from rich.console import Console
from rich.table import Table

console = Console()

def run_retrain():
    # 1. Run Data Extraction
    import src.data_pipeline
    import src.feature_engineering
    import src.train_lgbm
    import src.train_xg_mlp
    import src.train_dixon_coles

    console.print("\n[bold cyan]1/5: Extracting and Processing Data...[/]")
    src.data_pipeline.run_pipeline()

    console.print("\n[bold cyan]2/5: Feature Engineering...[/]")
    src.feature_engineering.build_features()

    console.print("\n[bold cyan]3/5: Training LightGBM Models...[/]")
    src.train_lgbm.train_lgbm()

    console.print("\n[bold cyan]4/5: Training PyTorch xG Models...[/]")
    src.train_xg_mlp.train_xg_mlp()

    console.print("\n[bold cyan]5/5: Fitting Dixon-Coles Distribution...[/]")
    src.train_dixon_coles.train_dixon_coles()

    console.print("\n[bold green]Retraining Pipeline Complete! Models updated.[/]\n")

def run_predict(home, away):
    from src.predict import predictor

    with console.status("[bold blue]Loading components...[/]") as status:
        try:
            import pandas as pd
            from pathlib import Path
            df = pd.read_csv("data/processed/features_clean.csv", parse_dates=["Date"])
            
            # Fetch latest home team stats
            home_matches = df[df["Home"] == home].sort_values("Date")
            if not home_matches.empty:
                last_h = home_matches.iloc[-1]
                h_gf, h_ga, h_xg, h_ppda = last_h["H_roll_gf"], last_h["H_roll_ga"], last_h["H_roll_xg"], last_h["HomePPDA"]
            else:
                h_gf, h_ga, h_xg, h_ppda = 1.5, 1.0, 1.4, 9.0

            # Fetch latest away team stats
            away_matches = df[df["Away"] == away].sort_values("Date")
            if not away_matches.empty:
                last_a = away_matches.iloc[-1]
                a_gf, a_ga, a_xg, a_ppda = last_a["A_roll_gf"], last_a["A_roll_ga"], last_a["A_roll_xg"], last_a["AwayPPDA"]
            else:
                a_gf, a_ga, a_xg, a_ppda = 1.2, 1.1, 1.3, 10.5

            # Fetch Head-to-Head stats
            h2h = df[((df["Home"]==home) & (df["Away"]==away)) | ((df["Home"]==away) & (df["Away"]==home))].sort_values("Date")
            if not h2h.empty:
                last_h2h = h2h.iloc[-1]
                h2h_hg, h2h_ag, h2h_c = last_h2h["H2H_HomeGoals_Avg"], last_h2h["H2H_AwayGoals_Avg"], last_h2h["H2H_Count"]
            else:
                h2h_hg, h2h_ag, h2h_c = 1.3, 1.1, 5

            features = {
                "HomeTeam": home,
                "AwayTeam": away,
                "H_roll_gf": h_gf, "H_roll_ga": h_ga, "H_roll_xg": h_xg,
                "A_roll_gf": a_gf, "A_roll_ga": a_ga, "A_roll_xg": a_xg,
                "H2H_HomeGoals_Avg": h2h_hg, "H2H_AwayGoals_Avg": h2h_ag, "H2H_Count": h2h_c,
                "HomePPDA": h_ppda, "AwayPPDA": a_ppda,
            }
            
            result = predictor.predict(features)
        except FileNotFoundError as e:
            console.print(f"[red]Error: {e}[/red]")
            return
        except Exception as e:
            console.print(f"[red]Error during prediction: {e}[/red]")
            return

    # Display Results
    console.print(f"\n[bold]Match Prediction:[/] [cyan]{home}[/] vs [cyan]{away}[/]")
    home_goals_int = int(round(result['home_goals']))
    away_goals_int = int(round(result['away_goals']))
    console.print(f"[bold]Predicted Score:[/] {home_goals_int} - {away_goals_int}")
    console.print(f"[bold]Expected Goals (xG):[/] {result['home_xg']:.2f} - {result['away_xg']:.2f}\n")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Outcome")
    table.add_column("Probability")
    
    table.add_row(f"{home} Win", f"{result['win_prob']*100:.1f}%")
    table.add_row("Draw", f"{result['draw_prob']*100:.1f}%")
    table.add_row(f"{away} Win", f"{result['loss_prob']*100:.1f}%")
    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Football Prediction CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Re-train Command
    parser_retrain = subparsers.add_parser("retrain", help="Run the full pipeline to fetch data and train models")

    # Predict Command
    parser_predict = subparsers.add_parser("predict", help="Predict an upcoming match")
    parser_predict.add_argument("--home", required=True, help="Home team name")
    parser_predict.add_argument("--away", required=True, help="Away team name")

    args = parser.parse_args()

    if args.command == "retrain":
        run_retrain()
    elif args.command == "predict":
        run_predict(args.home, args.away)

if __name__ == "__main__":
    main()
