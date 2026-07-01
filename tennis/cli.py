from xml.parsers.expat import model

import typer
import joblib
import numpy as np
import pandas as pd
from tennis.config import DataConfig, TrainingConfig, FeatureConfig
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    

from tennis.enums import ModelName, PredictMode

from tennis.models.factory import build_model

from tennis.data_loader import (
    EloState,
    load_elo_state,
    load_temporal_train_test_split,
    prepare_data as prepare_data_func,
    save_elo_state,
    read_dataframe
)

from tennis.training import (
    evaluate_model,
    print_training_results,
    save_model,
)

app = typer.Typer()

@app.command()
def test():
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        brier_score_loss,
        classification_report,
        confusion_matrix,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    config = DataConfig()
    features = FeatureConfig()

    raw_df = read_dataframe(config.testing_data_path)
    test_dates = pd.to_datetime(raw_df["Date"], errors="coerce")

    if test_dates.isna().any():
        raise ValueError("Testing data contains invalid dates")

    elo_state = load_elo_state(config.elo_state_path)

    if elo_state.last_processed_date is None:
        raise ValueError(
            "The Elo state does not contain a last processed date"
        )

    history_end = pd.Timestamp(elo_state.last_processed_date)
    test_start = test_dates.min()

    if test_start <= history_end:
        raise ValueError(
            "Testing data must start after the Elo history"
        )

    testing_set = prepare_data_func(
        raw_df,
        elo_state=elo_state,
    )

    X_test = testing_set[features.trainable]
    y_test = testing_set["y"]

    model_path = config.get_model_path(
        ModelName.RANDOM_FOREST.value
    )
    pipeline = joblib.load(model_path)

    y_probability = pipeline.predict_proba(X_test)[:, 1]
    y_prediction = (y_probability >= 0.5).astype(int)

    majority_accuracy = y_test.value_counts(normalize=True).max()

    odds_1 = pd.to_numeric(raw_df["Odd_1"], errors="coerce")
    odds_2 = pd.to_numeric(raw_df["Odd_2"], errors="coerce")
    valid_odds = odds_1.gt(0) & odds_2.gt(0)
    player_1_won = raw_df["Winner"].eq(raw_df["Player_1"])
    odds_prediction = odds_1.lt(odds_2)
    odds_accuracy = accuracy_score(
        player_1_won[valid_odds],
        odds_prediction[valid_odds],
    )

    elo_prediction = (
        testing_set["Elo_Diff"] >= 0
    ).astype(int)

    typer.echo("\n--- TEST DATASET ---")
    typer.echo(f"Matches: {len(testing_set)}")
    typer.echo(
        f"Period: {test_dates.min().date()} to "
        f"{test_dates.max().date()}"
    )
    typer.echo(f"Positive class: {y_test.mean():.2%}")

    typer.echo("\n--- BASELINES ---")
    typer.echo(f"Majority accuracy: {majority_accuracy:.2%}")
    typer.echo(
        "Odds accuracy: "
        f"{odds_accuracy:.2%} "
        f"({valid_odds.sum()}/{len(raw_df)} matches)"
    )
    typer.echo(
        "Elo accuracy: "
        f"{accuracy_score(y_test, elo_prediction):.2%}"
    )

    typer.echo("\n--- RANDOM FOREST ---")
    typer.echo(
        f"Accuracy: {accuracy_score(y_test, y_prediction):.2%}"
    )
    typer.echo(
        "Balanced accuracy: "
        f"{balanced_accuracy_score(y_test, y_prediction):.2%}"
    )
    typer.echo(
        f"Precision: {precision_score(y_test, y_prediction):.2%}"
    )
    typer.echo(
        f"Recall: {recall_score(y_test, y_prediction):.2%}"
    )
    typer.echo(f"F1 score: {f1_score(y_test, y_prediction):.2%}")
    typer.echo(f"ROC AUC: {roc_auc_score(y_test, y_probability):.4f}")
    typer.echo(f"Log loss: {log_loss(y_test, y_probability):.4f}")
    typer.echo(
        f"Brier score: {brier_score_loss(y_test, y_probability):.4f}"
    )

    typer.echo("\nClassification report:")
    typer.echo(
        classification_report(
            y_test,
            y_prediction,
            digits=4,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(y_test, y_prediction)
    typer.echo("Confusion matrix [[TN, FP], [FN, TP]]:")
    typer.echo(str(matrix))
    

@app.command()
def prepare_data():
    config = DataConfig()

    raw_df = read_dataframe(config.raw_data_path)
    
    raw_df["Date"] = pd.to_datetime(
        raw_df["Date"],
        errors="coerce",
    )
    
    history_end = pd.Timestamp(
        config.history_end_date
    )
    
    historical_df = raw_df[
        raw_df["Date"] <= history_end
    ].copy()

    elo_state = EloState()

    df = prepare_data_func( 
        historical_df,
        elo_state=elo_state,
    )

    print(df.head())

    df.to_excel(
        config.prepared_data_path,
        index=False,
    )

    save_elo_state(
        elo_state,
        config.elo_state_path,
    )
    
    typer.echo("Data prepared successfully.")


@app.command()
def train(
    model: ModelName = typer.Argument(
        ..., # no default value, required argument
        help="Model to train",
    )
):
    config = DataConfig()
    features = FeatureConfig()
    training_config = TrainingConfig()

    typer.echo(f"Training {model.value} model...")

    train_set, test_set = load_temporal_train_test_split(
        path=config.prepared_data_path,
        min_date=config.min_training_date,
        max_date=config.max_training_date,
        test_size=training_config.test_size,
    )

    model_to_train = build_model(model, training_config)

    best_params, cv_accuracy = model_to_train.train(train_set)

    test_accuracy = evaluate_model(
        model=model_to_train,
        test_set=test_set,
        features=features,
    )

    print_training_results(
        best_params=best_params,
        cv_accuracy=cv_accuracy,
        test_accuracy=test_accuracy,
    )

    save_model(
        model=model_to_train,
        model_path=config.get_model_path(model.value),
    )

@app.command()
def predict(
    model: ModelName = typer.Argument(..., help="Model to use [random-forest|xgboost|ensemble]"),
    mode: PredictMode = typer.Option(PredictMode.dataset, help="Prediction mode"),
    player1: str = typer.Option(None, help="Player 1 name (only for players mode)"),
    player2: str = typer.Option(None, help="Player 2 name (only for players mode)"),
):
    config = DataConfig()
    features = FeatureConfig()

    if mode == PredictMode.dataset:
        
        raw_df = pd.read_csv(
            config.testing_data_path
        )

        elo_state = load_elo_state(
            config.elo_state_path
        )
        
        test_dates = pd.to_datetime(
            raw_df["Date"],
            errors="coerce",
        )

        if test_dates.isna().any():
            raise ValueError(
                "Testing data contains invalid dates"
            )

        if elo_state.last_processed_date is None:
            raise ValueError(
                "The Elo state does not contain "
                "a last processed date"
            )

        test_min_date = test_dates.min()
        history_end = pd.Timestamp(
            elo_state.last_processed_date
        )

        if test_min_date <= history_end:
            raise ValueError(
                "Testing data must start after "
                "the Elo history"
            )

        testing_set = prepare_data_func(
            raw_df,
            elo_state=elo_state,
        )
        
        X = testing_set[features.trainable]
        y = testing_set["y"]
        
        if model == ModelName.ENSEMBLE:
            # Load all models and average their predictions
            pipelines = [
                joblib.load(config.get_model_path(ModelName.RANDOM_FOREST.value)),
                # joblib.load(config.get_model_path(ModelName.XGBOOST.value)),
            ]
            
            # for each model: array of probabilities for the first player winning
            # random-forest: [0.3, 0.6, 0.8]
            # xgboost:       [0.4, 0.5, 0.9]
            # Average them:  [0.35, 0.55, 0.85]
            probs = np.mean([p.predict_proba(X)[:, 1] for p in pipelines], axis=0)
            predictions = (probs > 0.5).astype(int)
        else:
            path = config.get_model_path(model.value)
            pipeline = joblib.load(path)
            predictions = pipeline.predict(X)

        accuracy = (predictions == y).mean()
        print(f"Test Accuracy: {accuracy:.2%}")

    elif mode == PredictMode.players:
        if not player1 or not player2:
            typer.echo("You must specify --player1 and --player2", err=True)
            raise typer.Exit(1)
        

        typer.echo(f"{player1} vs {player2}")
        # X = build_matchup(player1, player2)
        # prediction = pipeline.predict(X)
        typer.echo("Player mode not yet implemented")
