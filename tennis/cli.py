import typer
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from tennis.config import DataConfig, TrainingConfig, FeatureConfig    

from tennis.enums import ModelName, PredictMode

from tennis.models.factory import build_model

from tennis.data_loader import (
    EloState,
    create_temporal_train_test_split,
    load_elo_state,
    prepare_data as prepare_data_func,
    save_elo_state,
    read_dataframe
)

from tennis.training import save_model

app = typer.Typer()

@app.command()
def test():
    from sklearn.metrics import (
        accuracy_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )

    config = DataConfig()
    features = FeatureConfig()

    raw_df = read_dataframe(config.raw_testing_data_path)
    test_dates = pd.to_datetime(raw_df["Date"], errors="coerce")

    if test_dates.isna().any():
        raise ValueError("Testing data contains invalid dates")

    elo_state = load_elo_state(config.elo_state_path)

    if elo_state.last_processed_date is None:
        raise ValueError(
            "The Elo state does not contain a last processed date"
        )

    history_end = pd.Timestamp(elo_state.last_processed_date)

    if test_dates.min() <= history_end:
        raise ValueError(
            "Testing data must start after the Elo history"
        )

    testing_set = prepare_data_func(
        raw_df,
        elo_state=elo_state,
    )
    X_test = testing_set[features.trainable]
    y_test = testing_set["y"]

    model_names = (
        ModelName.RANDOM_FOREST,
        ModelName.XGBOOST,
    )
    pipelines = {}
    for model_name in model_names:
        try:
            pipelines[model_name] = joblib.load(
                config.get_model_path(model_name.value)
            )
        except (AttributeError, FileNotFoundError) as error:
            typer.echo(
                f"Skipping {model_name.value}: {error}"
            )

    if not pipelines:
        raise RuntimeError("No compatible trained models found")
    model_probabilities = {
        model_name: pipeline.predict_proba(X_test)[:, 1]
        for model_name, pipeline in pipelines.items()
    }
    global_elo_probability = 1 / (
        1 + 10 ** (-testing_set["Elo_Diff"] / 400)
    )
    surface_elo_probability = 1 / (
        1 + 10 ** (-testing_set["Surface_Elo_Diff"] / 400)
    )

    odds_1 = pd.to_numeric(raw_df["Odd_1"], errors="coerce")
    odds_2 = pd.to_numeric(raw_df["Odd_2"], errors="coerce")
    valid_odds = odds_1.gt(0) & odds_2.gt(0)
    odds_1_inverse = 1 / odds_1[valid_odds]
    odds_2_inverse = 1 / odds_2[valid_odds]
    bookmaker_probability = odds_1_inverse / (
        odds_1_inverse + odds_2_inverse
    )
    bookmaker_target = raw_df.loc[valid_odds, "Winner"].eq(
        raw_df.loc[valid_odds, "Player_1"]
    ).astype(int)

    def print_metrics(
        name: str,
        target: pd.Series,
        probability: pd.Series | np.ndarray,
    ) -> None:
        prediction = (np.asarray(probability) >= 0.5).astype(int)
        typer.echo(f"\n{name}:")
        typer.echo(
            f"  Accuracy: {accuracy_score(target, prediction):.2%}"
        )
        typer.echo(
            f"  ROC AUC: {roc_auc_score(target, probability):.4f}"
        )
        typer.echo(
            f"  Log loss: {log_loss(target, probability):.4f}"
        )
        typer.echo(
            "  Brier score: "
            f"{brier_score_loss(target, probability):.4f}"
        )

    typer.echo("\n--- TEST DATASET ---")
    typer.echo(f"Matches: {len(testing_set)}")
    typer.echo(
        f"Period: {test_dates.min().date()} to "
        f"{test_dates.max().date()}"
    )
    typer.echo("\n--- MODEL COMPARISON ---")
    print_metrics(
        "Bookmaker",
        bookmaker_target,
        bookmaker_probability,
    )
    print_metrics("Global Elo", y_test, global_elo_probability)
    print_metrics(
        "Surface Elo",
        y_test,
        surface_elo_probability,
    )
    for model_name in pipelines:
        print_metrics(
            model_name.value,
            y_test,
            model_probabilities[model_name],
        )

    for model_name, pipeline in pipelines.items():
        preprocessor = pipeline.named_steps["preprocessor"]
        estimator = pipeline.named_steps["model"]
        feature_names = preprocessor.get_feature_names_out()

        typer.echo(
            f"\n--- {model_name.value.upper()} "
            "FEATURE IMPORTANCE ---"
        )
        for feature_name, impact in sorted(
            zip(feature_names, estimator.feature_importances_),
            key=lambda item: item[1],
            reverse=True,
        )[:10]:
            typer.echo(f"{feature_name}: {impact:.4f}")

@app.command()
def split_data():
    config = DataConfig()
    training_config = TrainingConfig()
    
    training_set, testing_set = create_temporal_train_test_split(
        src_path=config.raw_data_path,
        training_path=config.raw_training_data_path,
        testing_path=config.raw_testing_data_path,
        min_date=config.min_date,
        max_date=config.max_date,
        test_size=training_config.test_size,
    )
    
    typer.echo(f"Training matches: {len(training_set)}")
    typer.echo(f"Testing matches: {len(testing_set)}")
    typer.echo(
        f"Split date: {testing_set['Date'].min().date()}"
    )

@app.command()
def prepare_data():
    config = DataConfig()

    raw_df = read_dataframe(
        config.raw_training_data_path
    )

    elo_state = EloState()

    df = prepare_data_func(
        raw_df,
        elo_state=elo_state,
    )

    print(df.head())

    prepared_path = config.prepared_training_data_path
    Path(prepared_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    df.to_csv(
        prepared_path,
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

    training_set = read_dataframe(
        config.prepared_training_data_path
    )
    training_set["Date"] = pd.to_datetime(
        training_set["Date"],
        errors="raise",
    )
    training_set = training_set.sort_values(
        "Date",
        kind="stable",
    )

    model_to_train = build_model(model, training_config)

    best_params, best_cv_score = model_to_train.train(
        training_set
    )
    
    scoring = training_config.search.scoring

    if scoring == "neg_log_loss":
        metric_name = "log loss"
        metric_value = -best_cv_score
        formatted_value = f"{metric_value:.4f}"
    elif scoring == "accuracy":
        metric_name = "accuracy"
        metric_value = best_cv_score
        formatted_value = f"{metric_value:.2%}"
    else:
        metric_name = scoring
        metric_value = best_cv_score
        formatted_value = f"{metric_value:.4f}"

    typer.echo(f"Best hyperparameters: {best_params}")
    typer.echo(
        f"Cross-validation {metric_name}: {formatted_value}"
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
        
        raw_df = read_dataframe(
            config.raw_testing_data_path
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
