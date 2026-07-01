import typer
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from tennis.data_loader import load_temporal_train_test_split, prepare_data as prepare_data_func
from tennis.config import DataConfig, TrainingConfig, FeatureConfig
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    

from tennis.enums import ModelName, PredictMode

from tennis.models.factory import build_model

from tennis.training import (
    evaluate_model,
    print_training_results,
    save_model,
)

app = typer.Typer()

@app.command()
def test():
    config = DataConfig()
    training_config = TrainingConfig()
    features = FeatureConfig()

    df = pd.read_excel(config.prepared_data_path)
    
    # sort data to train on past matches and test on future matches
    df = df.sort_values(by="Date")
    
    # where split df
    # e.g. df with 10 matches, test_size=0.3 --> split_index = 10 * 0.7 = 7
    split_index = int(len(df) * (1 - training_config.test_size))

    train_set = df.iloc[:split_index]
    test_set = df.iloc[split_index:]

    # Naive baseline
    baseline_accuracy = test_set["y"].value_counts().max() / test_set["y"].value_counts().sum()
    print("Majority class accuracy", f"{baseline_accuracy:.2%}")

    X_test, y_test = test_set[features.trainable], test_set["y"]

    # Lista dei modelli da analizzare
    models = {
        "Random Forest": config.get_model_path(
            ModelName.RANDOM_FOREST.value
        )
    }

    for name, path in models.items():
        try:
            pipeline = joblib.load(path)
            y_pred = pipeline.predict(X_test)
            
            print(f"\n--- ANALYSIS: {name} ---")
            
            # 2. Classification Report (Precision, Recall, F1)
            print("Classification Report:")
            print(classification_report(y_test, y_pred))
            
            # 3. Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            print("Confusion Matrix:")
            print(cm)
            
            # 4. ROC AUC Score (richiede le probabilità)
            if hasattr(pipeline, "predict_proba"):
                y_probs = pipeline.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_probs)
                print(f"ROC AUC Score: {auc:.4f}")
                
        except Exception as e:
            print(f"Could not analyze {name}: {e}")
    

@app.command()
def prepare_data():
    config = DataConfig()

    raw_df = pd.read_excel(config.raw_data_path)
    df = prepare_data_func(raw_df)
    print(df.head())

    df.to_excel(config.prepared_data_path, index=False)
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
        raw_df = pd.read_excel(config.testing_data_path)
        testing_set = prepare_data_func(raw_df)
        X, y = testing_set[features.trainable], testing_set["y"]

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