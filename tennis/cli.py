import typer
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from tennis.data_loader import prepare_data as prepare_data_func
from tennis.config import DataConfig, TrainingConfig, FeatureConfig
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    
import numpy as np
from tennis.models.random_forest import TennisRandomForest

from tennis.enums import ModelName, PredictMode

app = typer.Typer()

@app.command()
def test():
    config = DataConfig()
    training_config = TrainingConfig()
    features = FeatureConfig()

    df = pd.read_excel(config.prepared_data_path)

    train_set, test_set = train_test_split(df, 
                            test_size=training_config.test_size, 
                            random_state=training_config.random_state)

    # Naive baseline
    baseline_accuracy = test_set["y"].value_counts().max() / test_set["y"].value_counts().sum()
    print("Majority class accuracy", f"{baseline_accuracy:.2%}")

    X_test, y_test = test_set[features.trainable], test_set["y"]

    # Lista dei modelli da analizzare
    models = {
        "Random Forest": config.random_forest_model_path
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
def train(model: ModelName = typer.Argument(..., help="Model to train")):

    config = DataConfig()
    features = FeatureConfig()
    training_config = TrainingConfig()
    
    if model == ModelName.RANDOM_FOREST:
        typer.echo("Training Random Forest model...")
        model_to_train = TennisRandomForest(training_config.random_forest)
    else:
        typer.echo(f"Model {model.value} not implemented yet.")
        raise typer.Exit()
    
    # TODO: move this logic outside

    df = pd.read_excel(config.prepared_data_path)

    # Split the data into training and testing sets
    train_set, test_set = train_test_split(df, 
                            test_size=training_config.test_size, 
                            random_state=training_config.random_state)

    # Train the model and tune hyperparameters
    params, accuracy = model_to_train.train(train_set)

    typer.echo(f"Best hyperparameters: {params}")
    typer.echo(f"Training accuracy: {accuracy:.2%}")

    # Test the model on the testing set
    X_test, y_test = test_set[features.trainable], test_set["y"]
    test_accuracy = model_to_train.fitted_pipeline.score(X_test, y_test)
    typer.echo(f"Testing accuracy: {test_accuracy:.2%}")

    typer.echo("Training completed. Saving the model...")
    # Save the model for future tests
    joblib.dump(model_to_train.fitted_pipeline, config.models_path + f"{model.value}_model.pkl")

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