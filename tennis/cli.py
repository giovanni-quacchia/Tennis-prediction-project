import typer
import joblib
import pandas as pd
from tennis.models.knn import TennisKNN
from tennis.data_loader import prepare_data as prepare_data_func
from tennis.config import DataConfig, TrainingConfig, FeatureConfig

app = typer.Typer()

@app.command()
def hello(name: str):
    typer.echo(f"Hello {name}!")

@app.command()
def prepare_data():
    config = DataConfig()

    raw_df = pd.read_excel(config.raw_data_path)
    df = prepare_data_func(raw_df)
    print(df.head())

    df.to_excel(config.prepared_data_path, index=False)
    typer.echo("Data prepared successfully.")

@app.command()
def train_knn():

    config, training_config = DataConfig(), TrainingConfig()
    model = TennisKNN(training_config.knn)

    training_set = pd.read_excel(config.prepared_data_path)

    typer.echo("Training KNN model...")
    params, accuracy = model.train(training_set)

    typer.echo(f"Best hyperparameters: {params}")
    typer.echo(f"Cross-validation accuracy: {accuracy:.2%}")

    typer.echo("Training completed. Saving the model...")
    # Save the model for future tests
    joblib.dump(model.fitted_pipeline, config.knn_model_path)

@app.command()
def predict_knn():

    config, training_config = DataConfig(), TrainingConfig()
    features = FeatureConfig()

    pipeline = joblib.load(config.knn_model_path)
    
    # Testing set
    raw_df = pd.read_excel(config.testing_data_path)
    testing_set = prepare_data_func(raw_df)
    X,y = testing_set[features.trainable], testing_set["y"]

    predictions = pipeline.predict(X)

    accuracy = (predictions == y).mean()
    print("Test Accuracy:", f"{accuracy:.2%}")