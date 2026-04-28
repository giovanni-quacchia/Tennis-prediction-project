import typer
import joblib
import pandas as pd
from tennis.models.knn import TennisKNN
from tennis.data_loader import prepare_data as prepare_data_func
from tennis.config import DataConfig, TrainingConfig, FeatureConfig
from sklearn.model_selection import train_test_split

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

    config = DataConfig()
    training_config = TrainingConfig()
    model = TennisKNN(training_config.knn)
    features = FeatureConfig()

    df = pd.read_excel(config.prepared_data_path)

    # Split the data into training and testing sets
    train_set, test_set = train_test_split(df, 
                            test_size=training_config.test_size, 
                            random_state=training_config.random_state)

    # Train the model and tune hyperparameters
    typer.echo("Training KNN model...")
    params, accuracy = model.train(train_set)

    typer.echo(f"Best hyperparameters: {params}")
    typer.echo(f"Training accuracy: {accuracy:.2%}")

    # Test the model on the testing set
    X_test, y_test = test_set[features.trainable], test_set["y"]
    test_accuracy = model.fitted_pipeline.score(X_test, y_test)
    typer.echo(f"Testing accuracy: {test_accuracy:.2%}")

    typer.echo("Training completed. Saving the model...")
    # Save the model for future tests
    joblib.dump(model.fitted_pipeline, config.knn_model_path)

@app.command()
def predict_knn():

    config = DataConfig()
    features = FeatureConfig()

    pipeline = joblib.load(config.knn_model_path)
    
    # Testing set
    raw_df = pd.read_excel(config.testing_data_path)
    testing_set = prepare_data_func(raw_df)
    X,y = testing_set[features.trainable], testing_set["y"]

    predictions = pipeline.predict(X)

    accuracy = (predictions == y).mean()
    print("Test Accuracy:", f"{accuracy:.2%}")