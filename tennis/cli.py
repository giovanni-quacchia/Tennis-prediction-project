import typer
import joblib
import pandas as pd
from tennis.models.knn import TennisKNN
from tennis.models.decision_tree import TennisDecisionTree
from tennis.data_loader import prepare_data as prepare_data_func
from tennis.config import DataConfig, TrainingConfig, FeatureConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    
import numpy as np

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
        "KNN": config.knn_model_path,
        "Decision Tree": config.decision_tree_model_path
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


@app.command()
def train_decision_tree():

    config = DataConfig()
    training_config = TrainingConfig()
    model = TennisDecisionTree(training_config.decision_tree)
    features = FeatureConfig()

    df = pd.read_excel(config.prepared_data_path)

    # Split the data into training and testing sets
    train_set, test_set = train_test_split(df, 
                            test_size=training_config.test_size, 
                            random_state=training_config.random_state)

    # Train the model and tune hyperparameters
    typer.echo("Training Decision Tree model...")
    params, accuracy = model.train(train_set)

    typer.echo(f"Best hyperparameters: {params}")
    typer.echo(f"Training accuracy: {accuracy:.2%}")

    # Test the model on the testing set
    X_test, y_test = test_set[features.trainable], test_set["y"]
    test_accuracy = model.fitted_pipeline.score(X_test, y_test)
    typer.echo(f"Testing accuracy: {test_accuracy:.2%}")

    typer.echo("Training completed. Saving the model...")
    # Save the model for future tests
    joblib.dump(model.fitted_pipeline, "tennis/models/decision_tree_model.pkl")

@app.command()
def predict_decision_tree():

    config = DataConfig()
    features = FeatureConfig()

    pipeline = joblib.load(config.decision_tree_model_path)
    
    # Testing set
    raw_df = pd.read_excel(config.testing_data_path)
    testing_set = prepare_data_func(raw_df)
    X,y = testing_set[features.trainable], testing_set["y"]

    predictions = pipeline.predict(X)

    accuracy = (predictions == y).mean()
    print("Test Accuracy:", f"{accuracy:.2%}")