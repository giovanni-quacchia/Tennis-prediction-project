import typer
from tennis.config import TrainingConfig
from tennis.models.knn import TennisKNN
from tennis.data_loader import prepare_data as prepare_data_func
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from tennis.config import FeatureConfig


app = typer.Typer()

@app.command()
def hello(name: str):
    typer.echo(f"Hello {name}!")

@app.command()
def prepare_data():
    prepare_data_func()
    typer.echo("Data prepared successfully.")

@app.command()
def train_knn():

    training_config, config = TrainingConfig(), FeatureConfig()
    
    model = TennisKNN(training_config.knn)

    df = prepare_data_func()

    X, y = df.drop(columns=["y", "P1", "P2"]), df["y"]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, config.numeric),
        ("cat", cat_pipeline, config.categorical)
    ], verbose_feature_names_out=False)

    typer.echo("Starting Grid Search Tuning...")
    best_params, best_score = model.tune_hyperparameters(X, y, preprocessor)

    print("Best Hyperparameters:", best_params)
    print("Best CV Accuracy:", f"{best_score:.2%}")