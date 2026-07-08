from pathlib import Path

import joblib
import pandas as pd
import typer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)

from tennis_prediction.config import DataConfig, FeatureConfig, TrainingConfig
from tennis_prediction.data_loader import (
    HistoricalState,
    create_temporal_train_test_split,
    merge_raw_datasets,
    prepare_data as prepare_features,
    read_dataframe,
)
from tennis_prediction.xgboost import TennisXGBoost


app = typer.Typer()


@app.command()
def merge_raw_data():
    """Merge annual tennis-data.co.uk files from 2006 onward."""
    config = DataConfig()
    dataset = merge_raw_datasets(
        source=config.raw_dataset_dir,
        destination=config.raw_data_path,
        min_date=config.min_date,
        max_date=config.max_date,
    )
    typer.echo(f"Merged matches: {len(dataset)}")
    typer.echo(f"Saved to: {config.raw_data_path}")


@app.command()
def split_data():
    """Create chronological training and testing datasets."""
    data_config = DataConfig()
    training_config = TrainingConfig()
    training, testing = create_temporal_train_test_split(
        source=data_config.raw_data_path,
        training_path=data_config.raw_training_data_path,
        testing_path=data_config.raw_testing_data_path,
        test_size=training_config.test_size,
    )
    typer.echo(f"Training matches: {len(training)}")
    typer.echo(f"Testing matches: {len(testing)}")
    typer.echo(f"Split date: {testing['Date'].min().date()}")


@app.command()
def prepare_data():
    """Prepare train and test features with continuous historical state."""
    config = DataConfig()
    historical_state = HistoricalState()
    prepared_training = prepare_features(
        read_dataframe(config.raw_training_data_path),
        historical_state=historical_state,
        random_state=config.random_state,
    )
    prepared_testing = prepare_features(
        read_dataframe(config.raw_testing_data_path),
        historical_state=historical_state,
        random_state=config.random_state,
    )

    outputs = (
        (config.prepared_training_data_path, prepared_training),
        (config.prepared_testing_data_path, prepared_testing),
    )
    for output_path, dataset in outputs:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_excel(output_path, index=False)

    typer.echo(f"Prepared training matches: {len(prepared_training)}")
    typer.echo(f"Prepared testing matches: {len(prepared_testing)}")


@app.command()
def train():
    """Tune and train the XGBoost pipeline with temporal cross-validation."""
    data_config = DataConfig()
    training_config = TrainingConfig()
    training_set = read_dataframe(
        data_config.prepared_training_data_path
    )
    training_set["Date"] = pd.to_datetime(
        training_set["Date"], errors="raise"
    )
    training_set = training_set.sort_values("Date", kind="stable")

    model = TennisXGBoost(
        model_config=training_config.xgboost,
        search_config=training_config.search,
    )
    best_params, best_score = model.train(training_set)

    model_path = Path(data_config.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model.fitted_pipeline, model_path)

    typer.echo(f"Best hyperparameters: {best_params}")
    typer.echo(f"Cross-validation log loss: {-best_score:.4f}")
    typer.echo(f"Model saved to: {model_path}")


@app.command()
def predict():
    """Evaluate the trained XGBoost pipeline on the temporal test set."""
    config = DataConfig()
    features = FeatureConfig()
    testing_set = read_dataframe(
        config.prepared_testing_data_path
    )
    pipeline = joblib.load(config.model_path)
    probabilities = pipeline.predict_proba(
        testing_set[features.trainable]
    )[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    target = testing_set["y"]

    typer.echo(f"Test matches: {len(testing_set)}")
    typer.echo(
        f"Accuracy: {accuracy_score(target, predictions):.2%}"
    )
    typer.echo(f"ROC AUC: {roc_auc_score(target, probabilities):.4f}")
    typer.echo(f"Log loss: {log_loss(target, probabilities):.4f}")


@app.command()
def analyze_model():
    """Print detailed classification metrics on the temporal test set."""
    config = DataConfig()
    features = FeatureConfig()
    testing_set = read_dataframe(config.prepared_testing_data_path)
    pipeline = joblib.load(config.model_path)
    probabilities = pipeline.predict_proba(
        testing_set[features.trainable]
    )[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    target = testing_set["y"]

    typer.echo(f"Accuracy: {accuracy_score(target, predictions):.2%}")
    typer.echo(f"ROC AUC: {roc_auc_score(target, probabilities):.4f}")
    typer.echo("\nClassification report:")
    typer.echo(
        classification_report(
            target,
            predictions,
            target_names=["Player 2 wins", "Player 1 wins"],
            zero_division=0,
        )
    )
    typer.echo("Confusion matrix [[TN, FP], [FN, TP]]:")
    typer.echo(str(confusion_matrix(target, predictions)))


@app.command()
def market_baseline():
    """Measure how often the Bet365 favorite wins."""
    dataset = read_dataframe(DataConfig().raw_testing_data_path)
    winner_odds = pd.to_numeric(dataset["B365W"], errors="coerce")
    loser_odds = pd.to_numeric(dataset["B365L"], errors="coerce")
    valid = winner_odds.gt(1) & loser_odds.gt(1)
    accuracy = winner_odds[valid].lt(loser_odds[valid]).mean()
    typer.echo(f"Matches: {valid.sum()}")
    typer.echo(f"Bet365 accuracy: {accuracy:.2%}")

@app.command()
def analyze_dataset(path: Path):
    """Inspect a dataset structure, feature types, and missing values."""
    dataset = read_dataframe(path)

    typer.echo(f"Dataset path: {path}")
    typer.echo(f"Rows: {len(dataset)}")
    typer.echo(f"Columns: {dataset.shape[1]}")
    typer.echo(f"Shape: {dataset.shape}")

    typer.echo("\nColumn types:")
    typer.echo(str(dataset.dtypes))

    # Count missing values per feature and show both absolute and percentage values.
    null_summary = pd.DataFrame({
        "dtype": dataset.dtypes.astype(str),
        "null_count": dataset.isna().sum(),
        "null_pct": dataset.isna().mean().mul(100).round(2),
        "non_null_count": dataset.notna().sum(),
        "unique_count": dataset.nunique(dropna=True),
    }).sort_values(
        by=["null_count", "null_pct"],
        ascending=False,
    )

    typer.echo("\nMissing values by feature:")
    typer.echo(str(null_summary))

    # Highlight object columns because they often contain numeric values read as text.
    object_columns = dataset.select_dtypes(include="object").columns.tolist()

    typer.echo("\nObject columns:")
    if object_columns:
        typer.echo(", ".join(object_columns))
    else:
        typer.echo("No object columns found.")

    # Show a few sample values for object columns to quickly spot dirty values.
    for column in object_columns:
        sample_values = (
            dataset[column]
            .dropna()
            .astype(str)
            .unique()[:10]
        )
        typer.echo(f"\nSample values for {column}:")
        typer.echo(str(list(sample_values)))

    # Print pandas info output, including memory usage and non-null counts.
    typer.echo("\nDataFrame info:")
    dataset.info()