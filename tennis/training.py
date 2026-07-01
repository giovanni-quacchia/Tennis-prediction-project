from pathlib import Path

import joblib
import typer


def print_training_results(best_params: dict, cv_accuracy: float, test_accuracy: float):
    typer.echo(f"Best hyperparameters: {best_params}")
    typer.echo(f"Cross-validation accuracy: {cv_accuracy:.2%}")
    typer.echo(f"Testing accuracy: {test_accuracy:.2%}")


def evaluate_model(model, test_set, features) -> float:
    X_test = test_set[features.trainable]
    y_test = test_set["y"]

    return model.score(X_test, y_test)


def save_model(model, model_path: str | Path):
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model.fitted_pipeline, model_path)

    typer.echo(f"Model saved to {model_path}")