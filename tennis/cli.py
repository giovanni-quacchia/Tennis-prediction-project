import typer
from tennis.config import TrainingConfig
from tennis.models.knn import TennisKNN


app = typer.Typer()

@app.command()
def hello(name: str):
    typer.echo(f"Hello {name}!")

@app.command()
def train_knn():

    config = TrainingConfig()
    
    model = TennisKNN(config.knn)

    # model.train(X_train, y_train)

    typer.echo(f"Model initialized: {model}")