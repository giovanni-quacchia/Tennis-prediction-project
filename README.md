# tennis_prediction

Tennis prediction project for Artificial Intelligence course at University of Venice 2025/2026.

Testing different machine learning models algorithms: Random forest, XGBoost

## Commands

```bash
python3 -m tennis <CMD> # -m runs module as a script
```

### Train a model

```bash
python3 -m tennis train random-forest | xgboost
```

### Predict a match

```bash
python3 -m tennis predict random-forest | xgboost --p1 <player_1> --p2 <player_2>
```

## General info

Using randomized CV, instead of grid search, to speed up hyperparams tuning.

## Future updates

- Data transformation: Ordinal encoding for type of tournament
- Feature engineering: Add player ELO
- CLI: improve commands to be more user-friendly
    - predict p1 p2
    - train --model <model_name>
