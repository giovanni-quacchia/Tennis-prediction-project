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


### Review

```bash
python3 -m tennis train random-forest
Training random-forest model...
Fitting 5 folds for each of 50 candidates, totalling 250 fits
Best hyperparameters: {'model__bootstrap': True, 'model__criterion': 'entropy', 'model__max_depth': 5, 'model__max_features': 'log2', 'model__max_leaf_nodes': 10, 'model__min_samples_leaf': 3, 'model__min_samples_split': 17, 'model__n_estimators': 494}
Cross-validation accuracy: 67.14%
Testing accuracy: 67.25%
Model saved to models/random-forest_model.pkl
```

Random forest train accuracy with random CV: 66.75%
- randomCV is faster than grid search
- Simple feature engineering with ranking, points and bets differences to reduce number of features and improve accuracy

What improve: better feature engineering with ELO

https://www.tennisabstract.com/blog/2019/12/03/an-introduction-to-tennis-elo/

TODO: dynamic k factor. higher for more important matches, lower for less ones.

Training accuracy with ELO: 67.25% (+0.5% improvement)

Trying a new dataset with more matches for a better ELO calculation.

Adapt kaggle datasets, it uses -1 instead of null values

With randomized CV and dataset with data between 2006-2025, train was much slower, but final training accuracy was 68.23% (+1% improvement)

- from 2006, because previous matches didn't have points and betting odds.

Then, prediction on 2026 is 69.49%

In this paper is said best reachable accuracy is around 72-73% with ELO and betting odds, so we put this as a target for our model.

https://journals.sagepub.com/doi/10.1177/17543371231212235?__cf_chl_f_tk=3U.dlm_VRPP7qzLazEGnkreQSYd5lJdSgTyhkZz5DTY-1782938538-1.0.1.1-8hb23WdZJ4oS2gWE1W8VaGjPx9rLg8ozYQ9lAoRkeQM

```bash
python3 -m tennis train random-forest
```

elo_state JSON

- version 1: only ELO 