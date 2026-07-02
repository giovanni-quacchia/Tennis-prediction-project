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

Start using other features like best of, series, round, tournament. Score is a post-match feature, so we can't use it for prediction.

TODO:

Add order to series

```python
series_mapping = {
    "International": "ATP250",
    "International Gold": "ATP500",
    "Masters": "Masters 1000",
    "Masters Cup": "ATP Finals",
}

df["Series"] = df["Series"].replace(series_mapping)
```

Fix: training was on 70% of 2006-2025, after trainin on entire 2006-2025, CV accuracy is 69.69% and testing accuracy on 2026 is 69.06%

Looking at random forest structure

```python
pipeline = joblib.load("models/random-forest_model.pkl")

preprocessor = pipeline.named_steps["preprocessor"]
forest = pipeline.named_steps["model"]
names = preprocessor.get_feature_names_out()

typer.echo("\nTop feature impacts:")
for name, impact in sorted(
    zip(names, forest.feature_importances_),
    key=lambda x: x[1],
    reverse=True,
)[:10]:
    typer.echo(f"{name}: {impact:.4f}")
```

It comes out that odds is too much important

```bash
Top feature impacts:
num__Odds_Prob_Diff: 0.9999
num__Surface_Elo_Diff: 0.0001
num__Elo_Diff: 0.0000
num__Pts_Diff: 0.0000
num__Rank_Diff: 0.0000
cat__Round_Semifinals: 0.0000
num__Is_Best_Of_5: 0.0000
num__missingindicator_Pts_Diff: 0.0000
num__missingindicator_Odds_Prob_Diff: 0.0000
cat__Court_Indoor: 0.0000
```

Lets try remove odds feature to determine the impact of other features

```bash
Cross-validation accuracy: 67.20%

Top feature impacts:
num__Elo_Diff: 0.8541
num__Surface_Elo_Diff: 0.0668
num__Rank_Diff: 0.0486
num__Pts_Diff: 0.0164
cat__Series_Grand Slam: 0.0059
num__Is_Best_Of_5: 0.0043
cat__Surface_Grass: 0.0008
cat__Series_Masters 1000: 0.0005
cat__Series_ATP250: 0.0004
cat__Round_Semifinals: 0.0003
```

Now elo is the most important feature

I remove isBestOf5 as it has not a significant impact on accuracy.

We can try implementing a combined ELO: $\alpha ELO + (1-\alpha) SurfaceELO $ (70% ELO, 30% SurfaceELO) to see if it improves accuracy.

We create a custom transformer to combine ELO, surfELO and fine-tune alpha parameter to maximize accuracy.

Lets try `neg_log_loss` instead of accuracy, because it is a better metric for probabilistic predictions.

I selected Random Forest as the scikit-learn model because tennis match outcomes depend on nonlinear interactions between player strength, surface, recent form and match context. Random Forest can model these interactions better than Logistic Regression while remaining relatively robust and interpretable through feature importance.

I selected XGBoost as the beyond-scikit-learn model because gradient boosting is well suited for structured tabular data and can capture nonlinear interactions between ranking, Elo, surface performance, recent form and fatigue. It is also widely used in sports analytics and provides strong tools for feature importance and error analysis.


Ranking difference is unfair as 

- Rank 1 vs Rank 10 is more important than Rank 100 vs Rank 110, so we can use a logarithmic transformation of ranking difference to reduce the impact of large differences and emphasize smaller differences.

- training CV accuracy: 69.69%, test accuracy: 69.49%

Try adding log1p for points
- training CV accuracy: 69.67%, test accuracy: 69.56%

Lets now try implementing XGBoost (on macos install brew install libomp)

- training CV accuracy: 69.66%, testing accuracy: 69.77%

Still getting similar accuracy with both random forest and xgboost, so we should improve feature engineering

#### Testing new features

**recent condition on surface on last 10 matches**

| Metrica | Prima | Dopo |
|---|---:|---:|
| CV accuracy | 69.66% | 69.70% |
| Test accuracy | **69.77%** | 69.49% |
| ROC AUC | 0.7574 | 0.7571 |
| Log loss | 0.5858 | 0.5855 |
| Brier | 0.2005 | 0.2004 |

The accuracy has not improved, so we discard it 

**Rest_Days_Diff**: count how many days off have the players before the match

| Metrica | Baseline | Rest days |
|---|---:|---:|
| CV accuracy | **69.66%** | 69.65% |
| Test accuracy | **69.77%** | 69.13% |
| ROC AUC | **0.7574** | 0.7551 |
| Log loss | **0.5858** | 0.5868 |
| Brier | **0.2005** | 0.2011 |

Discard as it has not improved accuracy

We added Tournament, a high-cardinality categorical feature. To limit dimensionality and overfitting, we one-hot encode the k most frequent tournaments and group the remaining and unseen tournaments into a single infrequent category.

We try different k values in hyperparameter tuning

- training CV accuracy: 69.65%, test accuracy: 70.05%

and the importance of odds has decreased significantly

```bash
--- XGBOOST FEATURE IMPORTANCE ---
num__Odds_Logit_Diff: 0.3909
...
```

After this improvements lets retest random forest

- training CV accuracy: 69.70%, test accuracy: 69.06%

Try training xgboost on 2000-2025: 
- training CV accuracy: 69.81%, test accuracy: 69.77%

Adjusted splitting for bigger testing set, now run

# TODO: specify test-size on cli arg

```bash
python3 -m tennis split-data
python3 -m tennis prepare-data
python3 -m tennis train <model>
python3 -m tennis predict <model>
```

Now xgboost: training CV: 70.46%, test: 68.05%
accuracy descreased a little bit, but the test is more reliable as it is bigger (20% of the dataset)