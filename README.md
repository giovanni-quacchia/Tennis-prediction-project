# tennis_prediction

Tennis prediction project for the Artificial Intelligence course at the
University of Venice 2025/2026.

The current pipeline uses ATP data from tennis-data.co.uk (2006 onward),
temporal validation, Elo features and XGBoost.

## Commands

```bash
python3 -m tennis <CMD> # -m runs module as a script
```

### Pipeline

```bash
python3 -m tennis merge-raw-data
python3 -m tennis split-data
python3 -m tennis prepare-data
python3 -m tennis train
python3 -m tennis predict
```

## General info

Using randomized CV, instead of grid search, to speed up hyperparams tuning.

## Future updates

- Data transformation: Ordinal encoding for type of tournament
- Feature engineering: Add player ELO
- Add a command for predicting a future player matchup.


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
python3 -m tennis train
python3 -m tennis predict
```

Now xgboost: training CV: 70.46%, test: 68.00%
accuracy descreased a little bit, but the test is more reliable as it is bigger (20% of the dataset)

Lets try adding data about age, hand, height

https://huggingface.co/datasets/joshsgoldstein/atp-players/blob/main/atp_players.csv

### Analyzing players dataset

```python
# Analyze dataset: head, rows, nulls
@app.command()
def analyze_dataset(
    input_path: Path = typer.Argument(
        ...,
        help="Path to the dataset to analyze",
    )
):
    df = read_dataframe(input_path)

    typer.echo("\n--- HEAD ---")
    typer.echo(df.head().to_string())

    typer.echo("\n--- ROWS ---")
    typer.echo(f"{len(df)}")

    typer.echo("\n--- AVAILABLE DATA % ---")
    available_percentages = df.notna().mean().mul(100).sort_values(
        ascending=False
    )
    typer.echo(
        available_percentages.to_string(float_format="%.2f")
    )
```

```bash
--- AVAILABLE DATA % ---
player_id     100.00
hand           99.98
name_last      99.93
ioc            98.98
name_first     98.60
dob            72.11
wikidata_id     7.42
height          6.30
```

but we should consider players in our matches entries

<!-- TODO: show relevant info -->

Considering age, height or hand doesn't improve accuracy too much, so we currently discard them.

```python

```

```bash
--- ERROR SUMMARY ---
Model: xgboost
Matches: 10259
Errors: 3283
Accuracy: 68.00%
Confident errors (confidence >= 65%): 1175
Errors where bookmaker was correct: 34 (1.04%)
Errors on market upsets: 98.96%

--- ERROR RATE BY YEAR ---
      Matches  Errors Error_Rate
Year
2022     1115     370     33.18%
2023     2607     857     32.87%
2025     2487     805     32.37%
2024     2631     823     31.28%
2026     1419     428     30.16%

--- ERROR RATE BY SURFACE ---
         Matches  Errors Error_Rate
Surface
Clay        3161    1021     32.30%
Hard        5851    1882     32.17%
Grass       1247     380     30.47%

--- ERROR RATE BY SERIES ---
              Matches  Errors Error_Rate
Series
Masters Cup        59      21     35.59%
ATP250           3839    1346     35.06%
Masters 1000     2662     898     33.73%
ATP500           1761     522     29.64%
Grand Slam       1938     496     25.59%

--- ERROR RATE BY ROUND ---
               Matches  Errors Error_Rate
Round
The Final          253      91     35.97%
Semifinals         490     169     34.49%
1st Round         4540    1478     32.56%
2nd Round         2955     955     32.32%
Quarterfinals      946     301     31.82%
4th Round          282      78     27.66%
3rd Round          746     192     25.74%

--- TOP 25 ERRORS ---
      Date                                 Tournament Surface         Round     Player_1             Player_2               Winner Predicted_Winner Player_1_Probability Market_Probability Confidence
2026-05-28                                French Open    Clay     2nd Round    Sinner J.       Cerundolo J.M.       Cerundolo J.M.        Sinner J.                0.961              0.963      0.961
2024-08-30                                    US Open    Hard     2nd Round   Alcaraz C. Van De Zandschulp B. Van De Zandschulp B.       Alcaraz C.                0.959              0.966      0.959
2022-08-29                                    US Open    Hard     1st Round     Fritz T.              Holt B.              Holt B.         Fritz T.                0.957              0.926      0.957
2022-08-30                                    US Open    Hard     1st Round Tsitsipas S.           Galan D.E.           Galan D.E.     Tsitsipas S.                0.957              0.926      0.957
2024-03-12                           BNP Paribas Open    Hard     3rd Round     Nardi L.          Djokovic N.             Nardi L.      Djokovic N.                0.043              0.057      0.957
2023-05-15                Internazionali BNL d'Italia    Clay     3rd Round  Marozsan F.           Alcaraz C.          Marozsan F.       Alcaraz C.                0.044              0.042      0.956
2026-05-26                                French Open    Clay     1st Round  Medvedev D.            Walton A.            Walton A.      Medvedev D.                0.951              0.926      0.951
2025-03-22                                 Miami Open    Hard     2nd Round   Alcaraz C.            Goffin D.            Goffin D.       Alcaraz C.                0.948              0.936      0.948
2024-07-02                                  Wimbledon   Grass     1st Round  Comesana F.            Rublev A.          Comesana F.        Rublev A.                0.053              0.057      0.947
2024-05-12                Internazionali BNL d'Italia    Clay     3rd Round    Tabilo A.          Djokovic N.            Tabilo A.      Djokovic N.                0.054              0.106      0.946
2023-05-30                                French Open    Clay     1st Round  Medvedev D.      Seyboth Wild T.      Seyboth Wild T.      Medvedev D.                0.942              0.913      0.942
2023-01-18                            Australian Open    Hard     2nd Round     Nadal R.          Mcdonald M.          Mcdonald M.         Nadal R.                0.942              0.894      0.942
2024-01-20                            Australian Open    Hard     3rd Round    Borges N.          Dimitrov G.            Borges N.      Dimitrov G.                0.058              0.096      0.942
2025-06-30                                  Wimbledon   Grass     1st Round     Bonzi B.          Medvedev D.             Bonzi B.      Medvedev D.                0.062              0.112      0.938
2025-05-29                                French Open    Clay     2nd Round     Rocha H.            Mensik J.             Rocha H.        Mensik J.                0.063              0.087      0.937
2026-02-24                           Abierto Mexicano    Hard     1st Round    Kypson P.         De Minaur A.            Kypson P.     De Minaur A.                0.063              0.096      0.937
2025-06-02                                French Open    Clay     4th Round    Bublik A.            Draper J.            Bublik A.        Draper J.                0.065              0.106      0.935
2024-02-28                           Abierto Mexicano    Hard     1st Round    Zverev A.          Altmaier D.          Altmaier D.        Zverev A.                0.933              0.904      0.933
2025-01-03                     Brisbane International    Hard Quarterfinals  Djokovic N.            Opelka R.            Opelka R.      Djokovic N.                0.933              0.913      0.933
2023-10-19            Japan Open Tennis Championships    Hard     2nd Round     Fritz T.         Mochizuki S.         Mochizuki S.         Fritz T.                0.932              0.894      0.932
2026-02-19                     Qatar Exxon Mobil Open    Hard Quarterfinals    Sinner J.            Mensik J.            Mensik J.        Sinner J.                0.931              0.894      0.931
2025-10-28                        BNP Paribas Masters    Hard     2nd Round   Alcaraz C.            Norrie C.            Norrie C.       Alcaraz C.                0.929              0.913      0.929
2026-01-30                            Australian Open    Hard    Semifinals    Sinner J.          Djokovic N.          Djokovic N.        Sinner J.                0.927              0.904      0.927
2026-03-30                       Grand Prix Hassan II    Clay     1st Round     Halys Q.           Bennani K.           Bennani K.         Halys Q.                0.927              0.913      0.927
2024-08-16 Western & Southern Financial Group Masters    Hard     2nd Round   Alcaraz C.           Monfils G.           Monfils G.       Alcaraz C.                0.927              0.894      0.927
```

The main problem is that the model follows the market.

The market baseline on bet365 on test set is 66.43%, we try improving the model accuracy without considering market features.

Lets test another datasets from data-co.uk

1. Check common columns
    - Wpts, Lpts only from 2006

Note: W1, L1, ... cannot be considered on current match and would cause data leakage, but we can use them for historical features, like recent form

### Recent fatigue feature

`Fatigue_Diff` estimates which player has accumulated more match workload from
at most 5 matches in the 14 days before a match. Set scores from the current
match are never used to predict that match: they update the history only for
later matches.

For each completed match `m`, its workload is approximated from the games
played in every available set:

```text
MatchLoad_m = sum(i * (W_i + L_i)), for i = 1, ..., 5
```

`W_i + L_i` is the number of games played in set `i`. The multiplier `i` gives
later sets more weight because they occur after more accumulated effort. This
set weighting is a project-specific heuristic, not a formula taken directly
from a scientific paper.

Previous workloads are then discounted according to how many days ago they
occurred:

```text
Fatigue_p(t) = sum(exp(-days_since_m / 7) * MatchLoad_m)
               for up to 5 matches m played by p in the previous 14 days
```

Finally, the model receives a directional difference:

```text
Fatigue_Diff = Fatigue_Player_1 - Fatigue_Player_2
```

A positive value means Player 1 has carried more recent workload. The
exponential decay is inspired by exponentially weighted workload models, which
give recent activity more influence than older activity. The 14-day window,
7-day decay constant, and set weighting are modeling choices that should be
validated rather than treated as physiological facts. Reference: Murray et al.,
[Calculating acute:chronic workload ratios using exponentially weighted moving
averages](https://pubmed.ncbi.nlm.nih.gov/28003238/) (2017).

With Bet365 odds excluded, adding this feature changed the temporal test
results from 64.72% to 64.83% accuracy, from 0.7097 to 0.7113 ROC AUC, and from
0.6205 to 0.6188 log loss. Its XGBoost importance was 0.0111 (rank 44), so it
provided a small complementary signal rather than dominating the prediction.

(also wsets, lsets would cause data leakage)

### Elo progression

`Elo_Progression_Diff` measures the difference between the players' recent Elo
momentum. For each player, it compares the pre-match Elo with the oldest
pre-match Elo available across the previous 5 matches:

```text
EloProgression_p = EloCurrent_p - EloOldest_p
Elo_Progression_Diff = EloProgression_Player_1 - EloProgression_Player_2
```

A positive value means Player 1 has gained more Elo recently. Pre-match Elo
snapshots are stored before applying the current result, preventing data
leakage. `FeatureConfig.history_window` sets the shared 5-match limit for Elo
progression, recent form, first-set form, and fatigue history. With Bet365 odds
excluded, this configuration obtains 65.11% test accuracy, 0.7114 ROC AUC, and
0.6193 log loss. A 10-match experiment obtained 65.04%, 0.7113, and 0.6194
respectively, so the shorter window performed slightly better on this test
period.


#### TOML file

Configured a toml file and installed with `pip install -e .` from root directory to run 

```bash
tennis <cmd>
```


Note: a tennis prediction problem has an high variance based on weather, injuries, fatigue, but this variance from the real world and not from the model, so we can't use bootstrapping with random forests to reduce variance.