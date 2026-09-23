# Tennis Prediction Project

In this project, we test different machine learning algorithms to predict the outcome of tennis matches and analyze the performance and results obtained.

**Disclaimer**: Betting odds contain the most predictive information, and adding match- and player-specific features did not significantly improve accuracy. This report presents the engineered features, model results, and notable predictions.

Please refer to **`report/main.pdf`** for the detailed report.

### Project Structure

The project is organized as follows:

- `notebooks/`: Jupyter notebooks containing the codebase of the project

- `data/`: Directory with matches and players data, both raw and ready for training and testing

- `models/`: Models trained for the analysis phase

### 1. Datasets and Data Cleaning

[Open the data-cleaning notebook](notebooks/01_Clean_data.ipynb)

Two main datasets were used in this project

- **Tennis matches**: A merge of the datasets provided by [tennis-data.co.uk](https://www.tennis-data.co.uk) with matches from 2005 to 2026 with approximately 55,000 matches

- **Tennis Players**: dataset with players' data, including date of birth (dob) and dominant hand.

Both datasets were cleaned by removing duplicates, rows with missing values and player names were uniformed to simplify joining the datasets and compute the age of each player at the time of the match.

### 2. Feature Engineering

[Open the feature-engineering notebook](notebooks/02_Feature_engineering.ipynb)

In this phase, the most important features were computed to train the models and achieve the best possible performance. I will introduce a few of them, but if you want further details, just take a look at the referenced notebook and the report.

#### ELO

It estimates a player's skill level, both overall and surface-specific. Players start with a rating of 1500, with win probability $E_1$ for Player 1 defined as

$$ E_1 = (1 + 10^{(R_2 - R_1)/400})^{-1} $$

The rating is updated post-match as $\Delta R = K \cdot (S_1 - E_1)$, where $R_i$ are pre-match ratings and $S_1 \in \{0, 1\}$ is the outcome. To account for rating maturity, I employ a dynamic update factor $K$ decreasing with career matches $M_i(t)$.  

$$ K_i(t) = 250 \cdot (M_i(t) + 5)^{-0.4} $$

The surface ELO is computed using the same formula, for each type of surface.
To address sparse data on specific courts, it is combined in a 50/50 blend with the overall rating.

#### Fatigue

It is modeled using an Exponentially Weighted Moving Average (EWMA). It recursively updates the fatigue metric to capture recent workload spikes while giving progressively less weight to older sessions:

$$
    \text{EWMA}_{\text{today}} = \text{Load}_{\text{today}} \cdot \lambda + (1 - \lambda) \cdot \text{EWMA}_{\text{yesterday}}, \quad \lambda = \frac{2}{N+1}
$$

where $\lambda$ is the degree of decay, $N$ is the fatigue window size in days and $\text{LOAD} \in \{0, 1\}$.

#### Head to Head statistics

They compare two players' historical performance against each other, both overall and surface-specific. See the example below

| Match ($P_1$ vs $P_2$) | $y$ | Pre-match H2H | Post-match H2H |
| :--- | :---: | :---: | :---: |
| nadal r. vs djokovic n. | 1 | 0 | -1 |
| djokovic n. vs nadal r. | 0 | -1 | -2 |
| nadal r. vs djokovic n. | 1 | 2 | -3 |

*Figure: Sample head-to-head calculation. `pair_key = (djokovic, nadal)` is sorted alphabetically, and $y=1$ when $P_1$ wins.*

This phase produce a prepared dataset, which is split into training and testings datasets (70-30).

### 3. Models Training

[Open the models-training notebook](notebooks/03_Models_training.ipynb)

For this project, I decided to use Random Forest and XGBoost. Tree-based algorithms naturally excel in sports prediction by capturing non-linear feature interactions (e.g., fatigue and rankings). Comparing them evaluates Random Forest's robust bagging approach against XGBoost's sequential boosting, which typically achieves higher predictive accuracy.

Each algorithm is trained with and without betting odds to assess the predictive power of the market and using randomized cross-validation for hyperparameter fine-tuning, employing a time-series split to avoid data leakage. 

The average cross-validation accuracy obtained on the training set is about 68\% and 70\% for the model trained without and with betting odds, respectively.

I also used preprocessing pipelines to handle both numerical and categorical features:

- **Numerical features**: Missing values are imputed using the median and an indicator column is added to flag missing values.

- **Categorical features**: Missing values are imputed using the most frequent value and one-hot encoding is applied, grouping rare or too frequent categories into "other".

### 4. Results Evaluation

[Open the analysis notebook](notebooks/04_Analysis.ipynb)

#### Naive Baseline

The frequency of the most frequent class is about 50\%, as we randomly swapped the players in each match. Other interesting baselines regard ranking and points features, which reach an accuracy of about 63.5\%.
The market odds baseline is 68.42\%, which is the most important benchmark to beat.

The results obtained on the test set are summarized in the following table:

| Model | Accuracy | F1-Score | ROC AUC | Log loss | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| RF no-odds | 64.96% | 0.6444 | 0.7136 | 0.6172 | 0.2149 |
| XGB no-odds | 65.32% | 0.6505 | 0.7168 | 0.6159 | 0.2143 |
| RF with-odds | 67.24% | 0.6692 | 0.7410 | 0.5963 | 0.2057 |
| XGB with-odds | **67.92%** | **0.6785** | **0.7459** | **0.5921** | **0.2039** |

I noticed that, as expected, the models with odds assign an importance of about 34\% to the odds feature, making it one of the most important features used. 
Other important features are the surface-specific Elo, ranking and points differences, which accurately capture the players' current form.

#### Prediction Anomalies

The analysis performed on the test set revealed that the most confident correct predictions regard matches with a top player winner and a lower-ranked player loser, with a high Elo and ranking differences.
Conversely, the most confident incorrect predictions share the same statistical profile --high Elo, ranking gap-- but the lower-ranked player wins against the favorite, often due to the favorite's fatigue or a recent loss of form.
For instance, Cerundolo defeated Sinner in May 2026 due to a heat stroke, and Vesely upset Djokovic in 2022 during the Serbian's first tournament back after missing several events due to his refusal to get the Covid vaccine.

This further analysis is essential to understand why the market odds are not always accurate, but they are still stronger than any model based solely on past match statistics, which cannot account for unpredictable physical or psychological variables.

#### Feature Distribution

 The distribution of the most important features, such as surface Elo, odds and ranking differences, show that the model is more likely to make correct predictions when the differences are larger, while it struggles with matches where the players are more skill-balanced.
However, when the model does a wrong prediction on a large-gap match, it tends to do so with high confidence.

The model achieves a higher accuracy in Grand Slam tournaments, where matches are longer and the stronger players have more time to recover from a bad set and making the final outcome much more predictable based on Elo and rankings.
