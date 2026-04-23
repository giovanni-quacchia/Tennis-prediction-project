import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from tennis.config import DataConfig, FeatureConfig

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from tennis.config import FeatureConfig

# https://www.pecan.ai/blog/data-preparation-for-machine-learning/

"""
Shape: (2644, 38)

0 duplicates
"""
def prepare_data(raw_df: pd.DataFrame) -> pd.DataFrame:

    features = FeatureConfig()
    df = raw_df[features.raw].copy()

    raw_numeric_cols = ["WRank", "LRank", "WPts", "LPts", "B365W", "B365L", "PSW", "PSL", "MaxW", "MaxL", "AvgW", "AvgL"]
    for col in raw_numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Feature engineering ---
    df["y"] = 1 # P1 wins  

    df["Rank_Diff"] = df["WRank"] - df["LRank"]
    df["Pts_Diff"] = df["WPts"] - df["LPts"]
    
    df["B365_Bet_Diff"] = df["B365W"] - df["B365L"]
    df["PS_Bet_Diff"] = df["PSW"] - df["PSL"]
    df["Max_Bet_Diff"] = df["MaxW"] - df["MaxL"]
    df["Avg_Bet_Diff"] = df["AvgW"] - df["AvgL"]

    random_swap(df)

    # --- Selection ---
    df = df[features.numeric + features.categorical + features.debug + ["y"]]
    
    return df

# --- Random Swap Player 1 and Player 2 ---
# Otherwise, the model will learn to alway pick Player 1 as the winner
def random_swap(df):
    rng = np.random.default_rng(seed=42)
    swap_mask = rng.integers(0, 2, size=len(df)).astype(bool) # Randomly swap 50% of the rows

    # Swap players randomly 50% of the time 
    df.loc[swap_mask, ["Winner", "Loser"]] = df.loc[swap_mask, ["Loser", "Winner"]].values

    diff_cols = ["Rank_Diff", "Pts_Diff", "B365_Bet_Diff", "PS_Bet_Diff", "Max_Bet_Diff", "Avg_Bet_Diff"]
    for col in diff_cols:
        df.loc[swap_mask, col] = df.loc[swap_mask, col] * -1 # Invert difference for swapped rows

    df.loc[swap_mask, "y"] = 0 # P2 wins

def preprocess_data(X_train, X_test, y_train, y_test, scale=True):
    features = FeatureConfig()

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale: num_steps.append(("scaler", StandardScaler()))   
    
    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline(num_steps, features.numeric)),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # One-hot encode categorical variables
            # Column_value: Boolean for each category
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), features.categorical),
    ], 
    remainder="passthrough", # keep P1, P2 wihtout transformation
    verbose_feature_names_out=False # don't change column names after transformation
    ) 
    
    preprocessor.set_output(transform="pandas")

    # Don't learn from test data, only transform it
    X_train_final = preprocessor.fit_transform(X_train)
    X_test_final = preprocessor.transform(X_test)

    return X_train_final, X_test_final, y_train, y_test