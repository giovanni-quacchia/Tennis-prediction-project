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
def prepare_data():

    config, features = DataConfig(), FeatureConfig()
    raw_df = pd.read_excel(config.raw_data_path)
    df = raw_df[features.raw].copy()

    # --- Random Swap Player 1 and Player 2 ---
    # Otherwise, the model will learn to alway pick Player 1 as the winner

    rng = np.random.default_rng(seed=42)
    swap_mask = rng.integers(0, 2, size=len(df)).astype(bool) # Randomly swap 50% of the rows

    df["P1"], df["P2"] = df["Winner"], df["Loser"]
    df["P1_Rank"], df["P2_Rank"] = df["WRank"], df["LRank"]
    df["P1_Pts"], df["P2_Pts"] = df["WPts"], df["LPts"]
    df["P1_Bet"], df["P2_Bet"] = df["B365W"], df["B365L"] # TODO: avg odds from multiple bookmakers
    df["y"] = 1 # P1 wins

    # Swap players randomly 50% of the time 
    df.loc[swap_mask, ["P1", "P2"]] = df.loc[swap_mask, ["P2", "P1"]].values
    df.loc[swap_mask, ["P1_Rank", "P2_Rank"]] = df.loc[swap_mask, ["P2_Rank", "P1_Rank"]].values
    df.loc[swap_mask, ["P1_Pts", "P2_Pts"]] = df.loc[swap_mask, ["P2_Pts", "P1_Pts"]].values
    df.loc[swap_mask, ["P1_Bet", "P2_Bet"]] = df.loc[swap_mask, ["P2_Bet", "P1_Bet"]].values
    df.loc[swap_mask, "y"] = 0 # P2 wins

    # --- Feature engineering ---
    df["Rank_Diff"] = df["P1_Rank"] - df["P2_Rank"]
    df["Pts_Diff"] = df["P1_Pts"] - df["P2_Pts"]
    df["Bet_Diff"] = df["P1_Bet"] - df["P2_Bet"]

    # --- Selection ---
    df = df[features.numeric + features.categorical + features.debug + ["y"]]
    
    return df

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