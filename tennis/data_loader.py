import pandas as pd, numpy as np
from tennis.config import FeatureConfig

# https://www.pecan.ai/blog/data-preparation-for-machine-learning/

def load_temporal_train_test_split(path: str, test_size):
    df = pd.read_excel(path)
    
    # sort data to train on past matches and test on future matches
    df["Date"] = pd.to_datetime(df["Date"]) #  convert to datetime (excel may store dates as strings)
    df = df.sort_values("Date")

    # where split df
    # e.g. df with 10 matches, test_size=0.3 --> split_index = 10 * 0.7 = 7
    split_index = int(len(df) * (1 - test_size))

    # iloc (integer location) select rows
    train_set = df.iloc[:split_index]
    test_set = df.iloc[split_index:]

    return train_set, test_set

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