

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

def build_tree_preprocessor(features):
    
    # numerical features: impute missing values with median and add indicator for missing values
    num_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
                keep_empty_features=True # set 0 for missing values in the indicator column
            ),
        ),
    ])

    # categorical features: impute missing values with most frequent and one-hot encode
    cat_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ])

    return ColumnTransformer([
        ("num", num_pipeline, features.numeric),
        ("cat", cat_pipeline, features.categorical),
    ])