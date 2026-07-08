from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_preprocessor(features) -> ColumnTransformer:
    numeric = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
                keep_empty_features=True,
            ),
        ),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    tournament = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                max_categories=30,
            ),
        ),
    ])

    return ColumnTransformer([
        ("numeric", numeric, features.numeric),
        ("categorical", categorical, features.categorical),
        (
            "tournament",
            tournament,
            features.high_cardinality_categorical,
        ),
        ("elo", "passthrough", features.elo),
    ])
