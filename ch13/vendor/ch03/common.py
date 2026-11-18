"""Shared loading, splitting and preprocessing for chapter 3."""
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_california_housing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
TARGET = "MedHouseVal"


def load():
    """Return the housing data as one DataFrame."""
    return fetch_california_housing(as_frame=True).frame


def split(df):
    """Split first: 80% to learn from, 20% held back."""
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )


def preprocessor(columns):
    """Fill gaps with the median, then standardize."""
    numeric = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler()
    )
    return ColumnTransformer([("num", numeric, list(columns))])
