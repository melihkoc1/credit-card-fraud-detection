"""Data loading, cleaning and the time-based train/test split."""
import pandas as pd

RANDOM_STATE = 42
TARGET = "Class"
# Time and Hour never enter the model (see README: only ~48 hours of data).
DROP_COLUMNS = [TARGET, "Time"]


def load_data(path):
    """Read the Kaggle creditcard.csv from ``path``."""
    return pd.read_csv(path)


def clean_data(df):
    """Drop the helper column (Hour) and remove fully duplicated rows."""
    return (
        df.drop(columns=["Hour"], errors="ignore")
        .drop_duplicates()
        .reset_index(drop=True)
    )


def time_split(df, test_frac=0.2):
    """Sort by Time; the last ``test_frac`` of the rows form the test set."""
    df = df.sort_values("Time").reset_index(drop=True)
    cut = int(len(df) * (1 - test_frac))
    return df.iloc[:cut], df.iloc[cut:]


def split_features_target(df):
    """Return (X, y): X without Class and Time, y = Class."""
    return df.drop(columns=DROP_COLUMNS), df[TARGET]
