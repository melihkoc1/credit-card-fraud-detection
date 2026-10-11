import pandas as pd

from src.data import clean_data, split_features_target, time_split


def test_clean_data_drops_exact_duplicates_and_hour():
    df = pd.DataFrame(
        {
            "Time": [0.0, 0.0, 1.0, 2.0],
            "Amount": [5.0, 5.0, 5.0, 7.0],
            "Class": [0, 0, 0, 1],
            "Hour": [0, 0, 0, 0],
        }
    )
    out = clean_data(df)
    assert len(out) == 3                      # only the exact copy is removed
    assert "Hour" not in out.columns
    assert list(out.index) == [0, 1, 2]       # index reset
    assert out["Class"].sum() == 1


def test_clean_data_keeps_rows_that_differ_in_one_column():
    df = pd.DataFrame({"Time": [0.0, 0.0], "Amount": [5.0, 5.01], "Class": [0, 0]})
    assert len(clean_data(df)) == 2


def test_time_split_orders_by_time_and_keeps_ratio(synthetic_df):
    train, test = time_split(synthetic_df, test_frac=0.2)
    assert len(train) == 1200 and len(test) == 300
    assert test["Time"].min() > train["Time"].max()
    assert len(train) + len(test) == len(synthetic_df)


def test_split_features_target_drops_time_and_class(synthetic_df):
    X, y = split_features_target(synthetic_df)
    assert "Time" not in X.columns and "Class" not in X.columns
    assert y.name == "Class"
