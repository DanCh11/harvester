import pandas as pd

from pytest import fixture


@fixture(scope="session")
def mock_reviews() -> pd.DataFrame:
    return pd.read_csv("../resources/aldi_mock_reviews.csv")
