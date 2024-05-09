
import numpy as np
import pandas as pd

from harvester.services.pipelines.preprocessor import Preprocessor


PREPROCESSED_COMMENT = 'perforation teilweise vorhanden reißen plötzlich größere länge dünne streifen beim abreißen'


def mock_reviews() -> pd.DataFrame:
    return pd.read_csv("./harvester/tests/resources/aldi_mock_reviews.csv")

def test_preprocessor():
    preprocessor = Preprocessor(dataset=mock_reviews(), language='german')
    preprocessor.execute()
    preprocessed_comment = preprocessor.dataset['comment'][2]
    preprocessed_rating = preprocessor.dataset['rating'][2]

    assert str == type(preprocessed_comment)
    assert preprocessed_comment == PREPROCESSED_COMMENT

    assert np.float64 == type(preprocessed_rating)
    assert 1.0 == preprocessed_rating
