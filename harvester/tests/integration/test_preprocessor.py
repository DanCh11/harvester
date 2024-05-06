
import numpy as np
import pandas as pd

from harvester.services.pipelines.preprocessor import Preprocessor


PREPROCESSED_COMMENT = ['beworben', 'antike', 'mal', 'verfügbar', 'letzter', 'sonst', 
                        'viermal', 'passieren', 'fahren']


def mock_reviews() -> pd.DataFrame:
    return pd.read_csv("./harvester/tests/resources/mock_data.csv")

def test_preprocessor():
    preprocessor = Preprocessor(dataset=mock_reviews())
    preprocessor.execute()
    preprocessed_comment = preprocessor.dataset['comment'][0]
    preprocessed_rating = preprocessor.dataset['rating'][1]

    assert list == type(preprocessed_comment)
    assert preprocessed_comment == PREPROCESSED_COMMENT

    assert np.float64 == type(preprocessed_rating)
    assert 1.0 == preprocessed_rating
