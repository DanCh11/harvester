
import numpy as np

from harvester.services.pipelines.reviews_preprocessor import ReviewsPreprocessor


def test_preprocessor(mock_reviews):
    PREPROCESSED_COMMENT = 'perforation teilweise vorhanden reißen plötzlich größere länge dünne streifen beim abreißen'

    preprocessor = ReviewsPreprocessor(dataset=mock_reviews, language='german')
    preprocessor.execute()
    preprocessed_comment = preprocessor.dataset['comment'][2]
    preprocessed_rating = preprocessor.dataset['rating'][2]

    assert str == type(preprocessed_comment)
    assert preprocessed_comment == PREPROCESSED_COMMENT

    assert np.float64 == type(preprocessed_rating)
    assert 1.0 == preprocessed_rating
