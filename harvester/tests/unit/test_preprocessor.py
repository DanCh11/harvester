import numpy as np
import pandas as pd

from harvester.services.preprocessor import (ReviewsPreprocessor,
                                             PreprocessingHandler,
                                             TranslationHandler,
                                             ProcessingHandler,
                                             RatingConverterHandler)


def test_preprocessor(mock_reviews):
    expected_preprocessed_comment = 'perforation partially missing thin strip suddenly tear longer length torn'

    preprocessor = ReviewsPreprocessor(mock_reviews)
    preprocessor.execute()
    preprocessed_comment = preprocessor.dataset['comment'][2]
    preprocessed_rating = preprocessor.dataset['rating'][2]

    assert str == type(preprocessed_comment)
    assert preprocessed_comment == expected_preprocessed_comment

    assert np.int64 == type(preprocessed_rating)
    assert 1.0 == preprocessed_rating


def test_preprocessing_handler(mock_reviews):
    expected_preprocessed_comment = "ich kaufe seit jahrzehnten das toilettenpapier von aldi"
    preprocessor = PreprocessingHandler().handle(dataset=mock_reviews)

    assert pd.DataFrame == type(preprocessor)
    assert expected_preprocessed_comment == preprocessor['comment'][0]


def test_translation_handler(mock_reviews):
    expected_translated_comment = "I've been buying Aldi toilet paper for decades."
    translator = TranslationHandler().handle(dataset=mock_reviews)

    assert pd.DataFrame == type(translator)
    assert expected_translated_comment == translator['comment'][0]


def test_processing_handler(mock_reviews):
    expected_processed_handler = "Ich kaufe seit Jahrzehnten da Toilettenpapier von Aldi ."
    processor = ProcessingHandler().handle(dataset=mock_reviews)

    assert pd.DataFrame == type(processor)
    assert expected_processed_handler == processor['comment'][0]


def test_rating_converter_handler(mock_reviews):
    converter = RatingConverterHandler().handle(dataset=mock_reviews)

    assert pd.DataFrame == type(converter)
    assert np.float64 == type(converter['rating'][1])
    assert 2.0 == converter['rating'][1]
