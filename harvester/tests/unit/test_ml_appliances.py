
import pandas as pd

from harvester.services.pipelines.ml_algorithms.ml_appliances import (CreateSentimentAnalysisStrategy,
                                                                      ExtractDominantTopicsStrategy)


def test_create_sentiment_analysis(mock_reviews):
    sentiment_df = CreateSentimentAnalysisStrategy().execute(dataset=mock_reviews, text_column='comment')
    expected_columns = ['neg', 'neu', 'pos', 'compound']

    assert pd.DataFrame == type(sentiment_df)
    assert all(column in sentiment_df.columns for column in expected_columns)
    assert 0.0 == sentiment_df['neg'][0]
    assert 1.0 == sentiment_df['neu'][0]
    assert 0.0 == sentiment_df['pos'][0]
    assert 0.0 == sentiment_df['compound'][0]


def test_extract_dominant_topics(mock_reviews):
    dominant_topics = ExtractDominantTopicsStrategy().execute(dataset=mock_reviews, text_column='comment',
                                                              num_topics=10)

    expected_columns = ['Topic', 'Count', 'Words']
    assert pd.DataFrame == type(dominant_topics)
    assert dominant_topics['Topic'][0] == 1
    assert dominant_topics['Count'][0] == 1
    assert dominant_topics['Words'][0] == ['Ich', 'das', 'Jahrzehnten', 'von', 'Toilettenpapier', 'seit', 'Aldi.',
                                           'kaufe', 'anderen', 'umsteigen,']
