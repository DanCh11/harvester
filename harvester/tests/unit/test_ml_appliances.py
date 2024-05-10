
import pandas as pd

from harvester.services.pipelines.ml_appliances import create_sentiment_analysis


def test_create_sentiment_analysis(mock_reviews):
    sentiment_df = create_sentiment_analysis(dataset=mock_reviews, column_name='comment', batch_size=1)
    expected_columns = ['sentiment', 'positive', 'negative', 'neutral']

    assert pd.DataFrame == type(sentiment_df)
    assert all(column in sentiment_df.columns for column in expected_columns)
    assert 'negative' == sentiment_df['sentiment'][0]
    assert 0.03846367076039314 == sentiment_df['positive'][0]
    assert 0.7865731716156006 == sentiment_df['negative'][0]
    assert 0.17496313154697418 == sentiment_df['neutral'][0]
