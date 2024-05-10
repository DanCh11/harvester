
import pandas as pd

from germansentiment import SentimentModel


def create_sentiment_analysis(dataset: pd.DataFrame, column_name: str, batch_size: int) -> pd.DataFrame:
    """Creates sentiment analysis to given dataset and column name. germansentiment module uses
       a lot of processing power, which can lead to execution failure if the hard drive is too weak.

    Args:
        dataset (pd.DataFrame): given dataset
        column_name (str): textual column
        batch_size (int): batch size

    Returns:
        pd.DataFrame: preprocesed dataset with sentiment analysis data
    """
    num_rows = len(dataset)
    num_batches = (num_rows + batch_size - 1) // batch_size

    sentiment_data = []

    sentiment_model = SentimentModel()

    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, num_rows)

        batch_texts = dataset.iloc[start_index:end_index][column_name].tolist()

        sentiments, probabilities = sentiment_model.predict_sentiment(batch_texts, output_probabilities=True)

        sentiment_data.extend([
            {'sentiment': sentiment, **{label: probability for label, probability in probs}}
            for sentiment, probs in zip(sentiments, probabilities)
        ])

    sentiment_df = pd.DataFrame(sentiment_data)

    return pd.concat([dataset.reset_index(drop=True), sentiment_df.reset_index(drop=True)], axis=1)
