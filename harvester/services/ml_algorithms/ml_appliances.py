
import nltk
import pandas as pd

from collections import Counter

from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from nltk import pos_tag, sent_tokenize, word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

from .abstract_strategy import MLStrategy

nltk.download('averaged_perceptron_tagger', quiet=True)


class CreateSentimentAnalysisStrategy(MLStrategy):
    def execute(self, dataset: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
            Creates sentiment analysis to given dataset and column name.
            @param dataset: given dataset
            @param text_column: textual column

            @return: pre-processed dataset with sentiment analysis data
        """
        sentiments = dataset[text_column].apply(
            lambda text: SentimentIntensityAnalyzer().polarity_scores(text)
            if isinstance(text, str)
            else {'neg': None, 'neu': None, 'pos': None, 'compound': None})

        sentiment_df = pd.json_normalize(sentiments)

        return pd.concat([dataset, sentiment_df], axis=1)


class ExtractDominantTopicsStrategy(MLStrategy):
    def __init__(self):
        self.lda_model: LdaModel
        self.corpus: list
        self.id2word: Dictionary

    def execute(self,
                dataset: pd.DataFrame,
                text_column: str,
                num_topics: int,
                minimum_probability: float = .8,
                most_common_elements: int = 10) -> pd.DataFrame:
        """
            Extracts dominant topics from given dataset and column name.
            @param dataset: given dataset
            @param text_column: textual column
            @param num_topics: the number of topics
            @param minimum_probability: sets a threshold for the dominant topics
            @param most_common_elements: most common elements

            @return: dataframe with dominant topics
        """
        words = dataset[text_column].apply(lambda doc: doc.split()).tolist()
        self._create_dictionary_and_corpus(words)
        self._train_lda_model(num_topics=num_topics)
        topics = self._extract_dominant_topics(minimum_probability=minimum_probability)

        return self._summarize_topics(topics, num_topics, most_common_elements)

    def _create_dictionary_and_corpus(self, words: list[list[str]]) -> None:
        self.id2word = Dictionary(words)
        self.corpus = [self.id2word.doc2bow(word) for word in words]

    def _train_lda_model(self, num_topics: int, random_state: int = 42) -> None:
        self.lda_model = LdaModel(corpus=self.corpus, id2word=self.id2word,
                                  num_topics=num_topics, random_state=random_state)

    def _extract_dominant_topics(self, minimum_probability: float) -> list:
        """
            Designed to extract the most dominant topic for each document in the corpus.
            For each document (doc_bow), this function retrieves a list of topics
            with their associated probabilities. The result is a list of tuples,
            where each tuple contains a topic ID and its probability.
            @param minimum_probability: sets a threshold for the dominant topics
            @return:
        """
        probability_index: int = 1
        topic_id_index: int = 0

        return [
            max(self.lda_model.get_document_topics(
                doc_bow, minimum_probability=minimum_probability),
                key=lambda topic: topic[probability_index], default=(None,))[topic_id_index]

            for doc_bow in self.corpus
        ]

    def _summarize_topics(self, topics: list[int], num_words: int, most_common_elements: int) -> pd.DataFrame:
        """
        Summarizes the most common topics in the given list of topics.
        @param topics: A list of topics identified for each document.
        @param num_words: The number of words to display for each topic.
        @param most_common_elements: The number of most common topics to summarize.

        @return: A DataFrame summarizing the top topics, their counts, and their most significant words.
        """
        topics_count = Counter(topic for topic in topics if topic is not None)
        top_n_topics = topics_count.most_common(most_common_elements)

        result = [
            (topic, count, [word for word, _ in self.lda_model.show_topic(topic, topn=num_words)])
            for topic, count in top_n_topics
        ]

        return pd.DataFrame(result, columns=['Topic', 'Count', 'Words'])


class PerformAspectAnalysisStrategy(MLStrategy):
    def execute(self,
                dataset: pd.DataFrame,
                text_column: str,
                aspects_sentiments_column: str = 'aspects_sentiments') -> pd.DataFrame:

        dataset[aspects_sentiments_column] = dataset[text_column].apply(self._extract_aspects_and_sentiments)
        return dataset

    def _extract_aspects_and_sentiments(self, text: str) -> list:
        word = word_tokenize(text)
        pos_tags = pos_tag(word)
        aspects = [word for word, pos in pos_tags if pos.startswith('NN')]

        return [
            (aspect,
             'POSITIVE' if (sentiment := TextBlob(next(
                 sent for sent in sent_tokenize(text) if aspect in sent)).sentiment.polarity) > 0
             else 'NEGATIVE' if sentiment < 0 else 'NEUTRAL')
            for aspect in aspects
        ]
