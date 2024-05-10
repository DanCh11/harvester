import re

import nltk
import pandas as pd

from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download(['stopwords', 'punkt', 'wordnet'])

REQUIRED_COLUMNS = ['posting_time', 'rating', 'comment']


class ReviewsPreprocessor:
    """
        This class represents a preprocessing pipeline for crawled reviews.
        It requires a strict column identification with concrete type.
        The data is passed through several techniques to have as output
        clean textual data.

        Returns:
            pd.Dataframe: clean textual data
    """
    def __init__(self, dataset: pd.DataFrame, language: str) -> None:
        self.dataset = dataset
        self.language = language

    def validate_data(self):
        """
            Validates required columns in the dataset
        """
        for column in self.dataset.columns:
            assert column in REQUIRED_COLUMNS

    def preprocess(self, text: str) -> list:
        """Executes the following preprocessing steps:
            1. lowering the letters
            2. removes whitespaces
            3. removes digits
            4. removes special characters
            5. tokenizes preprocessed words

        Args:
            text (str): given text

        Returns:
            list: tokenized preprocessed words
        """
        text = text.lower()
        text = text.strip()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)

        return word_tokenize(text)
        
    def remove_stopwords(self, tokens: list) -> list:
        stop_words = set(stopwords.words(self.language))
        
        return [word for word in tokens if word not in stop_words]
    
    def lemmatize(self, tokens: list) -> list:
        lemmatizer = WordNetLemmatizer()

        return [lemmatizer.lemmatize(token) for token in tokens]

    def convert_rating_into_numbers(self, rating: str) -> float | None:
        """
            Because of some sources having different rating representation,
            it's necessary to normalize them into workable type

            Args:
                rating (str): given rating

            Returns:
                float | None: converted rating
        """
        if pd.isna(rating) or not isinstance(rating, str):
            return None
        else:
            rating_match = re.search(r'\b\d+\b', rating)
            if rating_match:
                return int(rating_match.group())
            else:
                return None
            
    def clean_text(self, text: str) -> str:
        tokens = self.preprocess(text)
        filtered_tokens = self.remove_stopwords(tokens)
        lemmatized_tokens = self.lemmatize(filtered_tokens)

        return ' '.join(lemmatized_tokens)

    def execute(self):
        self.dataset = self.dataset[pd.notnull(self.dataset['comment'])]
        self.dataset['comment'] = self.dataset['comment'].apply(self.clean_text)
        self.dataset['rating'] = self.dataset['rating'].apply(self.convert_rating_into_numbers)
        
    
    @staticmethod
    def show_available_languages():
        print(stopwords.fileids())