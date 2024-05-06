import re

import spacy
import pandas as pd

from contractions import fix
from spellchecker import SpellChecker
from spacy.lang.de.stop_words import STOP_WORDS

REQUIRED_COLUMNS = ['posting_time', 'rating', 'comment']


class Preprocessor:
    """
        This class represents a preprocessing pipeline for crawled reviews.
        It requires a strict column identification with concrete type.
        The data is passed through several techniques to have as output
        clean textual data. Crawled reviews are normally in german, so
        the preprocessor works only with german language.

        Returns:
            pd.Dataframe: clean textual data
    """
    def __init__(self, dataset: pd.DataFrame) -> None:
        self.spell_checker = SpellChecker(language='de')
        self.nlp = spacy.load('de_core_news_sm')
        self.dataset = dataset

    def validate_data(self):
        """
            Validates required columns in the dataset
        """
        for column in self.dataset.columns:
            assert column in REQUIRED_COLUMNS

    def lowercase(self, text: str) -> str:
        doc = self.nlp(text)
        
        return " ".join(token.text.lower() for token in doc)
    
    def remove_whitespaces(self, text: str):
        return text.strip()

    def tokenize_and_lemmatize(self, text: str):
        doc = self.nlp(text)
        return [token.lemma_ for token in doc if token.lemma_ != '-PRON-']
    
    def remove_depricated_elemets(self, words: list) -> list:
        """
            In depricated elements enter punctiation, stopwords and special characters.
            
            Args:
                words (list): the list of tokenized words

            Returns:
                list: the list of words without depricated elements.
        """
        return [token for token in words if token.isalnum() and token.lower() not in STOP_WORDS]

    def spell_check(self, words: list) -> list:
        """
            Replaces incorrect spelled words with dictionary correct form. 

            Args:
                words (list): the list of tokenized words

            Returns:
                list: a list of correct spelled words
        """
        return [self.spell_checker.correction(word) for word in words]

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
            
    def expand_contractions(self, text: str) -> str:
        """
            Resolves contractions (and slang), such as: I'm -> I am, etc.

            Args:
                text (str): given text

            Returns:
                str: contracted text
        """
        return fix(text)

    def execute(self):
        self.dataset['comment'] = self.dataset['comment'].apply(self.lowercase)
        self.dataset['comment'] = self.dataset['comment'].apply(self.expand_contractions)
        self.dataset['comment'] = self.dataset['comment'].apply(self.remove_whitespaces)
        self.dataset['comment'] = self.dataset['comment'].apply(self.tokenize_and_lemmatize)
        self.dataset['comment'] = self.dataset['comment'].apply(self.remove_depricated_elemets)
        self.dataset['comment'] = self.dataset['comment'].apply(self.spell_check)
        
        self.dataset['rating'] = self.dataset['rating'].apply(self.convert_rating_into_numbers )
