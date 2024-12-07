
import re

import contractions
import pandas as pd

from nltk import word_tokenize, download
from nltk.corpus import stopwords

from .abstract_handler import AbstractHandler

download(['stopwords', 'punkt_tab', 'wordnet'], quiet=True)


class PreprocessingHandler(AbstractHandler):
    def handle(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset.loc[:, 'comment'] = dataset['comment'].apply(self._preprocess)

        return super().handle(dataset)

    def _preprocess(self, text: str) -> str:
        """Executes the following preprocessing steps:
            1. fix contractions such as `you're` to you `are`
            2. lowering the letters
            3. removes whitespaces
            4. removes digits
            5. removes special characters

        Args:
            text (str): given text

        Returns:
            list: tokenized preprocessed words
        """
        if not isinstance(text, str):
            return ""

        text = contractions.fix(text)
        text = text.lower()
        text = text.strip()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)

        return text


class ProcessingHandler(AbstractHandler):
    def handle(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset.loc[:, 'comment'] = dataset['comment'].apply(self._processing_text)

        return super().handle(dataset)

    def _processing_text(self, text: str) -> str:
        tokens = self._tokenize(text)
        filtered_tokens = self._remove_stopwords(tokens)

        return ' '.join(filtered_tokens)

    def _tokenize(self, text: str) -> list[str]:
        return word_tokenize(text)

    def _remove_stopwords(self, tokens: list[str]) -> list[str]:
        stop_words = set(stopwords.words("german"))

        return [word for word in tokens if word not in stop_words]


class RatingConverterHandler(AbstractHandler):
    def handle(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset.loc[:, 'rating'] = dataset['rating'].apply(self._convert_rating)

        return super().handle(dataset)

    def _convert_rating(self, rating: str) -> float | None:
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
