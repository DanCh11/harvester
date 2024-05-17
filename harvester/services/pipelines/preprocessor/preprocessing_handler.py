
import re

import pandas as pd

from concurrent.futures import ThreadPoolExecutor

from deep_translator import GoogleTranslator
from nltk import word_tokenize, WordNetLemmatizer, download
from nltk.corpus import stopwords

from .abstract_handler import AbstractHandler

download(['stopwords', 'punkt', 'wordnet'], quiet=True)


class PreprocessingHandler(AbstractHandler):
    def handle(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset['comment'] = dataset['comment'].apply(self._preprocess)

        return super().handle(dataset)

    def _preprocess(self, text: str) -> str:
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
        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = text.strip()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)

        return text


class TranslationHandler(AbstractHandler):
    def handle(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset['comment'] = self._translate_batch(dataset['comment'].to_list())
        return super().handle(dataset)

    def _translate_batch(self, texts: list[str], max_workers: int = 5) -> list[str]:
        """
            Translates given batch of texts using above translating function. Because
            the dataset can be huge, it uses threads to split the execution process
            to a number of workers.

            @param texts: given texts
            @param max_workers: number of workers

            @return: translated batch of text
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self._translate, texts))
        return results

    def _translate(self, text: str) -> str:
        """
            Helper function that translate a single string into given language
            @param text: given text

            @return: translated text
        """
        if text is None:
            return ""

        return GoogleTranslator(source='auto', target='en').translate(text)


class ProcessingHandler(AbstractHandler):
    def handle(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset['comment'] = dataset['comment'].apply(self._processing_text)

        return super().handle(dataset)

    def _processing_text(self, text: str) -> str:
        tokens = self._tokenize(text)
        lemmatized_tokens = self._lemmatize(tokens)
        filtered_tokens = self._remove_stopwords(lemmatized_tokens)

        return ' '.join(self._lemmatize(filtered_tokens))

    def _tokenize(self, text: str) -> list[str]:
        return word_tokenize(text)

    def _remove_stopwords(self, tokens: list[str]) -> list[str]:
        stop_words = set(stopwords.words("english"))

        return [word for word in tokens if word not in stop_words]

    def _lemmatize(self, tokens: list[str]) -> list[str]:
        return [WordNetLemmatizer().lemmatize(token) for token in tokens]


class RatingConverterHandler(AbstractHandler):
    def handle(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset['rating'] = dataset['rating'].apply(self._convert_rating)

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
