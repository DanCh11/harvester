
import pandas as pd

from .preprocessing_handler import (PreprocessingHandler, TranslationHandler,
                                    ProcessingHandler, RatingConverterHandler)


class ReviewsPreprocessor:
    """
        This class represents a preprocessing pipeline for crawled reviews.
        It requires a strict column identification with concrete type.
        The data is passed through several techniques to have as output
        clean textual data.

        Returns:
            pd.Dataframe: clean textual data
    """

    def __init__(self, dataset: pd.DataFrame) -> None:
        self.dataset = dataset
        self.pipeline = TranslationHandler()
        (self.pipeline
             .set_next(PreprocessingHandler())
             .set_next(ProcessingHandler())
             .set_next(RatingConverterHandler()))

        self.required_column = ['posting_time', 'rating', 'comment']
        self.__validate_data()

    def __validate_data(self) -> None:
        """
            Validates required columns in the dataset
        """
        for column in self.dataset.columns:
            assert column in self.required_column, f"Column {column} not present in the dataset"

    def execute(self) -> None:
        self.dataset = self.dataset.dropna()
        self.dataset = self.pipeline.handle(self.dataset)
