
from abc import ABC, abstractmethod

import pandas as pd


class MLStrategy(ABC):
    """
        The Strategy interface declares operations common to all supported versions
        of machine learning algorithm.
    """
    @abstractmethod
    def execute(self, *args, **kwargs) -> pd.DataFrame:
        pass
