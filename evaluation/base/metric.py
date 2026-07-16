from abc import ABC, abstractmethod
from typing import Dict, List


class BaseMetric(ABC):
    name: str = "base"

    @abstractmethod
    def compute(self, references: List[List[str]], predictions: List[str]) -> Dict[str, float]:
        ...
