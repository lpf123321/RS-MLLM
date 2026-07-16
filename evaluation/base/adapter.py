from abc import ABC, abstractmethod
from typing import List


class BaseModelAdapter(ABC):
    @abstractmethod
    def generate(self, images: List[str], prompt: str) -> str:
        ...
