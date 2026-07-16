from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseModelAdapter(ABC):
    @abstractmethod
    def generate(self, images: List[str], prompt: str) -> str:
        ...

    def batch_generate(self, batch: List[Tuple[List[str], str]], batch_size: int = 4) -> List[str]:
        results = []
        for i in range(0, len(batch), batch_size):
            for images, prompt in batch[i:i + batch_size]:
                results.append(self.generate(images, prompt))
        return results
