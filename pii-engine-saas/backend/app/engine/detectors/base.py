from abc import ABC, abstractmethod
from typing import List
from app.engine.models import Detection


class BaseDetector(ABC):
    @abstractmethod
    def detect(self, text: str) -> List[Detection]:
        raise NotImplementedError
