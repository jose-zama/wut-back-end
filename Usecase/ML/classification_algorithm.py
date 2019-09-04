from abc import ABC, abstractmethod


class ClassificationAlgorithm(ABC):
    @abstractmethod
    def classify(self, movements):
        pass
