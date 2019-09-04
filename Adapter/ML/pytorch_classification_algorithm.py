from ...Usecase.ML.classification_algorithm import ClassificationAlgorithm
import pandas as pd


class PytorchClassificationAlgorithm(ClassificationAlgorithm):

    def classify(self, movements):
        df = pd.DataFrame(movements)
        # Call Pytorch
        return [
            {"classification": "Food & Dining", "movement": "8284120518LOAC720417H51CAFE DAS CORTEZ I"}
        ]