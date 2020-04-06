from ...Usecase.ML import ClassificationAlgorithm
from ...PyTorchModelLSTM.ModelWrapper import predict


class PytorchClassificationAlgorithm(ClassificationAlgorithm):

    def classify(self, movements):
        prediction = predict(movements)
        print(prediction)
        return prediction
