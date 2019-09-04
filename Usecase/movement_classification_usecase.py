
class MovementClassificationUseCase(object):

    def __init__(self, algorithm):
        self._algorithm = algorithm
        return

    def apply(self, movements):
        # Call algorithm
        return self._algorithm.classify(movements)
