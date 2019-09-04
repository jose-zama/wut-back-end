
class ApplicationController(object):

    def __init__(self, movement_classification_usecase):
        self._movement_classification_usecase = movement_classification_usecase
        return

    def healthcheck(self):
        return 'OK'

    def check_movements(self, movements):
        return self._movement_classification_usecase.apply(movements)
