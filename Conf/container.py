from dependency_injector import providers, containers
from ..Adapter import ApplicationController
from ..Usecase import MovementClassificationUseCase
from ..Adapter.ML import PytorchClassificationAlgorithm


class Algorithms(containers.DeclarativeContainer):
    pytorch = providers.Factory(PytorchClassificationAlgorithm)


class UseCases(containers.DeclarativeContainer):
    movement_classification = providers.Factory(MovementClassificationUseCase, algorithm=Algorithms.pytorch)


class Controllers(containers.DeclarativeContainer):
    app = providers.Factory(
        ApplicationController,
        movement_classification_usecase=UseCases.movement_classification
    )