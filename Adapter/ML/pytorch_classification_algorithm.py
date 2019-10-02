from ...Usecase.ML import ClassificationAlgorithm
from spacy.lang.en.examples import sentences
from nltk.corpus import stopwords
from torchtext import data

import spacy
import nltk
import re
import torchtext
import torch.nn as nn
import torch
import numpy as np
import pandas as pd


class PytorchClassificationAlgorithm(ClassificationAlgorithm):

    def classify(self, movements):
        return {
            "Movements": [
                {"Category": "Entertainment", "Movement": "Cheves en la azotea"}
            ]
        }
