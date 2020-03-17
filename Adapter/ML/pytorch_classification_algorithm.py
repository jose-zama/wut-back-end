from ...Usecase.ML import ClassificationAlgorithm
from spacy.lang.en.examples import sentences
from ...PyTorchModelLSTM.PyTorchModel import predict
#from PyTorchModelLSTM.PyTorchModel import predict
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
        #prediction = predict("../PyTorchModelLSTM/Datasets/test.csv")
        prediction = predict(movements)
        print(prediction)
        return prediction
