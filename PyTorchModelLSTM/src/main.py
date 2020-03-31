import pandas as pd
import torch
from dataPreprocessing import (applyOneHotEncoding, splitData, setDataFields, setTabularData, buildVocabulary,
                                setVectors, setDataForBatches)
from modelComponents import (modelTraining, hyperparameters)

def prepareDataAndTrainModel():
    df = pd.read_csv("./Datasets/training-data.csv", header=0, names=["Category", "Details"])
    df = applyOneHotEncoding(df)
    splitData(df)
    vocabulary, dataFields = setDataFields()
    training, validation = setTabularData(dataFields)
    vocabulary = buildVocabulary(vocabulary, training, validation)
    vectors = setVectors(training)
    training, validation = setDataForBatches(training, validation)
    model = modelTraining.train(10, vocabulary, vectors, training, validation)
    torch.save(model.state_dict(), "./Output/LSTM_transactions_model.pt")

prepareDataAndTrainModel()

