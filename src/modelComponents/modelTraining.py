import torch
import torchtext
import torch.nn as nn
from .modelDefinition import MyModel
from .hyperparameters import oneEpoch
    
def train(epochs, TEXT, vectors, training, validation):
    trainLossArr=[]
    valLossArr=[]
    rocTrainLoss=[]
    rocValLoss=[]
    model= MyModel(11,len(TEXT.vocab),vectors,1)
    loss_func= torch.nn.BCEWithLogitsLoss()
    model = model.to("cpu")
    for i in range(epochs):
        tLoss,vLoss,tRocLoss,vRocLoss= oneEpoch(1e-4, training, validation, model, loss_func)
        print(f"Epoch - {i}")
        print(f"Train Loss - {tLoss} vs Val Loss is {vLoss}")
        print(f"Train ROC - {tRocLoss} vs Val ROC is {vRocLoss}")
        trainLossArr.append(tLoss)
        valLossArr.append(vLoss)
        rocTrainLoss.append(tRocLoss)
        rocValLoss.append(vRocLoss)
    return model