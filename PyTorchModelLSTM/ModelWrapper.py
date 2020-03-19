
import numpy as np 
import pandas as pd 
import torch
import torchtext
from torchtext import data
import spacy
import os
import re
import nltk
from nltk.corpus import stopwords
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from spacy.lang.en.examples import sentences 
from .ModelSchema import LSTM_Model

##### 1. Upload the Model
def predict(datazama):
    
    temp = open("movs_temp.csv", "w")
    temp.write(datazama)
    temp.close()

    dataPath = "movs_temp.csv"
    print(datazama)

    # Load columns
    #columns = pd.read_csv("../PyTorchModelLSTM/Datasets/test.csv")
    columns = pd.read_csv("../PyTorchModelLSTM/Datasets/train.csv")
    columns = columns.columns[1:]


    # Load work embeddings english and spanish from spacy
    my_tok = spacy.load('en_core_web_sm')
    my_stopwords = spacy.lang.en.stop_words.STOP_WORDS
    STOPWORDS = set(stopwords.words('spanish'))
    my_stopwords.update(STOPWORDS)



    def spacy_tok(x):
        x= re.sub(r'[^a-zA-Z\s]','',x)
        x= re.sub(r'[\n]','',x)
        return [tok.text for tok in my_tok.tokenizer(x)]


    # Set up Text and Label to read csv files as torchText Tabular Dataset
    TEXT = data.Field(lower=True, tokenize=spacy_tok,eos_token='EOS',stop_words=my_stopwords,include_lengths=True)
    LABEL = data.Field(sequential=False, use_vocab=False, pad_token=None, unk_token=None)


    dataFields = [("Details", TEXT),
                 ("Auto & Transport", LABEL), ("Bills & Utilities", LABEL),
                 ("Entertainment", LABEL), ("Fees & Charges", LABEL),  
                 ("Food & Dining", LABEL), ("Gifts & Donations", LABEL),
                 ("Health & Fitness", LABEL), ("Shopping", LABEL),
                 ("Transfer", LABEL), ("Travel", LABEL), ("Withdrawal", LABEL)]


    train, val = data.TabularDataset.splits(path="../PyTorchModelLSTM/Datasets/",train="train.csv", validation="val.csv", format="csv", fields=dataFields, skip_header=True)


    # Build the vocabulary
    TEXT.build_vocab(train,val, vectors=['fasttext.simple.300d'])


    # Convert tabular dataset to vocabulary vectors
    vectors= train.fields['Details'].vocab.vectors.to("cpu")


    # loadad the model
    model = LSTM_Model(len(columns),len(TEXT.vocab),vectors,1)
    model.load_state_dict(torch.load("../PyTorchModelLSTM/LSTM_transactions_model.pt"))


    # Read csv test file
    dfTest = pd.read_csv(dataPath)
    dataFields = [("Details", TEXT)]
    testDataset = data.TabularDataset(path=dataPath,format='csv',fields=dataFields,skip_header=True)
    test_iter1 = torchtext.data.Iterator(testDataset, batch_size=32, device=torch.device('cpu'), sort=False, sort_within_batch=False, repeat=False,shuffle=False)


    # Process predictions
    myPreds=[]
    with torch.no_grad():
        model.eval()
        for obj in test_iter1:
            text= torch.transpose(obj.Details[0],0,1)[:2]
            pred= model(obj.Details[0],obj.Details[1])        
            pred= torch.sigmoid(pred)
            myPreds.append(pred.detach().numpy())
            del pred;del obj;


    myPreds= np.vstack(myPreds)
    for i, col in enumerate(columns):
        dfTest[col] = myPreds[:, i]

    for i, col in enumerate(columns):
        dfTest[col] = myPreds[:, i]

    dictOfCategories = { i : columns[i] for i in range(0, len(columns) ) }

    best = []
    for i in range(dfTest.shape[0]):
        best.append(dictOfCategories[dfTest.iloc[i,1:].values.argmax()])


    dfTest["Prediction"] = best
    dfTest = dfTest[["Details","Prediction"]]
    dfTest.to_csv("../PyTorchModelLSTM/Datasets/output.csv", index=False)
    pd.read_csv("../PyTorchModelLSTM/Datasets/output.csv").head()
    
    return dfTest.to_csv(index=False)


