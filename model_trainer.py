import numpy as np 
import pandas as pd 
import spacy
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
from torchtext import data
from sklearn.model_selection import train_test_split
from spacy.lang.en.examples import sentences
from ModelSchema import LSTM_Model
from sklearn.metrics import roc_auc_score

# using spaCy tokenizers and stop_words
my_tok = spacy.load('en_core_web_sm')
my_stopwords = spacy.lang.en.stop_words.STOP_WORDS


# tokenizer function
def spacy_tok(x):
    x= re.sub(r'[^a-zA-Z\s]','',x)
    x= re.sub(r'[\n]','',x)
    return [tok.text for tok in my_tok.tokenizer(x)]


# In[4]:
TEXT = data.Field(lower=True, tokenize=spacy_tok,eos_token='EOS',stop_words=my_stopwords,include_lengths=True)
LABEL = data.Field(sequential=False,use_vocab=False,pad_token=None,unk_token=None)


# defining out classes
dataFields = [("Details", TEXT),
             ("Auto & Transport", LABEL), ("Bills & Utilities", LABEL),
             ("Entertainment", LABEL), ("Fees & Charges", LABEL),  
             ("Food & Dining", LABEL), ("Gifts & Donations", LABEL),
             ("Health & Fitness", LABEL), ("Shopping", LABEL),
             ("Transfer", LABEL), ("Travel", LABEL), ("Withdrawal", LABEL)]


# Loading dataset and saving training (train.csv) and validation (val.csv) datasets
df = pd.read_csv("Datasets/trainingFinal.csv")
train, val = train_test_split(df, test_size=0.2)
train.to_csv("Datasets/train.csv", index_label=False, index=False)
val.to_csv("Datasets/val.csv", index_label=False, index=False)
train,val = data.TabularDataset.splits(path="Datasets/", train="train.csv", validation="val.csv", format="csv",
                                       fields=dataFields, skip_header=True)


# In[7]:
traindl, valdl = torchtext.data.BucketIterator.splits(datasets=(train, val),
                                                      batch_sizes=(64, 64),
                                                      sort_key=lambda x: len(x.Details),
                                                      device=torch.device('cpu'),
                                                      sort_within_batch=True
                                                     )


# In[8]:
TEXT.build_vocab(train,val, vectors=['fasttext.simple.300d'])


# In[9]:
vectors = train.fields['Details'].vocab.vectors.to("cpu")


# In[10]:
class BatchGenerator:
    def __init__(self, dl):
        self.dl = dl
        self.yFields= df.columns[1:]
        self.x= 'Details'    
    def __len__(self):
        return len(self.dl)
    def __iter__(self):
        for batch in self.dl:
            X = getattr(batch, self.x)
            y = torch.transpose( torch.stack([getattr(batch, y) for y in self.yFields]),0,1)
            yield (X,y)


# In[13]:
def getValidationLoss(valdl,model,loss_func):
    model.eval()
    runningLoss=0
    valid_batch_it = BatchGenerator(valdl)
    allPreds= []
    allActualPreds= []
    with torch.no_grad():
        for i,obj in enumerate(valid_batch_it):
            obj= ( (obj[0][0].to("cpu"),obj[0][1].to("cpu")),obj[1] )
            preds = model(obj[0][0],obj[0][1])
            loss = loss_func(preds,obj[1].float())
            runningLoss+= loss.item()
            allPreds.append(preds.detach().numpy())
            allActualPreds.append(obj[1].detach().numpy())
        rocLoss= roc_auc_score(np.vstack(allActualPreds),np.vstack(allPreds), average="micro")
        return runningLoss/len(valid_batch_it),rocLoss


# In[14]:

import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
def oneEpoch(lr):
    train_batch_it = BatchGenerator(traindl)
    opt = optim.Adam(model.parameters(),lr)
    runningLoss= 0
    allPreds=[]
    allActualPreds=[]
    for i,obj in enumerate(train_batch_it):
        obj= ( (obj[0][0].to("cpu"),obj[0][1].to("cpu")),obj[1] )
        model.train()
        opt.zero_grad()
        preds = model(obj[0][0],obj[0][1])
        loss = loss_func(preds,obj[1].float())
        runningLoss+= loss.item()
        loss.backward()
        opt.step()
        allPreds.append(preds.detach().numpy())
        allActualPreds.append(obj[1].detach().numpy())
        del obj;del preds
    trainRocLoss= roc_auc_score(np.vstack(allActualPreds),np.vstack(allPreds), average="micro")
    runningLoss= runningLoss/len(train_batch_it)
    valLoss,valRocLoss= getValidationLoss(valdl,model,loss_func)
    torch.cuda.empty_cache()
    return runningLoss,valLoss,trainRocLoss,valRocLoss


# In[15]:

epochs= 10
trainLossArr = []
valLossArr = []
rocTrainLoss = []
rocValLoss = []
model = LSTM_Model(11,len(TEXT.vocab),vectors,1)
loss_func = torch.nn.BCEWithLogitsLoss()
model = model.to("cpu")
for i in range(epochs):
    get_ipython().magic(u'time tLoss,vLoss,tRocLoss,vRocLoss= oneEpoch(1e-4)')
    print(f"Epoch - {i}")
    print(f"Train Loss - {tLoss} vs Val Loss is {vLoss}")
    print(f"Train ROC - {tRocLoss} vs Val ROC is {vRocLoss}")
    trainLossArr.append(tLoss)
    valLossArr.append(vLoss)
    rocTrainLoss.append(tRocLoss)
    rocValLoss.append(vRocLoss)


# In[16]:

import matplotlib.pyplot as plt 
plt.plot(trainLossArr,color='b')
plt.plot(valLossArr,color='g')
plt.plot(rocTrainLoss,color='r')
plt.plot(rocValLoss,color='c')
plt.show()


# In[19]:

import datetime 
dt = datetime.datetime.now()
torch.save(model.state_dict(), "LSTM_transactions_model.pt")


# In[20]:

dfTest = pd.read_csv("Datasets/test.csv")


# In[22]:

dataFields = [("Details", TEXT)]
testDataset= data.TabularDataset(path='Datasets/test.csv',
                                 format='csv',
                                 fields=dataFields,
                                 skip_header=True)


# In[23]:

test_iter1 = torchtext.data.Iterator(testDataset, batch_size=32, device=torch.device('cpu'), sort=False, sort_within_batch=False, repeat=False,shuffle=False)


# In[24]:

myPreds=[]
with torch.no_grad():
    model.eval()
    for obj in test_iter1:
        text= torch.transpose(obj.Details[0],0,1)[:2]
        pred= model(obj.Details[0],obj.Details[1])        
        pred= torch.sigmoid(pred)
        myPreds.append(pred.detach().numpy())
        del pred;del obj;


# In[25]:

myPreds= np.vstack(myPreds)
for i, col in enumerate(df.columns[1:]):
    dfTest[col] = myPreds[:, i]


# In[26]:

for i, col in enumerate(df.columns[1:]):
    dfTest[col] = myPreds[:, i]


# In[27]:

dictOfCategories = { i : dfTest.columns[1:][i] for i in range(0, len(dfTest.columns[1:]) ) }


# In[28]:

best = []
for i in range(dfTest.shape[0]):
    best.append(dictOfCategories[dfTest.iloc[i,1:].values.argmax()])


# In[29]:

dfTest["Prediction"] = best


# In[30]:

dfTest = dfTest[["Details","Prediction"]]


# In[32]:

dfTest.to_csv("testPredictions.csv", index=False)

