#!/usr/bin/env python
# coding: utf-8

# In[3]:


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
from sklearn.model_selection import train_test_split


# In[4]:


import spacy
from spacy.lang.en.examples import sentences


# In[5]:


my_tok = spacy.load('en_core_web_sm')
my_stopwords = spacy.lang.en.stop_words.STOP_WORDS


# In[6]:


def spacy_tok(x):
    x= re.sub(r'[^a-zA-Z\s]','',x)
    x= re.sub(r'[\n]','',x)
    #x = re.sub(r'[A-Za-z]+|\d+',x)
    #SPLIT_NUMBERS = re.compile(r'([+-]?\d*\.?\d+(?:[eE][-+]?\d+)?)')
    #x = ' '.join(SPLIT_NUMBERS.split(x))
    return [tok.text for tok in my_tok.tokenizer(x)]


# In[7]:


TEXT = data.Field(lower=True, tokenize=spacy_tok,eos_token='EOS',stop_words=my_stopwords,include_lengths=True)
LABEL = data.Field(sequential=False, 
                         use_vocab=False, 
                         pad_token=None, 
                            unk_token=None)


# In[8]:


df = pd.read_csv("Datasets/trainingFinal.csv")


# In[9]:


train, val = train_test_split(df, test_size=0.2)


# In[10]:


train.to_csv("Datasets/train.csv", index_label=False, index=False)
val.to_csv("Datasets/val.csv", index_label=False, index=False)


# In[11]:


dataFields = [("Details", TEXT),
             ("Auto & Transport", LABEL), ("Bills & Utilities", LABEL),
             ("Entertainment", LABEL), ("Fees & Charges", LABEL),  
             ("Food & Dining", LABEL), ("Gifts & Donations", LABEL),
             ("Health & Fitness", LABEL), ("Shopping", LABEL),
             ("Transfer", LABEL), ("Travel", LABEL), ("Withdrawal", LABEL)]


# In[12]:


train,val = data.TabularDataset.splits(path="Datasets/",train="train.csv", validation="val.csv", format="csv", fields=dataFields, skip_header=True)


# In[13]:


#vec = torchtext.vocab.Vectors('cc.es.300.vec', cache='./Datasets/')


# In[65]:


#TEXT.build_vocab(train,val, vectors=['fasttext.simple.300d',vec])


# In[14]:


TEXT.build_vocab(train,val, vectors=['fasttext.simple.300d'])


# In[15]:


len(TEXT.vocab)


# In[16]:


traindl, valdl = torchtext.data.BucketIterator.splits(datasets=(train, val),
                                            batch_sizes=(64, 64),
                                            sort_key=lambda x: len(x.Details),
                                            device=torch.device('cpu'),
                                            sort_within_batch=True
                                                     )


# In[17]:


vectors= train.fields['Details'].vocab.vectors.to("cpu")


# In[18]:


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


# In[19]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


# In[20]:


class MyModel(nn.Module):
    def __init__(self,op_size,n_tokens,pretrained_vectors,nl=2,bidirectional=True,emb_sz=300,n_hiddenUnits=100):
        super(MyModel, self).__init__()
        self.n_hidden= n_hiddenUnits
        self.embeddings= nn.Embedding(n_tokens,emb_sz)
        self.embeddings.weight.data.copy_(pretrained_vectors)
#         self.embeddings.weight.requires_grad = False
        self.rnn= nn.LSTM(emb_sz,n_hiddenUnits,num_layers=2,bidirectional=True,dropout=0.2)
        self.lArr=[]
        if bidirectional:
            n_hiddenUnits= 2* n_hiddenUnits
        self.bn1 = nn.BatchNorm1d(num_features=n_hiddenUnits)
        for i in range(nl):
            if i==0:
                self.lArr.append(nn.Linear(n_hiddenUnits*3,n_hiddenUnits))
            else:
                self.lArr.append(nn.Linear(n_hiddenUnits,n_hiddenUnits))
        self.lArr= nn.ModuleList(self.lArr)
        self.l1= nn.Linear(n_hiddenUnits,op_size)
        
    def forward(self,data,lengths):
        #torch.to("cpu").empty_cache()
        bs= data.shape[1]
        self.h= self.init_hidden(bs)
        embedded= self.embeddings(data)
        embedded= nn.Dropout()(embedded)
        #embedded = pack_padded_sequence(embedded, torch.as_tensor(lengths)) #
        rnn_out, self.h = self.rnn(embedded, (self.h,self.h))
        #rnn_out, lengths = pad_packed_sequence(rnn_out,padding_value=1) #
        avg_pool= F.adaptive_avg_pool1d(rnn_out.permute(1,2,0),1).view(bs,-1)
        max_pool= F.adaptive_max_pool1d(rnn_out.permute(1,2,0),1).view(bs,-1)
        ipForLinearLayer= torch.cat([avg_pool,max_pool,rnn_out[-1]],dim=1)
        for linearlayer in self.lArr:
            outp= linearlayer(ipForLinearLayer)
            ipForLinearLayer= self.bn1(F.relu(outp))
            ipForLinearLayer= nn.Dropout(p=0.6)(ipForLinearLayer)
        outp = self.l1(ipForLinearLayer)
        del embedded;del rnn_out;del self.h;
        #torch.to("cpu").empty_cache()
        return outp
        
    def init_hidden(self, batch_size):
        return torch.zeros((4,batch_size,self.n_hidden),device="cpu")


# In[21]:


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


# In[22]:


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


# In[23]:


epochs= 10
trainLossArr=[]
valLossArr=[]
rocTrainLoss=[]
rocValLoss=[]
model= MyModel(11,len(TEXT.vocab),vectors,1)
loss_func= torch.nn.BCEWithLogitsLoss()
model = model.to("cpu")
for i in range(epochs):
    get_ipython().run_line_magic('time', 'tLoss,vLoss,tRocLoss,vRocLoss= oneEpoch(1e-4)')
    print(f"Epoch - {i}")
    print(f"Train Loss - {tLoss} vs Val Loss is {vLoss}")
    print(f"Train ROC - {tRocLoss} vs Val ROC is {vRocLoss}")
    trainLossArr.append(tLoss)
    valLossArr.append(vLoss)
    rocTrainLoss.append(tRocLoss)
    rocValLoss.append(vRocLoss)


# In[24]:


import matplotlib.pyplot as plt 
plt.plot(trainLossArr,color='b')
plt.plot(valLossArr,color='g')
plt.plot(rocTrainLoss,color='r')
plt.plot(rocValLoss,color='c')
plt.show()


# In[25]:


torch.save(model.state_dict(), "LSTM_transactions_model.pt")


# In[26]:


dfTest = pd.read_csv("Datasets/test.csv")


# In[27]:


dfTest.head()


# In[28]:



dataFields = [("Details", TEXT)]

testDataset= data.TabularDataset(path='Datasets/test.csv', 
                                            format='csv',
                                            fields=dataFields, 
                                            skip_header=True)


# In[29]:


test_iter1 = torchtext.data.Iterator(testDataset, batch_size=32, device=torch.device('cpu'), sort=False, sort_within_batch=False, repeat=False,shuffle=False)


# In[30]:


myPreds=[]
with torch.no_grad():
    model.eval()
    for obj in test_iter1:
        text= torch.transpose(obj.Details[0],0,1)[:2]
        pred= model(obj.Details[0],obj.Details[1])        
        pred= torch.sigmoid(pred)
        myPreds.append(pred.detach().numpy())
        del pred;del obj;


# In[31]:


myPreds= np.vstack(myPreds)
for i, col in enumerate(df.columns[1:]):
    dfTest[col] = myPreds[:, i]


# In[32]:


for i, col in enumerate(df.columns[1:]):
    dfTest[col] = myPreds[:, i]


# In[33]:


dictOfCategories = { i : dfTest.columns[1:][i] for i in range(0, len(dfTest.columns[1:]) ) }


# In[34]:


best = []
for i in range(dfTest.shape[0]):
    best.append(dictOfCategories[dfTest.iloc[i,1:].values.argmax()])


# In[35]:


dfTest["Prediction"] = best


# In[36]:


dfTest = dfTest[["Details","Prediction"]]


# In[37]:


dfTest.head()


# In[38]:


dfTest.to_csv("testPredictions.csv", index=False)


# In[39]:


pd.read_csv("testPredictions.csv")


# In[40]:




