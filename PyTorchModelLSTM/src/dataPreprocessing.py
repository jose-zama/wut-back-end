import pandas as pd
import re
import spacy
import torch
import torchtext
from sklearn.model_selection import train_test_split
from spacy.lang.en.examples import sentences
from torchtext import data

my_tok = spacy.load('en_core_web_sm')
my_stopwords = spacy.lang.en.stop_words.STOP_WORDS

def applyOneHotEncoding(df):
    df = df.join(pd.get_dummies(df["Category"]))
    df.drop("Category", axis=1, inplace=True)
    return df

def splitData(df):
    traininig, validation = train_test_split(df, test_size=0.2)
    traininig.to_csv("./Datasets/training.csv", index_label=False, index=False)
    validation.to_csv("./Datasets/validation.csv", index_label=False, index=False)
    return

def spacyTok(x):
    x= re.sub(r'[^a-zA-Z\s]','',x)
    x= re.sub(r'[\n]','',x)
    return [tok.text for tok in my_tok.tokenizer(x)]

def setDataFields():
    TEXT = data.Field(lower=True, tokenize=spacyTok,eos_token='EOS',stop_words=my_stopwords,include_lengths=True)
    LABEL = data.Field(sequential=False, 
                         use_vocab=False, 
                         pad_token=None, 
                            unk_token=None)

    dataFields = [("Details", TEXT),
             ("Auto & Transport", LABEL), ("Bills & Utilities", LABEL),
             ("Entertainment", LABEL), ("Fees & Charges", LABEL),  
             ("Food & Dining", LABEL), ("Gifts & Donations", LABEL),
             ("Health & Fitness", LABEL), ("Shopping", LABEL),
             ("Transfer", LABEL), ("Travel", LABEL), ("Withdrawal", LABEL)]
    
    return TEXT, dataFields

    
def setTabularData(dataFields):
    train,val = data.TabularDataset.splits(path="Datasets/",train="training.csv", validation="validation.csv", format="csv", fields=dataFields, skip_header=True)
    return train,val

def buildVocabulary(TEXT, train ,val):
    TEXT.build_vocab(train,val, vectors=['fasttext.simple.300d'])
    return TEXT

def setDataForBatches(train, val):
    train, val = torchtext.data.BucketIterator.splits(datasets=(train, val),
                                            batch_sizes=(64, 64),
                                            sort_key=lambda x: len(x.Details),
                                            device=torch.device('cpu'),
                                            sort_within_batch=True
                                                     )
    return train, val

def setVectors(train):
    vectors= train.fields['Details'].vocab.vectors.to("cpu")
    return vectors