import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self,op_size,n_tokens,pretrained_vectors,nl=2,bidirectional=True,emb_sz=300,n_hiddenUnits=100):
        super(MyModel, self).__init__()
        self.n_hidden= n_hiddenUnits
        self.embeddings= nn.Embedding(n_tokens,emb_sz)
        self.embeddings.weight.data.copy_(pretrained_vectors)
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
        rnn_out, self.h = self.rnn(embedded, (self.h,self.h))
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