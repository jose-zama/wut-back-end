import torch
import numpy as np
import torch.optim as optim
from sklearn.metrics import roc_auc_score

class BatchGenerator:
    def __init__(self, dl):
        self.dl = dl
        self.yFields = ['Auto & Transport', 'Bills & Utilities', 'Entertainment',
       'Fees & Charges', 'Food & Dining', 'Gifts & Donations',
       'Health & Fitness', 'Shopping', 'Transfer', 'Travel', 'Withdrawal']
        self.x= 'Details'
        
    def __len__(self):
        return len(self.dl)
    
    def __iter__(self):
        for batch in self.dl:
            X = getattr(batch, self.x)
            y = torch.transpose( torch.stack([getattr(batch, y) for y in self.yFields]),0,1)
            yield (X,y)

def getValidationLoss(validation,model,loss_func):
    model.eval()
    runningLoss=0
    valid_batch_it = BatchGenerator(validation)
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

def oneEpoch(lr, training, validation, model, loss_func):
    train_batch_it = BatchGenerator(training)
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
    valLoss,valRocLoss= getValidationLoss(validation,model,loss_func)
    torch.cuda.empty_cache()
    return runningLoss,valLoss,trainRocLoss,valRocLoss

