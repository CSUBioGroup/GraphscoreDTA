import pickle
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dgl
import dgl.nn.pytorch as dglnn
import dgl.function as dglfn
from collections import deque
from tqdm.auto import tqdm
from IPython.display import clear_output
import time

from dataset import SepDataset, collate_fn
from model import ModelNew
import metrics

seed = np.random.randint(2021, 2022) ##random

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.manual_seed(seed)
np.random.seed(seed)


def timeSince(since):
    now = time.time()
    s = now - since
    return now, s

def train(model, train_loader_compound, criterion, optimizer,epoch,device):
    model.train()
    tbar = tqdm(train_loader_compound, total=len(train_loader_compound))
    losses = []
    t = time.time()
    for i, data in enumerate(tbar):
        data0 = [i.to(device) for i in data[0]]
        ga, gr, gi, aff = data0 
        vina = data[1]
        y_pred = model(ga,gr,gi,vina).squeeze()
        y_true = aff.float().squeeze()
        
        assert y_pred.shape == y_true.shape
        loss = criterion(y_pred,y_true).cuda()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5) 
        optimizer.step()    
        optimizer.zero_grad()     
        losses.append(loss.item())
#         tbar.set_description(f'epoch {epoch+1} loss {np.mean(losses[-10:]):.4f} grad {grad_norm:.4f}')
        
    m_losses=np.mean(losses)
    
    return m_losses   

def valid(model, valid_loader_compound, criterion,device):
    model.eval()
    losses = []
    outputs = []
    targets = []
    tbar = tqdm(valid_loader_compound, total=len(valid_loader_compound))
    for i, data in enumerate(tbar):
        data0 = [i.to(device) for i in data[0]]
        ga, gr, gi, aff = data0 
        vina = data[1]
        with torch.no_grad():
            y_pred = model(ga,gr,gi,vina).squeeze()
        y_true = aff.float().squeeze()
        assert y_pred.shape == y_true.shape
        loss = criterion(y_pred,y_true).cuda()
        losses.append(loss.item())
        outputs.append(y_pred.cpu().detach().numpy().reshape(-1))
        targets.append(y_true.cpu().detach().numpy().reshape(-1))
    targets = np.concatenate(targets).reshape(-1)
    outputs = np.concatenate(outputs).reshape(-1)
        
    evaluation = {
        'c_index': metrics.c_index(targets, outputs),
        'RMSE': metrics.RMSE(targets, outputs),
        'MAE': metrics.MAE(targets, outputs),
        'SD': metrics.SD(targets, outputs),
        'CORR': metrics.CORR(targets, outputs),}
    ml=np.mean(losses)  
    
    return ml, evaluation

def main():
    F=open(r'../test_set/train_valte_comp.pkl','rb')
    content=pickle.load(F)
    vina_list= []
    graphs = dgl.load_graphs('../test_set/all_in_one_graph_train13851.bin')[0]
    labels = pd.read_csv('../test_set/labels_train13851.csv')
    vina_terms=open(r'../test_set/Vina_terms13851.pkl','rb')
    vina=pickle.load(vina_terms)
    for i in range(13851):
        if labels.id[i] in vina.keys():
            vina_list.append(vina[labels.id[i]])
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compound_train = content[0]
    compound_valid = content[1]
    compound_test = content[2]
    
    train_dataset_compound = SepDataset([graphs[i] for i in compound_train], [vina_list[i] for i in compound_train], [labels.id[i] for i in compound_train], [labels.affinity[i] for i in compound_train], ['a_conn','r_conn', 'int_l'])
    valid_dataset_compound = SepDataset([graphs[i] for i in compound_valid], [vina_list[i] for i in compound_valid], [labels.id[i] for i in compound_valid], [labels.affinity[i] for i in compound_valid], ['a_conn','r_conn', 'int_l'])
    test_dataset_compound = SepDataset([graphs[i] for i in compound_test], [vina_list[i] for i in compound_test], [labels.id[i] for i in compound_test], [labels.affinity[i] for i in compound_test], ['a_conn','r_conn', 'int_l']) 
    
    train_loader_compound = DataLoader(train_dataset_compound, batch_size=8, shuffle=True, num_workers=0, collate_fn=collate_fn,pin_memory=False,drop_last=False,)
    valid_loader_compound = DataLoader(valid_dataset_compound, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader_compound = DataLoader(test_dataset_compound, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = ModelNew()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), 1.2e-4, weight_decay=1e-6)   ### (model.parameters(), 1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=40, eta_min=1e-6)
    criterion = torch.nn.MSELoss()
    
    n_epoch = 80
    best_R = 0.0
    for epoch in range(n_epoch):
        ll = train(model, train_loader_compound, criterion, optimizer,epoch,device)
        if epoch%1==0:
            l,evaluation = valid(model, valid_loader_compound, criterion,device)
            l_, evaluation_ = valid(model, test_loader_compound, criterion,device)
            print(f'epoch {epoch+1} train_loss {ll:.5f} valid_loss {l:.5f}')
            clear_output()
            if evaluation_['CORR']>best_R:
                best_R= evaluation_['CORR']
                torch.save({'model': model.state_dict()}, '../model/model.pth')
        scheduler.step()

if __name__ == "__main__":
    main()