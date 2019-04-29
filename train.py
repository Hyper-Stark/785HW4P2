import pytz
import time
import torch
import torchvision
import numpy as np
import torch.nn as nn
import datetime as dt
import Levenshtein as L
import torch.optim as optim

from model import ZLNet
from utils import details
from constant import CHARSET
from dataloader import loader


EPOCHS = 15
TIME_ZONE = pytz.timezone("America/New_York")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dev_data_file = '../data/dev.npy'
dev_trans_file = '../data/dev_transcripts.npy'
train_data_file = '../data/train.npy'
train_trans_file = '../data/train_transcripts.npy'

train_loader = loader(train_data_file, train_trans_file)
valid_loader = loader(dev_data_file)

model = ZLNet().to(DEVICE)
#model.load_state_dict(torch.load("model.pt"))
criterion = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(),lr=0.001)
_begin_time = time.time()

#manually load valid labels
labels = np.load(dev_trans_file, encoding="bytes")
validlabels = []
for sentence in labels:
    validlabels.append('@'+' '.join([word.decode('utf-8') for word in sentence])+'@')

for epoch in range(EPOCHS):

    model.train()

    #this epoch
    nt = dt.datetime.now(TIME_ZONE)
    print(" ")
    print("Starting epoch "+str(epoch)+" at "+nt.strftime("%H:%M"))

    i = ave_loss = ave_dis = total = 0
    for inputs,values,ilens,vlens in train_loader:
        
        #clean gradient
        opt.zero_grad()

        #use cuda
        inputs = inputs.to(DEVICE)
        values = values.to(DEVICE)
        ilens = ilens.to(DEVICE)

        #calculation
        output, charindice = model(inputs,ilens,values)

        #to string
        for i in range(vlens.shape[0]):
            pindice = charindice[i]
            tindice = values[i,:vlens[i]]
            pred = ''.join([CHARSET[idx] for idx in pindice])
            truth = ''.join([CHARSET[idx] for idx in tindice])
            ave_dis += L.distance(pred,truth)

        #flatten
        mergedim = output.shape[0] * output.shape[1]
        flattenpred = output.contiguous().view(mergedim,-1)
        flattentruth = values.view(values.numel())
        loss = criterion(flattenpred, flattentruth)

        #generate mask
        mask = torch.zeros(values.shape).to(DEVICE)
        for i in range(vlens.shape[0]):
            mask[i][:vlens[i]] = 1
        mask = mask.view(mask.numel())

        #mask loss
        loss = (loss * mask).sum()
        ave_loss += loss
        total += vlens.sum()

        #backward
        loss.backward()
        opt.step()

    #validation
    model.eval()
    validres = []
    for inputs,ilens in valid_loader:

        ilens = ilens.to(DEVICE)
        inputs = inputs.to(DEVICE)
        
        charindice = model(inputs, ilens)
        charindice = charindice[0]
        pred = ''.join([CHARSET[idx] for idx in charindice])
        validres.append(pred[1:-1])

    pairs = zip(validlabels, validres)
    diss = [(L.distance(truth,pred), len(pred)) for truth, pred in pairs]
    dis, lens = zip(*diss)
    vave_dis, vtotal = sum(dis), sum(lens)

    _end_time = time.time()
    details((_end_time - _begin_time), (ave_loss/total).item(), ave_dis/total.item(), 0, vave_dis/vtotal)
    _begin_time = time.time()

    torch.save(model.state_dict(),"models/model-"+str(int(_begin_time))+"-"+str(int(vave_dis/vtotal))+".pt")
