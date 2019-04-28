import pytz
import time
import torch
import torchvision
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

train_loader = loader(dev_data_file, dev_trans_file)
valid_loader = loader(dev_data_file, dev_trans_file, batch_size=1)

model = ZLNet().to(DEVICE)
criterion = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(),lr=0.001)
_begin_time = time.time()

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
            pindice = charindice[i,:vlens[i]]
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
    i = vave_loss = vave_dis = vtotal = 0
    for inputs,values,ilens,vlens in valid_loader:

        #use cuda
        inputs = inputs.to(DEVICE)
        values = values.to(DEVICE)
        ilens = ilens.to(DEVICE)

        #calculation
        output, charindice = model(inputs,ilens)

        #to string
        for i in range(vlens.shape[0]):
            pindice = charindice[i,:vlens[i]]
            tindice = values[i,:vlens[i]]
            pred = ''.join([CHARSET[idx] for idx in pindice])
            truth = ''.join([CHARSET[idx] for idx in tindice])
            vave_dis += L.distance(pred,truth)

        vtotal += vlens.sum()

    _end_time = time.time()
    details((_end_time - _begin_time), (ave_loss/total).item(), ave_dis/total.item(), (vave_loss/vtotal).item(), vave_dis/vtotal.item())
    _begin_time = time.time()

    torch.save(model.state_dict(),"models/model-"+str(int(_begin_time))+"-"+str(int(vave_dis/vtotal))+".pt")
