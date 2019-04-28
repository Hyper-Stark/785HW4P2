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

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_data_file = '../data/test.npy'
testloader = loader(test_data_file)

model = ZLNet().to(DEVICE)
# model.load_state_dict(torch.load("model.pt"))
model.eval()

result = []
for inputs, lens in testloader:
    
    lens = lens.to(DEVICE)
    inputs = inputs.to(DEVICE)
    
    charindice = model(inputs, lens)
    pred = ''.join([CHARSET[idx] for idx in charindice])
    result.append(pred)

#write to file
with open("result.csv",'w') as f:
    f.write("Id,Predicted\n")
    for i,item in enumerate(result):
        f.write(','.join([str(i),item]) + '\n')
