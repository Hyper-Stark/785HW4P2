import time
import torch
import numpy as np
import torch.utils.data as data

from constant import CHARIDX
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence

class Dataset(data.Dataset):
    def __init__(self, data, labels = None):
        super(Dataset, self).__init__()
        self.data = data
        self.labels = labels
        self.trainmode = True if labels is not None else False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.trainmode:
            a,b = self.data[index], self.labels[index]
            return (torch.tensor(a),torch.tensor(b, dtype=torch.long))
        else:
            a = self.data[index]
            return torch.tensor(a)

def vectorize(data):
    res = []
    for sentence in data:
        sent = '@'+' '.join([word.decode("utf-8") for word in sentence])+'@'
        vec = [CHARIDX[c] for c in sent]
        res.append(vec)
    return res

BATCH_SIZE = 32
NUM_WORKERS = 4

def loader(dataf,labelf=None):

    _begin_time = time.time()
    dset = ld = None
    chunk = np.load(dataf,encoding="bytes")
    
    #train mode
    if labelf is not None:
        labels = np.load(labelf, encoding="bytes")
        labels = vectorize(labels)
        dset = Dataset(chunk,labels)
        
        if torch.cuda.is_available():
            ld = data.DataLoader(\
                dset, \
                batch_size=BATCH_SIZE, \
                shuffle=True, \
                drop_last=True, \
                collate_fn=collate_train, \
                num_workers=NUM_WORKERS)
        else:
            ld = data.DataLoader(\
                dset, \
                batch_size=BATCH_SIZE, \
                shuffle=True, \
                drop_last=True, \
                collate_fn=collate_train)
    #test mode
    else:
        dset = Dataset(chunk)
        if torch.cuda.is_available():
            ld = data.DataLoader(\
                dset, \
                batch_size=1, \
                shuffle=False, \
                drop_last=False, \
                collate_fn=collate_test, \
                num_workers=NUM_WORKERS)
        else:
            ld = data.DataLoader(\
                dset, \
                batch_size=1, \
                shuffle=False, \
                drop_last=False, \
                collate_fn=collate_test)
    
    _end_time = time.time()
    print("load data time cost: " + str(_end_time - _begin_time))
    return ld

def collate_train(pairs):

    #split column
    inputs, labels = zip(*pairs)

    #collect lengths data
    seqlens = [(seq.shape[0], i) for i,seq in enumerate(inputs)]
    lablens = [lab.shape[0] for lab in labels]
    #sort indices according descending lengths
    sorted_seqlens = sorted(seqlens, key=lambda x: x[0], reverse=True)
    seqs,labs,nseqlens,nlablens = [],[],[],[]
    for lenz, oidx in sorted_seqlens:
        nseqlens.append(lenz)
        seqs.append(inputs[oidx])
        labs.append(labels[oidx])
        nlablens.append(labels[oidx].shape[0])

    pseqs = pad_sequence(seqs, batch_first=True)
    plabs = pad_sequence(labs, batch_first=True)

    tnseqlens = torch.tensor(nseqlens,dtype=torch.int)
    tnlablens = torch.tensor(nlablens,dtype=torch.int)
    return pseqs,plabs,tnseqlens,tnlablens

def collate_test(inputs):
    print(inputs)
    return inputs