import torch
import torch.nn as nn
import torch.nn.functional as F

from constant import CHARIDX
from dataloader import BATCH_SIZE
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pad_packed_sequence

MAX_LENGTH = 300
EMBEDDING_DIM = 128
TEACHER_FORCING_BAR = 0.01
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ZLNet(nn.Module):

    def __init__(self):
        super(ZLNet, self).__init__()
        self.listener = Listener(40,512)
        self.speller = Speller(BATCH_SIZE)

    def forward(self, x, seqlens, y = None):
        keys,values,newlens = self.listener(x,seqlens)
        if y is not None:
            pred = self.speller(keys, values, y, newlens)
        else:
            pred = self.speller.decode(keys, values, newlens)
        return pred

class Listener(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(Listener,self).__init__()
        self.pblstm1 = pBLSTM(input_size,hidden_size)
        self.dropout1 = nn.Dropout(p=0.25)
        self.pblstm2 = pBLSTM(2*hidden_size,hidden_size)
        self.dropout2 = nn.Dropout(p=0.25)
        self.pblstm3 = pBLSTM(2*hidden_size,hidden_size)
        self.dropout3 = nn.Dropout(p=0.25)

        self.keymlp = nn.Sequential(
            nn.Linear(hidden_size*2,512),
            nn.LeakyReLU(negative_slope = 0.1),
            nn.Linear(512,256)
        )

        self.valuemlp = nn.Sequential(
            nn.Linear(hidden_size*2,512),
            nn.LeakyReLU(negative_slope = 0.1),
            nn.Linear(512,256)
        )

    def forward(self, x, seqlens):
        x = self.pblstm1(x)
        x = self.dropout1(x)
        x = self.pblstm2(x)
        x = self.dropout2(x)
        x = self.pblstm3(x)
        x = self.dropout3(x)

        seqlens = (seqlens//8).type(torch.IntTensor)
        
        key = self.keymlp(x)
        value = self.valuemlp(x)

        return key,value,seqlens


class Speller(nn.Module):

    def __init__(self, batch_size, input_size=256, hidden_size=256):

        super(Speller,self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(len(CHARIDX),EMBEDDING_DIM)
        self.lstmcell1 = nn.LSTMCell(input_size+EMBEDDING_DIM, EMBEDDING_DIM)
        self.dropout1 = nn.Dropout(p=0.25)
        self.lstmcell2 = nn.LSTMCell(EMBEDDING_DIM, hidden_size)
        self.dropout2 = nn.Dropout(p=0.25)
        self.lstmcell3 = nn.LSTMCell(hidden_size, hidden_size)
        self.dropout3 = nn.Dropout(p=0.25)
        self.attention = Attention()
        self.predMLP = nn.Linear(256+hidden_size,len(CHARIDX))
        
        h1shape = (batch_size, EMBEDDING_DIM)
        h2shape = (batch_size, hidden_size)
        self.h1, self.c1 = [nn.Parameter(torch.zeros(h1shape)) for i in range(2)]
        self.h2, self.c2 = [nn.Parameter(torch.zeros(h2shape)) for i in range(2)]
        self.h3, self.c3 = [nn.Parameter(torch.zeros(h2shape)) for i in range(2)]

        self.start = nn.Parameter(torch.zeros(h2shape))
    
    def forward(self, keys, values, y, seqlens):

        results, indice = [],[]
        yembeeding = self.embedding(y)

        h1, c1 = self.h1, self.c1
        h2, c2 = self.h2, self.c2
        h3, c3 = self.h3, self.c3

        # start state: s_0, the beginning of a sentence
        # must be a <sos>, and 'must' means the probability 
        # is 1. Because <sos>('@') in my char dict has
        # index 0, so the 0th element's probability
        # in this matrix has to be 1.
        start = torch.zeros(self.batch_size, len(CHARIDX)).to(DEVICE)
        start[:,0] = 1
        results.append(start)
        indice.append(torch.zeros(self.batch_size,dtype=torch.long).to(DEVICE))

        # compute context c_0
        context, attdis = self.attention(self.start,keys,values,seqlens)

        for i in range(y.shape[1] - 1):

            embedding = yembeeding[:,i,:]
            # teaching force
            if i != 0 and torch.rand(1).to(DEVICE).item() < TEACHER_FORCING_BAR:
                embedding = self.embedding(charmaxi)
            
            # s_i = RNN(s_i-1, y_i-1, c_i-1)
            # c_i-1 = Attention(s_i-1, h), h is a vector
            input = torch.cat((embedding,context), dim=1)
            h1,c1 = self.lstmcell1(input, (h1,c1))
            h1 = self.dropout1(h1)
            h2,c2 = self.lstmcell2(h1, (h2,c2))
            h2 = self.dropout2(h2)
            h3,c3 = self.lstmcell3(h2, (h3,c3))
            h3 = self.dropout3(h3)

            # c_i = Attention(s_i, h), 
            # h3 is actually s_i, 
            # key/values is actually h
            context, attdis = self.attention(h3,keys,values,seqlens)

            # we concatnate lstm's output h3
            # and context information
            rnnres_hid = torch.cat((h3, context), dim=1)

            # use MLP to predict
            output = self.predMLP(rnnres_hid)
            results.append(output)

            # get the predicted character, 
            # index can be used to generate embeddings
            # to be used as teaching force
            charmaxv, charmaxi = output.max(1)
            indice.append(charmaxi)

        result = torch.stack(results)
        indice = torch.stack(indice)
        result = result.permute(1,0,2)
        indice = indice.permute(1,0)
        return result, indice

    def flat(self,matrix):
        return matrix.sum(dim=0).unsqueeze(0)/matrix.shape[0]

    def decode(self, keys, values, seqlens):

        indice = []
        h1,c1 = self.flat(self.h1), self.flat(self.c1)
        h2,c2 = self.flat(self.h2), self.flat(self.c2)
        h3,c3 = self.flat(self.h3), self.flat(self.c3)
        start = self.flat(self.start)

        last = 0
        indice.append(last)
        embedding = self.embedding(torch.tensor([last]).to(DEVICE))
        context, attdis = self.attention(start, keys, values, seqlens)
        
        for i in range(MAX_LENGTH):

            # s_i = RNN(s_i-1, y_i-1, c_i-1)
            # c_i-1 = Attention(s_i-1, h), h is a vector
            input = torch.cat((embedding,context), dim=1)
            h1,c1 = self.lstmcell1(input, (h1,c1))
            h1 = self.dropout1(h1)
            h2,c2 = self.lstmcell2(h1, (h2,c2))
            h2 = self.dropout2(h2)
            h3,c3 = self.lstmcell3(h2, (h3,c3))
            h3 = self.dropout3(h3)

            # c_i = Attention(s_i, h), 
            # h3 is actually s_i, 
            # key/values is actually h
            context, attdis = self.attention(h3,keys,values,seqlens)
            
            # we concatnate lstm's output h3
            # and context information
            rnnres_hid = torch.cat((h3, context), dim=1)

            # use MLP to predict
            output = self.predMLP(rnnres_hid)
            maxv, maxi = output.max(1)

            # append result
            last = maxi.item()
            indice.append(last)
            embedding = self.embedding(torch.tensor([last]).to(DEVICE))
            
            # found end, end prediction
            if indice[-1] == 0:
                break
        
        return torch.tensor([indice])
    

class Attention(nn.Module):

    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, keys, values, seqlens):
        
        batch_size = keys.shape[0]
        seq_len = keys.shape[1]

        # (batch_size, lstm_hidden_size) -> 
        # (batch_size, lstm_hidden_size, 1)
        query = query.unsqueeze(2)

        # attention evaluation function: dot product
        energy = torch.bmm(keys, query)
        energy = energy.squeeze(2)

        # generate mask for padded sequence
        mask = torch.zeros(batch_size, seq_len).to(DEVICE)
        for i in range(batch_size):
            mask[i][:seqlens[i]] = 1
        
        # attention distribution
        attdis = F.softmax(energy, dim=1)
        attdis = attdis * mask

        # normalize attention distribution
        normalization = torch.sum(attdis, 1)
        attdis = (attdis/normalization.unsqueeze(1)).unsqueeze(1)
        context = torch.bmm(attdis, values).squeeze(1)

        return context, attdis



class pBLSTM(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(pBLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size*2, hidden_size, 1, batch_first = True, bidirectional=True)

    def forward(self, x):

        array = []
        for tsr in x:
            seq_length = tsr.shape[0]
            feature_dim = tsr.shape[1]
            if seq_length % 2 != 0:
                seq_length -= 1
                tsr = tsr.narrow(0,0,seq_length)
            convd = tsr.contiguous().view(seq_length//2, feature_dim*2)
            array.append(convd)

        packedx = pack_sequence(array)
        output, (hidden, cell) = self.lstm(packedx, None)
        paddedx, _ = pad_packed_sequence(output)
        return paddedx.transpose(1,0)
