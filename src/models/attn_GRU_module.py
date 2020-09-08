import torch.nn as nn
import torch
import time
from torch.autograd import Variable
import math

class AttnGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttnGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Wr = nn.Linear(input_size, hidden_size)
        self.Ur = nn.Linear(hidden_size, hidden_size)
        self.W = nn.Linear(input_size, hidden_size)
        self.U = nn.Linear(hidden_size, hidden_size)
        self.initialize()
        
    def initialize(self):
        nn.init.xavier_uniform_(self.Wr.weight.data)
        self.Wr.bias.data.zero_()
        nn.init.xavier_uniform_(self.Ur.weight.data)
        self.Ur.bias.data.zero_()
        nn.init.xavier_uniform_(self.W.weight.data)
        self.W.bias.data.zero_()
        nn.init.xavier_uniform_(self.U.weight.data)
        self.U.bias.data.zero_()

    '''
        Input: B*HS, B*HS, B
        Output: B*HS
    '''
    def forward(self, fact, hi_1, g):
        r_i = torch.sigmoid(self.Wr(fact) + self.Ur(hi_1))
        h_tilda = torch.tanh(self.W(fact) + r_i*self.U(hi_1))
        g = g.unsqueeze(1)
        hi = g*h_tilda + (1 - g)*hi_1
        return hi # Returning the next hidden state considering the first fact and so on.


class AttnGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttnGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.AttnGRUCell = AttnGRUCell(input_size, hidden_size)

    '''
        Input: BN*N*HS, BN*N
        Output: B*HS
    '''
    def forward(self, D, G):
        h_0 = Variable(torch.zeros(self.hidden_size)).to(D.device)
        for sen in range(D.size()[1]):
            sentence = D[:, sen, :]
            g = G[:, sen]
            if sen == 0: # Initialization for first sentence only 
                hi_1 = h_0.unsqueeze(0).expand_as(sentence)
            hi_1 = self.AttnGRUCell(sentence, hi_1, g)

        C = hi_1 # Final hidden vector as the contextual vector used for updating memory
        return C