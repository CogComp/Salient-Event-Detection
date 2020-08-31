import torch.nn as nn
import torch
import time
import math
from models.attn_GRU_module import *
from torch.nn import functional as F
    
class MemoryModule(nn.Module): # Takes Document sentences, question and prev_mem as its and output next_mem
    def __init__(self, hidden_size):
        super(MemoryModule, self).__init__()
        self.hidden_size = hidden_size
        self.AttnGRU = AttnGRU(hidden_size, hidden_size)
        self.W1 = nn.Linear(4*hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, 1)
        self.W_mem = nn.Linear(3*hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.initialize()
#         torch.nn.init.xavier_normal_(self.W1.state_dict()['weight'])
#         torch.nn.init.xavier_normal_(self.W2.state_dict()['weight'])
#         torch.nn.init.xavier_normal_(self.W_mem.state_dict()['weight'])
    
    def initialize(self):
        nn.init.xavier_uniform_(self.W1.weight.data)
        self.W1.bias.data.zero_()
        nn.init.xavier_uniform_(self.W2.weight.data)
        self.W2.bias.data.zero_()
        nn.init.xavier_uniform_(self.W_mem.weight.data)
        self.W_mem.bias.data.zero_()

    '''
        Input: B*S*HS, B*1*HS, B*1*HS
        Output: B*S
    '''
    def gateMatrix(self, D, Q, prev_mem):
        Q = Q.expand_as(D)
        prev_mem = prev_mem.expand_as(D)
        batch_size,  embedding_length = D.shape[0], D.shape[2]

        z = torch.cat([D*Q, D*prev_mem, torch.abs(D - Q), torch.abs(D - prev_mem)], dim=2)
        z = z.view(-1, 4*embedding_length)
        Z = self.W2(torch.tanh(self.W1(z)))
        Z = Z.view(batch_size, -1) # B*S*4*HS
        G = F.softmax(Z, dim=1)
        return G
    
    '''
        Input: B*S*HS, B*HS, B*HS
        Output: B*HS, B*S
    '''
    def forward(self, D, Q, prev_mem):
        Q = Q.unsqueeze(1)
        prev_mem = prev_mem.unsqueeze(1)
        G = self.gateMatrix(D, Q, prev_mem) # (B*S*HS, B*HS, B*HS) -> (B*S)
        C = self.AttnGRU(D, G) # (B*S*HS, B*S) -> (B*HS)
        # VERTICAL RNN
        # Now considering prev_mem, C and question, we will update the memory state as follows
        concat = torch.cat([prev_mem.squeeze(1), C, Q.squeeze(1)], dim=1) # (B* 3*HS)
        concat = self.dropout(concat)
        next_mem = F.relu(self.W_mem(concat)) # (B* 3*HS) -> # (B*HS)
        return next_mem, G
    
class EpisodicMemoryModule(nn.Module):
    def __init__(self, hidden_size, num_passes):
        super(EpisodicMemoryModule, self).__init__()
        self.hidden_size = hidden_size
        self.num_passes = num_passes
        self.memory = MemoryModule(self.hidden_size)

    '''
        Input: BN*N*HS, BN*HS
        Output: BN*HS, num_passes*[BN*HS]
    '''
    def forward(self, D, Q):
        M = Q
        attns = []
        for passes in range(self.num_passes):
#             print("Pass: ", passes)
#             print(M.flatten()[:10])#, Q.flatten()[:10])
            M, G = self.memory(D, Q, M)
            attns.append(G)
        return M, attns