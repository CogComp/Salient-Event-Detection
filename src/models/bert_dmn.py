import torch.nn as nn
import torch
import time
import math
from models.event_rep_module import EventRepModule
from models.episodic_memory_module import *

class MHA_Model(nn.Module):
    def __init__(self, bert_layer, config):
        super().__init__()        
        self.config = config
        self.event_module = EventRepModule(config, bert_layer)
        self.fc = nn.Linear(768, config.hs)
        self.memory_model = EpisodicMemoryModule(config.hs, config.num_passes)
        self.ffc = nn.Linear(config.event_dim+config.hs, 1)
        self.initialize()
#         torch.nn.init.xavier_normal_(self.fc.state_dict()['weight'])
#         torch.nn.init.xavier_normal_(self.ffc.state_dict()['weight'])
        
    def initialize(self):
        nn.init.xavier_uniform_(self.fc.weight.data)
        self.fc.bias.data.zero_()
        nn.init.xavier_uniform_(self.ffc.weight.data)
        self.ffc.bias.data.zero_()

    '''
        Input: B*b*512, B*b*512, B*N*2
        Output: B*N (No Sigmoid because Loss function is BCEWithLogitsLoss)
    '''
    # CAN SAMPLE BEFORE THIS CALLING IT AFTER IT STARTS RUNNING
    def forward(self, input_ids, input_mask, finput_ids, finput_masks, locs, feats):
        xo = self.event_module(input_ids, 
                              input_mask, 
                              locs, 
                              doc_features = True,
                              finput_ids=finput_ids, 
                              finput_masks=finput_masks)
        # GET USEFUL CONTEXT FROM OTHER EVENTS
        x = self.fc(xo) # (B*N*768) -> (B*N*HS)
        B, N = x.shape[0], x.shape[1]
        D = x.repeat(N, 1, 1) # BN*N*HS
        Q = x.clone().detach().view(B*N, -1) # BN*HS
        M, attns = self.memory_model(D, Q) # BN*HS
        M = M.view(B,N,-1)
        # TODO: VISUALISE ATTENTION
        #         print(M.shape)
        #         print(M[0,:10])

        # SELF + CONTEXT
        M = torch.cat((M, xo), 2)
        #         print("After BAKVAAS", xo[0,0,:10])
        y = self.ffc(M).squeeze(2) # (B,N,768+HS) -> (B,N)
        return y    
#         
#         xo = self.event_module(input_ids, input_mask, locs) # (B*b*512, B*b*512, B*N*2) -> (B*N*768)
#         print("Before BAKVAAS", xo[0,0,:10])
