import torch.nn as nn
import torch
import time
import math
from models.event_rep_module import EventRepModule

class SelfVotingAttnModel(nn.Module):
    def __init__(self, bert_layer, config):
        super().__init__()        
        self.config = config
        self.event_module = EventRepModule(config, bert_layer)
        self.d = config.p_key_query_dim
        back_emb = 0
        if(config.sl):
            back_emb += 1
        if(config.unem or config.nem):
            back_emb += 1
        if(config.af):
            back_emb += 1 
        if(config.an):
            back_emb += 1 
        if(config.ps):
            back_emb += 1
        if(config.corref):
            back_emb += 1
        if(config.ch):
            back_emb += 1
        if(config.frame):
            self.proj_key = nn.Linear(2*768 + back_emb, self.d)
            self.proj_query = nn.Linear(2*768 + back_emb, self.d)
        else:
#             self.lin1 = nn.Linear(768 + back_emb, 4*self.d)
#             self.rel1 = nn.ReLU()
#             self.lin2 = nn.Linear(4*self.d, 2*self.d)
#             self.rel2 = nn.ReLU()
#             self.proj_key = nn.Linear(2*self.d, self.d)
#             self.proj_query = nn.Linear(2*self.d, self.d)
            self.proj_key = nn.Linear(768 + back_emb, self.d)
            self.proj_query = nn.Linear(768 + back_emb, self.d)
        self.initialize()
        
    def initialize(self):
        nn.init.xavier_uniform_(self.proj_key.weight.data)
        self.proj_key.bias.data.zero_()
        nn.init.xavier_uniform_(self.proj_query.weight.data)
        self.proj_query.bias.data.zero_()

    '''
        Input: B*N*d, B*N*d
        Output: B*N
    '''
    def apply_attention(self, K, Q):
        B, N = K.shape[0], K.shape[1]
        Qt = torch.transpose(Q, 1, 2) # (B*N*d) -> (B*d*N)
        W = torch.matmul(K, Qt) # (B*N*d, B*d*N) -> (B*N*N)
        i = torch.eye(N, N).unsqueeze(0) # (N*N)
        imat = i.repeat(B, 1, 1).to(K.device)  # (B*N*N)
        W = W * (1 - imat)
        Wa = torch.sum(W, dim=2) #(B*N*N) -> (B*N)
        return Wa, W

    def prep_features(self, feats, dev):
        features = None
        feat = feats[:,:,0:10].type(torch.FloatTensor).to(dev)
        if(self.config.sl):
            features = feat[:,:,0].unsqueeze(2) 
        if(self.config.unem):
            funem = feat[:,:,1].unsqueeze(2)
            if(features is None):
                features = funem
            else:
                features = torch.cat((features, funem), dim=2)
        if(self.config.nem):
            fnem = feat[:,:,7].unsqueeze(2)
            if(features is None):
                features = fnem
            else:
                features = torch.cat((features, fnem), dim=2)
        if(self.config.af):
            faf = feat[:,:,2].unsqueeze(2)
            if(features is None):
                features = faf
            else:
                features = torch.cat((features, faf), dim=2)
        if(self.config.an):
            fan = feat[:,:,4].unsqueeze(2)
            if(features is None):
                features = fan
            else:
                features = torch.cat((features, fan), dim=2)
        if(self.config.ps):
            fps =  feat[:,:,5].unsqueeze(2)
            if(features is None):
                features = fps
            else:
                features = torch.cat((features, fps), dim=2)
        if(self.config.corref):
            fcorref = feat[:,:,6].unsqueeze(2)
            if(features is None):
                features = fcorref
            else:
                features = torch.cat((features, fcorref), dim=2)
        return features       
        
        
    '''
        Input: B*b*512, B*b*512, -, -, B*N*2, B*N*x
        Output: B*N
    '''
    def forward(self, input_ids, input_mask, finput_ids, finput_masks, locs, feats):
        if(self.config.frame):
            x = self.event_module(input_ids, 
                              input_mask, 
                              locs, 
                              doc_features = False,
                              finput_ids=finput_ids, 
                              finput_masks=finput_masks) # (B*b*512, B*b*512, B*N*2) -> (B*N*768)
        else:
            x = self.event_module(input_ids, 
                              input_mask, 
                              locs, 
                              doc_features = True,
                              finput_ids=finput_ids, 
                              finput_masks=finput_masks) # (B*b*512, B*b*512, B*N*2) -> (B*N*768
        
        features = self.prep_features(feats, x.device)
        if(features is not None):
            x = torch.cat((x, features), dim=2) 
        K = self.proj_key(x) # (B*N*768) -> (B*N*d)
        Q = self.proj_query(x) # (B*N*768) -> (B*N*d)
        score, attn = self.apply_attention(K, Q) # (B*N*d, B*N*d) -> (B*N)
        return score, attn    