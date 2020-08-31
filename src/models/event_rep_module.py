import torch.nn as nn
import torch
import time
import math

'''
    Event Representation: Trigger, ARG0, ARG1, LOC, TMP (256+128+128+128+128=768)
'''
class EventRepModule(nn.Module):
    def __init__(self, config, bert_layer):
        super().__init__()        
        self.config = config
        self.bert_layer = bert_layer
        self.out_linear = nn.Linear(in_features=config.event_dim, out_features=1)
        self.output_softmax_scores = nn.Softmax(dim=1)        
      
        if(config.event_only_trigger):
            self.tres = config.event_dim
            self.trig_lstm = nn.LSTM(config.event_dim, int(self.tres/2), 1, bidirectional=True)
        else:
            self.tres = 2*int(config.event_dim/3)
            self.a0es, self.a1es = int(config.event_dim/12), int(config.event_dim/12)
            self.les, self.tes = int(config.event_dim/12), int(config.event_dim/12)
            
            self.trig_lstm = nn.LSTM(config.event_dim, int(self.tres/2), 1, bidirectional=True)
            self.arg0_lstm = nn.LSTM(config.event_dim, int(self.a0es/2), 1, bidirectional=True)
            self.arg1_lstm = nn.LSTM(config.event_dim, int(self.a1es/2), 1, bidirectional=True)
            self.loc_lstm = nn.LSTM(config.event_dim, int(self.les/2), 1, bidirectional=True)
            self.tmp_lstm = nn.LSTM(config.event_dim, int(self.tes/2), 1, bidirectional=True)
        self.init_weights()
        
    # Weights Initialization
    def init_weights(self, init_range=0.1):
        self.out_linear.weight.data.uniform_(-init_range, init_range)
        
    '''
        Convert variable length token embeddings to a fixed size embedding per event constituent
    '''
    def get_constituent_emb(self, multi_word_emb, mode, constituent):
        #Return If
        if(multi_word_emb.shape[0]<=0): # LET IT BE ZEROES
            return None

        if(mode=="LSTM"):
            if(constituent == "TRIGGER"):
                self.trig_lstm.flatten_parameters()
                lstm_out, (lstm_hid,_) = self.trig_lstm(multi_word_emb.unsqueeze(1)) # size:[batch_size,78,256]
            if(constituent == "ARG0"):
                lstm_out, (lstm_hid,_) = self.arg0_lstm(multi_word_emb.unsqueeze(1)) # size:[batch_size,78,256]
            if(constituent == "ARG1"):
                lstm_out, (lstm_hid,_) = self.arg1_lstm(multi_word_emb.unsqueeze(1)) # size:[batch_size,78,256]
            if(constituent == "LOC"):
                lstm_out, (lstm_hid,_) = self.loc_lstm(multi_word_emb.unsqueeze(1)) # size:[batch_size,78,256]
            if(constituent == "TMP"):
                lstm_out, (lstm_hid,_) = self.tmp_lstm(multi_word_emb.unsqueeze(1)) # size:[batch_size,78,256]
            return torch.flatten(lstm_hid)
        
        if(mode=="mean"):
            return torch.mean(multi_word_emb, 0)

    '''
        Input: x*512, x*512
        Output: x*512, x*512, x*512, x*512
    '''
    def get_adj_ids_masks(self, inp, masks):
        left_inp = torch.zeros([inp.shape[0], self.config.p_max_seq], dtype=torch.long).to(inp.device)
        left_mask = torch.zeros([inp.shape[0], self.config.p_max_seq], dtype=torch.long).to(inp.device)
        right_inp = torch.zeros([inp.shape[0], self.config.p_max_seq], dtype=torch.long).to(inp.device)
        right_mask = torch.zeros([inp.shape[0], self.config.p_max_seq], dtype=torch.long).to(inp.device)
        for i in range(0, inp.shape[0]):
            if(i>0):
                left_inp[i] = torch.cat((inp[i-1, 256:], inp[i, 0:256]), dim=0)
                left_mask[i] = torch.cat((masks[i-1, 256:], masks[i, 0:256]), dim=0)
            if(i<inp.shape[0]-1):
                right_inp[i] = torch.cat((inp[i, 256:], inp[i+1, 0:256]), dim=0)
                right_mask[i] = torch.cat((masks[i, 256:], masks[i+1, 0:256]), dim=0)
        return left_inp, left_mask, right_inp, right_mask
        
    '''
    Input:  Document's tokens, Documents's bert mask, Document's event's locations
            B*b*512, B*b*512, B*N*2
    Output: Event Representation
            B*N*768
    '''
    def get_event_embeddings(self, input_ids, input_mask, locs, finput_ids, finput_masks, mode="LSTM"):
        B = locs.shape[0]
        N = locs.shape[1]
        b = input_ids.shape[1]
        emb_output = torch.zeros([B, N, self.config.event_dim], dtype=torch.float).to(input_ids.device)

        input_ids_flat = input_ids.view(-1, self.config.p_max_seq)
        inp_mask_flat = input_mask.view(-1, self.config.p_max_seq)
        pib = self.config.p_internal_batch
        bat = math.ceil(input_ids_flat.shape[0]/pib)
        if (bat>2):
            print("B: {} N: {} b: {} bat: {}".format(B, N, b, bat))

        for j in range(0, bat):
            bs, be = j*pib, min(j*pib+pib, B*b)
#             left_inp, left_mask, right_inp, right_mask = self.get_adj_ids_masks(input_ids_flat[bs:be], inp_mask_flat[bs:be])
            all_hidden_states, _ = self.bert_layer(input_ids_flat[bs:be], attention_mask=inp_mask_flat[bs:be])[-2:] #pib(Bb)*512*d
#             all_left_hidden_states, _ = self.bert_layer(left_inp, attention_mask=left_mask)[-2:]
#             all_right_hidden_states, _ = self.bert_layer(right_inp, attention_mask=right_mask)[-2:]
            for batch in range(0, B):
                for event in range(0, N):
                    b_idx = int(locs[batch, event][0]/512)
                    emb_idx = batch * b + b_idx
                    idx = int(emb_idx/pib)
                    if(idx==j):
                        r_emb_idx = int(emb_idx%pib)
                        
                        s, e = locs[batch, event][0] - b_idx*512, locs[batch, event][1] - b_idx*512
                        trigger_emb = self.get_constituent_emb(all_hidden_states[11][r_emb_idx, s:e, :], mode, "TRIGGER")
                        
                        if(trigger_emb is not None):
                             emb_output[batch, event][0:self.tres] = trigger_emb

                        if(not self.config.event_only_trigger):
                            if(locs[batch, event][2] != -1 and locs[batch, event][3] != -1):
                                zs, ze = locs[batch, event][2] - b_idx*512, locs[batch, event][3] - b_idx*512
                                arg0_emb = self.get_constituent_emb(all_hidden_states[11][r_emb_idx, zs:ze, :], mode, "ARG0")
                                if(arg0_emb is not None):
                                    emb_output[batch, event][self.tres: self.tres+self.a0es] = arg0_emb

                            if(locs[batch, event][4] != -1 and locs[batch, event][5] != -1):
                                os, oe = locs[batch, event][4] - b_idx*512, locs[batch, event][5] - b_idx*512
                                arg1_emb = self.get_constituent_emb(all_hidden_states[11][r_emb_idx, os:oe, :], mode, "ARG1")
                                if(arg1_emb is not None):
                                    idxs = self.tres + self.a0es
                                    idxe = idxs  + self.a1es
                                    emb_output[batch, event][idxs:idxe] = arg1_emb

                            if(locs[batch, event][6] != -1 and locs[batch, event][7] != -1):
                                ls, le = locs[batch, event][6] - b_idx*512, locs[batch, event][7] - b_idx*512
                                loc_emb = self.get_constituent_emb(all_hidden_states[11][r_emb_idx, ls:le, :], mode, "LOC")
                                if(loc_emb is not None):
                                    idxs = self.tres + self.a0es + self.a1es
                                    idxe = idxs + self.les
                                    emb_output[batch, event][idxs: idxe] = loc_emb

                            if(locs[batch, event][8] != -1 and locs[batch, event][9] != -1):
                                ts, te = locs[batch, event][8] - b_idx*512, locs[batch, event][9] - b_idx*512
                                tmp_emb = self.get_constituent_emb(all_hidden_states[11][r_emb_idx, ts:te, :], mode, "TMP")
                                if(tmp_emb is not None):
                                    idxs = self.tres + self.a0es + self.a1es + self.les
                                    idxe = idxs + self.tes
                                    emb_output[batch, event][idxs:idxe] = tmp_emb

        if(finput_ids is not None):
            frame_emb = self.bert_layer(finput_ids.view(-1,16), attention_mask=finput_masks.view(-1,16))[1] #pib(Bb)*512*d
            emb_output = torch.cat((emb_output, frame_emb.view(B,N,768)), dim=2) 
        return emb_output

    '''
        Input: B*b*512, B*b*512, B*N*2
        Output: B*N
    '''
    def forward(self, input_ids, input_mask, locs, doc_features = False, finput_ids=None, finput_masks=None):
        if(doc_features):
            x = self.get_event_embeddings(input_ids, input_mask, locs, None, finput_masks) #(B*b*512,B*b*512,B*N*2)->(B*N*768)
        else:
            x = self.get_event_embeddings(input_ids, input_mask, locs, finput_ids, finput_masks) # (B*b*512, B*b*512, B*N*2) -> (B*N*1536)
        return x    
