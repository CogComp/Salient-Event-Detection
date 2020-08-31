import json
import numpy as np
import torch
import math
import re
from collections import OrderedDict
import time
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from random import sample
from traitlets import (
    List,
    Int,
    Unicode,
)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import BertModel, BartModel

import pytorch_lightning as pl
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

import warnings
warnings.filterwarnings('ignore')

from eval.metrics import Eval
from utils import *
from data_processing_utils import *
from iter_dataset import *
from models.bert_fc import BERTFcModel
from models.bert_sva import SelfVotingAttnModel
from models.bert_dmn import MHA_Model

class LitNet(pl.LightningModule):
    def __init__(self, eval_obj, args, mode):
        super().__init__()
        self.config = args
        self.eval_obj = eval_obj
        self.bce_loss = nn.BCEWithLogitsLoss(reduction = 'none')
        self.mode = mode        
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased', 
                                                output_hidden_states = True,
                                               output_attentions = True)
        
        if(mode == "BERT-FC"):
            self.iiemodel = BERTFcModel(self.bert_layer, self.config)
        if(mode == "BERT-SVA"):
            self.iiemodel = SelfVotingAttnModel(self.bert_layer, self.config)
        if(mode == "BERT-DMN"):
            self.iiemodel = MHA_Model(self.bert_layer, self.config)
        
        '''
        Input: B*b*512, B*b*512, B*N*2
        Output: B*N
        '''
    def forward(self, input_ids, input_mask, b_finput_ids, b_finput_masks, locs, feats): #B*N*768
        attn = None
        if(self.mode == "BERT-SVA"):
            y, attn = self.iiemodel(input_ids, input_mask, b_finput_ids, b_finput_masks, locs, feats)
        else:
            y = self.iiemodel(input_ids, input_mask, b_finput_ids, b_finput_masks, locs, feats)
        return y, attn
    
    
    def configure_optimizers(self):
        #Freeze bert layers
        if self.config.freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
        optimizer = optim.Adam([p for p in self.parameters() if p.requires_grad], 
                               lr = self.config.lr,
                               eps = 1e-08)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 1, factor = 0.5, verbose = True)
        return [optimizer], [scheduler]
        
        '''
        Input: B*N, B*N
        Output: Float
        '''
    def calculate_loss_old(self, output, target, b_lenghts):
        b = output.shape[0]
        loss_mat = self.bce_loss(output, target.float())#B*N
        loss = torch.tensor(0, dtype = torch.float, requires_grad = True).to(output.device)
        c = 0
        for i in range(0, b):
            actual_len = b_lenghts[i]
            #Sampling
            positives = (target[i, :actual_len] > 0).nonzero()
            if(len(positives) > 0):
                np_list = (target[i, :actual_len] < 1).nonzero().tolist()
                negatives = torch.tensor(sample(np_list, min(len(np_list), len(positives)))).long().to(output.device)
                sampl = torch.cat([positives, negatives], dim = 0)
                # Loss Summation
                loss = loss + torch.sum(loss_mat[i, sampl])/sampl.shape[0]
                c = c + 1
        if(c > 0):
            loss = loss/c
        return loss    

    def get_all_metrics(self, logits, b_lengths, b_labels):
        count = b_labels.shape[0]
        avg_acc, avg_p, avg_r = 0.0, 0.0, 0.0
        avg_p1, avg_p5, avg_p10, avg_p20 = 0.0, 0.0, 0.0, 0.0
        avg_r1, avg_r5, avg_r10, avg_r20 = 0.0, 0.0, 0.0, 0.0
        avg_nr1, avg_nr5, avg_nr10, avg_nr20 = 0.0, 0.0, 0.0, 0.0
        for i in range(0, count):
            rel_logits = logits[i, :b_lengths[i]]
            rel_labels = b_labels[i, :b_lengths[i]]
            avg_acc += self.eval_obj.accuracy(rel_logits, rel_labels)['accuracy']
            avg_p += self.eval_obj.precision(rel_logits, rel_labels)['precision']
            avg_r += self.eval_obj.recall(rel_logits, rel_labels)['recall']
            
            b_p_at_k = self.eval_obj.p_at_k(rel_logits, rel_labels)
            avg_p1 += b_p_at_k['p@01']
            avg_p5 += b_p_at_k['p@05']
            avg_p10 += b_p_at_k['p@10']
            avg_p20 += b_p_at_k['p@20']
            
            b_r_at_k = self.eval_obj.r_at_k(rel_logits, rel_labels)
            avg_r1 += b_r_at_k['r@01']
            avg_r5 += b_r_at_k['r@05']
            avg_r10 += b_r_at_k['r@10']
            avg_r20 += b_r_at_k['r@20']
            
            b_nr_at_k = self.eval_obj.norm_r_at_k(rel_logits, rel_labels)
            avg_nr1 += b_nr_at_k['nr@01']
            avg_nr5 += b_nr_at_k['nr@05']
            avg_nr10 += b_nr_at_k['nr@10']
            avg_nr20 += b_nr_at_k['nr@20']
        
        avg_acc, avg_p, avg_r = avg_acc/count, avg_p/count, avg_r/count
        avg_p1, avg_p5, avg_p10, avg_p20 = avg_p1/count, avg_p5/count, avg_p10/count, avg_p20/count
        avg_r1, avg_r5, avg_r10, avg_r20 = avg_r1/count, avg_r5/count, avg_r10/count, avg_r20/count
        avg_nr1, avg_nr5, avg_nr10, avg_nr20 = avg_nr1/count, avg_nr5/count, avg_nr10/count, avg_nr20/count        
        
        avg_acc = torch.tensor(avg_acc).to(logits.device)
        avg_p = torch.tensor(avg_p).to(logits.device)
        avg_r = torch.tensor(avg_r).to(logits.device)
        avg_p1 = torch.tensor(avg_p1).to(logits.device)
        avg_p5 = torch.tensor(avg_p5).to(logits.device)
        avg_p10 = torch.tensor(avg_p10).to(logits.device)
        avg_p20 = torch.tensor(avg_p20).to(logits.device)
        avg_r1 = torch.tensor(avg_r1).to(logits.device)
        avg_r5 = torch.tensor(avg_r5).to(logits.device)            
        avg_r10 = torch.tensor(avg_r10).to(logits.device)
        avg_r20 = torch.tensor(avg_r20).to(logits.device)
        avg_nr1 = torch.tensor(avg_nr1).to(logits.device)
        avg_nr5 = torch.tensor(avg_nr5).to(logits.device)            
        avg_nr10 = torch.tensor(avg_nr10).to(logits.device)
        avg_nr20 = torch.tensor(avg_nr20).to(logits.device)
        
        metrics = {'avg_acc':avg_acc, 'avg_p': avg_p, 'avg_r': avg_r,
                   'avg_p1': avg_p1, 'avg_p5': avg_p5, 'avg_p10': avg_p10, 'avg_p20': avg_p20, 
                   'avg_r1': avg_r1, 'avg_r5': avg_r5, 'avg_r10': avg_r10, 'avg_r20': avg_r20,
                   'avg_nr1': avg_nr1, 'avg_nr5': avg_nr5, 'avg_nr10': avg_nr10, 'avg_nr20': avg_nr20}
        return metrics
    
    def training_step(self, batch, batch_idx):
        b_input_ids, b_input_mask, b_labels, b_locs, b_feats, b_finput_ids, b_finput_masks, b_lengths, _ = batch
        logits,_ = self(b_input_ids, b_input_mask, b_finput_ids, b_finput_masks, b_locs, b_feats)
        loss = self.calculate_loss_old(logits, b_labels, b_lengths)
        return {"loss": loss}
    
    '''
        Input: Metrics per GPU (Batch/GPUs)
        Output: Metrics per Batch
    '''
    def training_step_end(self, losses):
        loss = torch.mean(losses['loss']) # Input: gpus, Output: Float
        tensorboard_logs = {'train_loss': loss}
        return {'loss':loss, 'log': tensorboard_logs}
        
    def validation_step(self, batch, batch_idx):
        b_input_ids, b_input_mask, b_labels, b_locs, b_feats, b_finput_ids, b_finput_masks, b_lengths, _ = batch
        logits, _ = self(b_input_ids, b_input_mask, b_finput_ids, b_finput_masks, b_locs, b_feats)        
        loss = self.calculate_loss_old(logits, b_labels, b_lengths)
        metrics = self.get_all_metrics(logits, b_lengths, b_labels)
        metrics["val_loss"] = loss
        return metrics
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['avg_acc'] for x in outputs]).mean()
        avg_p = torch.stack([x['avg_p'] for x in outputs]).mean()
        avg_r = torch.stack([x['avg_r'] for x in outputs]).mean()
        avg_p1 = torch.stack([x['avg_p1'] for x in outputs]).mean()
        avg_p5 = torch.stack([x['avg_p5'] for x in outputs]).mean()
        avg_p10 = torch.stack([x['avg_p10'] for x in outputs]).mean()
        avg_p20 = torch.stack([x['avg_p20'] for x in outputs]).mean()
        avg_r1 = torch.stack([x['avg_r1'] for x in outputs]).mean()
        avg_r5 = torch.stack([x['avg_r5'] for x in outputs]).mean()
        avg_r10 = torch.stack([x['avg_r10'] for x in outputs]).mean()
        avg_r20 = torch.stack([x['avg_r20'] for x in outputs]).mean()
        avg_nr1 = torch.stack([x['avg_nr1'] for x in outputs]).mean()
        avg_nr5 = torch.stack([x['avg_nr5'] for x in outputs]).mean()
        avg_nr10 = torch.stack([x['avg_nr10'] for x in outputs]).mean()
        avg_nr20 = torch.stack([x['avg_nr20'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 
                            'val_acc':avg_acc, 'val_p': avg_p, 'val_r': avg_r,
                            'val_p1': avg_p1, 'val_p5': avg_p5, 'val_p10': avg_p10, 'val_p20': avg_p20,
                            'val_r1': avg_r1, 'val_r5': avg_r5, 'val_r10': avg_r10, 'val_r20': avg_r20,
                            'val_nr1': avg_nr1, 'val_nr5': avg_nr5, 'val_nr10': avg_nr10, 'val_nr20': avg_nr20}
        output = OrderedDict({
            "val_loss": avg_loss,
            "progress_bar": tensorboard_logs,
            "log": tensorboard_logs
            })
        return output
    
#     {"bodyText": {"predict": [[339, -7.522473335266113], [544, -7.93048095703125], [1253, -7.240342617034912]]}, "docno": "1415110", "eval": {"auc": 0.5, "r@01": 0.0, "r@20": 1.0, "r@05": 1.0, "r@10": 1.0, "p@05": 0.2, "p@20": 0.05, "p@01": 0.0, "p@10": 0.1}}
    def get_all_metrics_test(self, logits, b_lengths, b_labels, b_doc):
        count = b_labels.shape[0]
        avg_acc, avg_p, avg_r = 0.0, 0.0, 0.0
        avg_p1, avg_p5, avg_p10, avg_p20 = 0.0, 0.0, 0.0, 0.0
        avg_r1, avg_r5, avg_r10, avg_r20 = 0.0, 0.0, 0.0, 0.0
        avg_nr1, avg_nr5, avg_nr10, avg_nr20 = 0.0, 0.0, 0.0, 0.0
        out_json = []
        for i in range(0, count):
            rel_logits = logits[i, :b_lengths[i]]
            rel_labels = b_labels[i, :b_lengths[i]]
            avg_acc += self.eval_obj.accuracy(rel_logits, rel_labels)['accuracy']
            avg_p += self.eval_obj.precision(rel_logits, rel_labels)['precision']
            avg_r += self.eval_obj.recall(rel_logits, rel_labels)['recall']
            
            b_p_at_k = self.eval_obj.p_at_k(rel_logits, rel_labels)
            avg_p1 += b_p_at_k['p@01']
            avg_p5 += b_p_at_k['p@05']
            avg_p10 += b_p_at_k['p@10']
            avg_p20 += b_p_at_k['p@20']
            
            b_r_at_k = self.eval_obj.r_at_k(rel_logits, rel_labels)
            avg_r1 += b_r_at_k['r@01']
            avg_r5 += b_r_at_k['r@05']
            avg_r10 += b_r_at_k['r@10']
            avg_r20 += b_r_at_k['r@20']
            
            eval_dict = dict()
            eval_dict['p@01'] = b_p_at_k['p@01']
            eval_dict['p@05'] = b_p_at_k['p@05']
            eval_dict['p@10'] = b_p_at_k['p@10']
            eval_dict['p@20'] = b_p_at_k['p@20']
            eval_dict['r@01'] = b_p_at_k['r@01']
            eval_dict['r@05'] = b_p_at_k['r@05']
            eval_dict['r@10'] = b_p_at_k['r@10']
            eval_dict['r@20'] = b_p_at_k['r@20']
            
            out_doc_json = dict()
            out_doc_json['docno'] = b_doc[i]
            out_doc_json['eval'] = eval_dict
            out_json.append(out_doc_json)
            
            
            b_nr_at_k = self.eval_obj.norm_r_at_k(rel_logits, rel_labels)
            avg_nr1 += b_nr_at_k['nr@01']
            avg_nr5 += b_nr_at_k['nr@05']
            avg_nr10 += b_nr_at_k['nr@10']
            avg_nr20 += b_nr_at_k['nr@20']
        
        avg_acc, avg_p, avg_r = avg_acc/count, avg_p/count, avg_r/count
        avg_p1, avg_p5, avg_p10, avg_p20 = avg_p1/count, avg_p5/count, avg_p10/count, avg_p20/count
        avg_r1, avg_r5, avg_r10, avg_r20 = avg_r1/count, avg_r5/count, avg_r10/count, avg_r20/count
        avg_nr1, avg_nr5, avg_nr10, avg_nr20 = avg_nr1/count, avg_nr5/count, avg_nr10/count, avg_nr20/count        
        
        avg_acc = torch.tensor(avg_acc).to(logits.device)
        avg_p = torch.tensor(avg_p).to(logits.device)
        avg_r = torch.tensor(avg_r).to(logits.device)
        avg_p1 = torch.tensor(avg_p1).to(logits.device)
        avg_p5 = torch.tensor(avg_p5).to(logits.device)
        avg_p10 = torch.tensor(avg_p10).to(logits.device)
        avg_p20 = torch.tensor(avg_p20).to(logits.device)
        avg_r1 = torch.tensor(avg_r1).to(logits.device)
        avg_r5 = torch.tensor(avg_r5).to(logits.device)            
        avg_r10 = torch.tensor(avg_r10).to(logits.device)
        avg_r20 = torch.tensor(avg_r20).to(logits.device)
        avg_nr1 = torch.tensor(avg_nr1).to(logits.device)
        avg_nr5 = torch.tensor(avg_nr5).to(logits.device)            
        avg_nr10 = torch.tensor(avg_nr10).to(logits.device)
        avg_nr20 = torch.tensor(avg_nr20).to(logits.device)
        metrics = {'avg_acc':avg_acc, 'avg_p': avg_p, 'avg_r': avg_r,
                   'avg_p1': avg_p1, 'avg_p5': avg_p5, 'avg_p10': avg_p10, 'avg_p20': avg_p20, 
                   'avg_r1': avg_r1, 'avg_r5': avg_r5, 'avg_r10': avg_r10, 'avg_r20': avg_r20,
                   'avg_nr1': avg_nr1, 'avg_nr5': avg_nr5, 'avg_nr10': avg_nr10, 'avg_nr20': avg_nr20}
        return metrics
    
    def test_step(self, batch, batch_idx):
        b_input_ids, b_input_mask, b_labels, b_locs, b_feats, b_finput_ids, b_finput_masks, b_lengths, b_doc = batch
        logits, _ = self(b_input_ids, b_input_mask, b_finput_ids, b_finput_masks, b_locs, b_feats)
        loss = self.calculate_loss_old(logits, b_labels, b_lengths)            
        metrics = self.get_all_metrics(logits, b_lengths, b_labels)
        metrics["val_loss"] = loss
        return metrics
    
    def test_step_backup(self, batch, batch_idx):
        output = self.validation_step(batch, batch_idx)
        return output
    
    def test_epoch_end(self, outputs):
        print("test_epoch_end")
        results = self.validation_epoch_end(outputs)
        # rename some keys
        results['progress_bar'].update({
            'test_loss': results['progress_bar'].pop('val_loss'),
            'test_acc': results['progress_bar'].pop('val_acc'),
            'test_p': results['progress_bar'].pop('val_p'),
            'test_r': results['progress_bar'].pop('val_r'),
            'test_p1': results['progress_bar'].pop('val_p1'),
            'test_p5': results['progress_bar'].pop('val_p5'),
            'test_p10': results['progress_bar'].pop('val_p10'),
            'test_p20': results['progress_bar'].pop('val_p20'),
            'test_r1': results['progress_bar'].pop('val_r1'),
            'test_r5': results['progress_bar'].pop('val_r5'),
            'test_r10': results['progress_bar'].pop('val_r10'),
            'test_r20': results['progress_bar'].pop('val_r20'),
            'test_nr1': results['progress_bar'].pop('val_nr1'),
            'test_nr5': results['progress_bar'].pop('val_nr5'),
            'test_nr10': results['progress_bar'].pop('val_nr10'),
            'test_nr20': results['progress_bar'].pop('val_nr20'),
        })
        results['log'] = results['progress_bar']
        results['test_loss'] = results.pop('val_loss')
        return results
    
    def train_dataloader(self):
        iterable_train_dataset = MyIterableDataset('', self.config.train_data_path,                                  ['xab','xaa','xac','xad','xae','xaf','xag','xah','xai','xaj','xak','xal','xam','xan','xao','xap','xaq','xar','xas','xat','xau','xav','xaw','xax','xay','xaz','xba','xbb','xbc','xbd','xbe','xbf'],
                                                    self.config.train_steps,
                                                    self.config.lm_model_type)
        return DataLoader(iterable_train_dataset, 
                    batch_size = self.config.train_batch_size,
                    collate_fn = my_collate,
                    pin_memory = True,
                    num_workers = self.config.num_workers_train,
                    drop_last = True,
                   )
    
    def val_dataloader(self):
        iterable_val_dataset = MyIterableDataset('', self.config.val_data_path,                                                   ['xaa','xab','xac','xad','xae','xaf','xag','xah','xai','xaj','xak','xal','xam','xan','xao','xap','xaq','xar','xas','xat','xau','xav','xaw','xax','xay','xaz','xba','xbb','xbc','xbd','xbe','xbf'],
                                                    self.config.val_steps,
                                                    self.config.lm_model_type)
        return DataLoader(iterable_val_dataset, 
                    batch_size = self.config.val_batch_size,
                    collate_fn = my_collate,
                    pin_memory = True,
                    num_workers = self.config.num_workers_val,
                    drop_last = True,
                   )

    def test_dataloader(self):
        iterable_test_dataset = MyIterableDataset('', self.config.test_data_path,                                               ['xaa','xab','xac','xad','xae','xaf','xag','xah','xai','xaj','xak','xal','xam','xan','xao','xap','xaq','xar','xas','xat','xau','xav','xaw','xax','xay','xaz','xba','xbb','xbc','xbd','xbe','xbf'],
                                                    self.config.test_steps,
                                                    self.config.lm_model_type)
        return DataLoader(iterable_test_dataset, 
                    batch_size = self.config.test_batch_size,
                    collate_fn = my_collate,
                    pin_memory = True,
                    num_workers = self.config.num_workers_test,
                    drop_last = True,
                   )