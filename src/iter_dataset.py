import torch
import json
from torch.utils.data import Dataset, IterableDataset, DataLoader, RandomSampler, SequentialSampler 
from itertools import cycle, islice
from data_processing_utils import get_events_from_doc, get_event_features_from_doc, get_tokenized_input
from transformers import BertTokenizer, BartTokenizer

class MyIterableDataset(IterableDataset):
    def __init__(self, file_path, folder_path, file_paths, length, lm_model):
        self.file_path = file_path
        self.folder_path = folder_path
        self.file_paths = file_paths
        self.length = length
        self.lm_model = lm_model
        if(lm_model=="BERT"):
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        else:
            self.tokenizer = BartTokenizer.from_pretrained('bart-large', do_lower_case=True)
            

    def parse_file(self, file_path, worker_id):
        line_num = 0
        with open(file_path, 'r') as f:
            for line in f:
                line_num = line_num + 1
                d = json.loads(line)
                event_sur_list, event_loc_list, event_lab_list, event_frame_list = get_events_from_doc(d)
                event_feature_list = get_event_features_from_doc(d)
                if(len(event_sur_list)>200 or len(d['bodyText'].split()) > 1999):
                    continue
                input_ids, attention_masks, frame_input_ids, frame_attention_masks = get_tokenized_input(self.tokenizer, 
                                                                                                 line_num, 
                                                                                                 d['bodyText'], 
                                                                                                 event_sur_list, 
                                                                                                 event_frame_list)
                input_ids = torch.tensor(input_ids) # b*512
                attention_masks = torch.tensor(attention_masks) # b*512
                frame_input_ids = torch.tensor(frame_input_ids) # n*16
                frame_attention_masks = torch.tensor(frame_attention_masks) # n*16
                
                labels = torch.tensor(event_lab_list) # n*1
                locs = torch.tensor(event_loc_list) # n*10
                features = torch.tensor(event_feature_list) # n*x
                
                yield input_ids, attention_masks, labels, locs, features, frame_input_ids, frame_attention_masks, int(d['docno'])

    def get_stream(self, file_path, worker_id):
        return self.parse_file(file_path, worker_id)

    def __iter__(self):
        worker_id = -2
        fpath = self.file_path
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            fpath = self.folder_path + self.file_paths[worker_id]
        return self.get_stream(fpath, worker_id)
    
    def __len__(self):
        return self.length
