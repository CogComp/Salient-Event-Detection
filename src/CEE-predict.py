import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')
import multiprocessing
MAX_PROCESSES = multiprocessing.cpu_count()
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from eval.metrics import Eval
from lightning_module import *
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Training IIE Model..')

# Mandatory
parser.add_argument('model_type', help='Model Type [BERT-FC, BERT-SVA, BERT-DMN]')
parser.add_argument('gpus', help='delimited list input', type=str)
parser.add_argument('model_to_test', help='Model Absolute Path', type=str)
parser.add_argument('out_file', help='Model Absolute Path', type=str)

# Data Related
parser.add_argument('-bs_test', '--test_batch_size', type=int, default=16, help='Testing Batch Size across GPUs')
parser.add_argument('-steps_test', '--test_steps', type=int, default=192, help='Number of testing steps')
parser.add_argument('-w_test', '--num_workers_test', type=int, default=2, help='Number of workers to load test data')
parser.add_argument('--test_dir', default='CEE-data/test_set/', help='Test Data Dir Path')
parser.add_argument('--test_file', default='full_test_data.json', help='Test File Name')

# Model: Event Representation
parser.add_argument('--freeze_bert', help='Freeze Bert?', action='store_true')
parser.add_argument('-event_emb_size', '--event_dim', type=int, default=768, help='Event Embedding Size')
parser.add_argument('-event_only_trigger', help='Only contextualized trigger word?', action='store_true')
parser.add_argument('--lm_model_type', default="BERT", help='Language Model Type, only supporting [BERT]')
parser.add_argument('-bert_batch', '--p_internal_batch', type=int, default=16, help='BERT batch per GPU')
parser.add_argument('-bert_max_seq', '--p_max_seq', type=int, default=512, help='Max sequence length BERT')
parser.add_argument('-back_features', help='Include background features?', action='store_true')
parser.add_argument('-frame', help='Include Frame Name features?', action='store_true')
parser.add_argument('-sl', help='Include SL feature?', action='store_true')
parser.add_argument('-unem', help='Include EM feature?', action='store_true')
parser.add_argument('-nem', help='Include Normalized EM feature?', action='store_true')
parser.add_argument('-af', help='Include Arg Frequency?', action='store_true')
parser.add_argument('-an', help='Include Arg Named Entity?', action='store_true')
parser.add_argument('-ps', help='Include Parent Score features?', action='store_true')
parser.add_argument('-corref', help='Include Corref Score?', action='store_true')
parser.add_argument('-ch', help='Include Corref Score?', action='store_true')

# Model: SVA
parser.add_argument('-sva_dim', '--p_key_query_dim', type=int, default=96, help='Key/Query number of dimensions')

# Model: DMN
parser.add_argument('-dmn_passes', '--num_passes', type=int, default=2, help='Number of episodes of DMN')
parser.add_argument('-dmn_hs', '--hs', type=int, default=256, help='Hidden Layer Size DMN')

# Parse Arguments
args = parser.parse_args()
print("\nArguments...\n")
print(args)
numdocs = 51610
def predict(input_dir, input_fname, out_fname, model, device, bs=1):
    eval_obj=Eval()
    outfile = open(out_fname, 'w')
    iterable_predict_dataset = MyIterableDataset("", input_dir, [input_fname], numdocs, "BERT")#46114 #2243#48360
    loader =  DataLoader(iterable_predict_dataset, 
                        batch_size = bs,
                        collate_fn = my_collate,
                        pin_memory = True,
                        num_workers = 1,
                        drop_last = True,
                       )
    model.freeze()
    k = 0
    out_doc_json  = dict()
    for batch in tqdm(islice(loader, numdocs)):
        b_input_ids, b_input_mask, b_labels, b_locs, b_feat, b_finput_ids, b_finput_masks, b_lengths, b_doc = batch
        input_ids, input_mask, labels, locs, feat, finput_ids, finput_masks, lengths = b_input_ids.to(device), b_input_mask.to(device), b_labels.to(device), b_locs.to(device), b_feat.to(device), b_finput_ids.to(device), b_finput_masks.to(device), b_lengths.to(device)
        logits,_ = model(input_ids, input_mask, finput_ids, finput_masks, locs, feat)

        for i in range(0, lengths.shape[0]):
            rel_logits = logits[i, :b_lengths[i]]
            rel_labels = b_labels[i, :b_lengths[i]]
            b_p_at_k = eval_obj.p_at_k(rel_logits, rel_labels)
            b_r_at_k = eval_obj.r_at_k(rel_logits, rel_labels)
            b_nr_at_k = eval_obj.norm_r_at_k(rel_logits, rel_labels)
            eval_dict = dict()
            eval_dict['p@01'] = b_p_at_k['p@01']
            eval_dict['p@05'] = b_p_at_k['p@05']
            eval_dict['p@10'] = b_p_at_k['p@10']
            eval_dict['p@20'] = b_p_at_k['p@20']
            eval_dict['r@01'] = b_r_at_k['r@01']
            eval_dict['r@05'] = b_r_at_k['r@05']
            eval_dict['r@10'] = b_r_at_k['r@10']
            eval_dict['r@20'] = b_r_at_k['r@20']
            eval_dict['nr@01'] = b_nr_at_k['nr@01']
            eval_dict['nr@05'] = b_nr_at_k['nr@05']
            eval_dict['nr@10'] = b_nr_at_k['nr@10']
            eval_dict['nr@20'] = b_nr_at_k['nr@20']
            eval_dict['output'] = rel_logits.cpu().numpy().tolist()
            out_doc_json[b_doc[i].item()] = eval_dict
    outfile.write(json.dumps(out_doc_json)) 
    outfile.write("\n")     
    outfile.close()

start = datetime.now()
print("\nLoading Model...\n")
model = LitNet.load_from_checkpoint(args.model_to_test, eval_obj=Eval(), args=args, mode=args.model_type)
dev = torch.device("cuda:"+args.gpus)
predict(args.test_dir, args.test_file , args.out_file, model.to(dev), dev)
end = datetime.now()
print((end-start).total_seconds())
