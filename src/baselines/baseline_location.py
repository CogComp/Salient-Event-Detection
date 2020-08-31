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
parser.add_argument('gpus', help='delimited list input', type=str)
parser.add_argument('out_file', help='Model Absolute Path', type=str)

# Data Related
parser.add_argument('-bs_train', '--train_batch_size', type=int, default=16, help='Training Batch Size across GPUs')
parser.add_argument('-bs_val', '--val_batch_size', type=int, default=16, help='Validation Batch Size across GPUs')
parser.add_argument('-bs_test', '--test_batch_size', type=int, default=16, help='Testing Batch Size across GPUs')
parser.add_argument('-steps_train', '--train_steps', type=int, default=1476, help='Number of training steps')
parser.add_argument('-steps_val', '--val_steps', type=int, default=380, help='Number of validation steps')
parser.add_argument('-steps_test', '--test_steps', type=int, default=192, help='Number of testing steps')
parser.add_argument('-w_train', '--num_workers_train', type=int, default=2, help='Number of workers to load training data')
parser.add_argument('-w_val', '--num_workers_val', type=int, default=2, help='Number of workers to load validation data')
parser.add_argument('-w_test', '--num_workers_test', type=int, default=2, help='Number of workers to load test data')
parser.add_argument('--train_data_path', default='/shared/djjindal/ASD/data/full/train_parts_v3_srl/', help='Train Data Path')
parser.add_argument('--val_data_path', default='/shared/djjindal/ASD/data/full/val_parts_v3_srl/', help='Validation Data Path')
parser.add_argument('--test_data_path', default='/shared/djjindal/ASD/data/full/test_parts_v3_srl/', help='Test Data Path')

# Parse Arguments
args = parser.parse_args()
print("\nArguments...\n")
print(args)
numdocs = 51610
def predict(file_name, out_filename, fname, device, bs=1):
    eval_obj=Eval()
    outfile = open(out_filename, 'w')
    iterable_predict_dataset = MyIterableDataset("", file_name, [fname], numdocs, "BERT")#46114 #2243#48360
    loader =  DataLoader(iterable_predict_dataset, 
                        batch_size = bs,
                        collate_fn = my_collate,
                        pin_memory = True,
                        num_workers = 1,
                        drop_last = True,
                       )
    k = 0
    out_doc_json  = dict()
    for batch in tqdm(islice(loader, numdocs)):
        b_input_ids, b_input_mask, b_labels, b_locs, b_feat, b_finput_ids, b_finput_masks, b_lengths, b_doc = batch
        input_ids, input_mask, labels, locs, feat, finput_ids, finput_masks, lengths = b_input_ids.to(device), b_input_mask.to(device), b_labels.to(device), b_locs.to(device), b_feat.to(device), b_finput_ids.to(device), b_finput_masks.to(device), b_lengths.to(device)

        for i in range(0, lengths.shape[0]):
            locFeat = b_feat[i][:,0]
            rel_logits = 10 - locFeat
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
predict('/shared/djjindal/CEE-data/test_set/', '/shared/djjindal/CEE-data/predictions/' + args.out_file, 'full_test_data.json', torch.device("cuda:"+args.gpus))
end = datetime.now()
print((end-start).total_seconds())
