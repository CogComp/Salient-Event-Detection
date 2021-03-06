import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

import pytorch_lightning as pl
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from eval.metrics import Eval
from lightning_module import *
from utils import *
import argparse

parser = argparse.ArgumentParser(description='Training IIE Model..') 

# Mandatory
parser.add_argument('model_type', help='Model Type [BERT-FC, BERT-SVA, BERT-DMN]')
parser.add_argument('save_mname', help='Model name, used for saving')
parser.add_argument('gpus', help='delimited list input', type=str)#"265340,268738,270774,270817"

# Data Related
parser.add_argument('-bs_train', '--train_batch_size', type=int, default=12, help='Training Batch Size across GPUs')
parser.add_argument('-bs_val', '--val_batch_size', type=int, default=12, help='Validation Batch Size across GPUs')
parser.add_argument('-bs_test', '--test_batch_size', type=int, default=12, help='Testing Batch Size across GPUs')
parser.add_argument('-steps_train', '--train_steps', type=int, default=3936, help='Number of training steps')
parser.add_argument('-steps_val', '--val_steps', type=int, default=380, help='Number of validation steps')
parser.add_argument('-steps_test', '--test_steps', type=int, default=192, help='Number of testing steps')
parser.add_argument('-w_train', '--num_workers_train', type=int, default=4, help='Number of workers to load training data')
parser.add_argument('-w_val', '--num_workers_val', type=int, default=2, help='Number of workers to load validation data')
parser.add_argument('-w_test', '--num_workers_test', type=int, default=2, help='Number of workers to load test data')
parser.add_argument('--train_data_path', default='CEE-data/train_set/', help='Train Data Path')
parser.add_argument('--val_data_path', default='CEE-data/val_set/', help='Validation Data Path')
parser.add_argument('--test_data_path', default='CEE-data/test_set/', help='Test Data Path')

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

# Model Training
parser.add_argument('-epochs', '--epochs', type=int, default=20, help='Number of Training Epochs')
parser.add_argument("-param_init", default=0, type=float)
parser.add_argument("-param_init_glorot", type=str2bool, nargs='?',const=True,default=True)
parser.add_argument("-optim", default='adam', type=str)
parser.add_argument("-lr", default=1e-5, type=float)
parser.add_argument("-lr_bert", default=5e-4, type=float)
parser.add_argument("-beta1", default= 0.9, type=float)
parser.add_argument("-beta2", default=0.999, type=float)
parser.add_argument("-warmup_steps", default=8000, type=int)
parser.add_argument("-warmup_steps_bert", default=8000, type=int)
parser.add_argument("-max_grad_norm", default=0, type=float)

args = parser.parse_args()
print("\nArguments...\n")
print(args)

print("\nCreating callbacks...\n")
early_stop_callback = EarlyStopping('val_p1', patience=5)
modelfilepath = '../trainedmodels/' + args.model_type + "_" + args.save_mname + '-{epoch:02d}-{val_loss:.2f}'
checkpoint_callback = ModelCheckpoint(filepath=modelfilepath, 
                                      save_top_k=10,
                                      monitor='val_p5')
logger = TensorBoardLogger("../tb_logs", name=args.model_type, version=args.save_mname)


print("\nLoading Model...\n")
model = LitNet(Eval(), args, mode = args.model_type)
trainer = Trainer(gpus = [int(item) for item in args.gpus.split(',')], 
                  min_epochs = 1, 
                  max_epochs = args.epochs, 
                  distributed_backend = 'dp', 
                  profiler = True, 
                  checkpoint_callback = checkpoint_callback, 
                  val_check_interval = 0.4,
                  early_stop_callback = early_stop_callback, 
                  accumulate_grad_batches = 1,
                  logger = logger)

print("\nStarting Training...\n")
trainer.fit(model)

print("\nSaving Model...\n")
trainer.save_checkpoint("../trainedmodels/" + args.model_type + "_" + args.save_mname + ".ckpt")

print("\nTesting...\n")
trainer.test()
