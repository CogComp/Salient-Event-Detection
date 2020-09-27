import json
import numpy as np
from eval.metrics import Eval
import sys
import argparse

parser = argparse.ArgumentParser(description='Augment predictions with unique precision/recall metrics')
parser.add_argument('-inp', type=str, default='predictions/out.json', help='Predictions file, an output of CEE-predict')
parser.add_argument('-out', type=str, default='predictions/out_uniq.json', help='Output file')
parser.add_argument('-input_doc_dir', type=str, default='CEE-data/test_set/docs/', help='Directory containing input documents')

# Parse Arguments
args = parser.parse_args()

eval_obj = Eval()
inp_pred = open(args.inp, 'r')
out_dict = dict()
pred_dict = json.load(inp_pred)

for key in pred_dict:
    try:
        path = args.input_doc_dir + str(key)+'.txt'
        doc = json.loads(open(path, 'r').read())
        outp = pred_dict[key]
        scoree = outp['output']
        namee = [e['surface'] for e in doc['event']['bodyText']]
        sali = [e['salience'] for e in doc['event']['bodyText']]
        score, sal, name, up_at_k = eval_obj.p_at_uniq_k(namee, scoree, sali)
        unr_at_k = eval_obj.int_norm_r_at_k(score, sal)
        outp['up@01'] = up_at_k['p@01']
        outp['up@05'] = up_at_k['p@05']
        outp['up@10'] = up_at_k['p@10']
        outp['up@20'] = up_at_k['p@20']
        outp['ur@01'] = unr_at_k['nr@01']
        outp['ur@05'] = unr_at_k['nr@05']
        outp['ur@10'] = unr_at_k['nr@10']
        outp['ur@20'] = unr_at_k['nr@20']
        outp['events'] = name
        outp['salience'] = sal
        outp['score'] = score
        out_dict[key] = outp
    except:
        print("ERROR", key)

        outp_pred = open(args.out, 'w')
        outp_pred.write(json.dumps(out_dict))
        outp_pred.write("\n")
        outp_pred.close()
        inp_pred.close()
