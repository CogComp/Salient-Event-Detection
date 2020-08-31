import json
import numpy as np
import sys
from eval.metrics import Eval

eval_obj=Eval()
inp_pred = open(sys.argv[1], 'r')
outp_pred = open(sys.argv[2], 'w')
baseline = inp_pred.readline()
while(baseline):
    basedoc = json.loads(baseline)
    key = basedoc['docno']
    try:
        path = '/shared/djjindal/CEE-data/test_set/docs/'+str(key)+'.txt'
        doc = json.loads(open(path, 'r').read())

        scoree = [f[1] for f in basedoc['bodyText']['predict']]
        namee = [e['surface'] for e in doc['event']['bodyText']]
        sali = [e['salience'] for e in doc['event']['bodyText']]
        nr_at_k = eval_obj.int_norm_r_at_k(scoree, sali)
        score, sal, name, up_at_k = eval_obj.p_at_uniq_k(namee, scoree, sali)
        unr_at_k = eval_obj.int_norm_r_at_k(score, sal)
        basedoc['eval']['up@01'] = up_at_k['p@01']
        basedoc['eval']['up@05'] = up_at_k['p@05']
        basedoc['eval']['up@10'] = up_at_k['p@10']
        basedoc['eval']['up@20'] = up_at_k['p@20']
        basedoc['eval']['ur@01'] = unr_at_k['nr@01']
        basedoc['eval']['ur@05'] = unr_at_k['nr@05']
        basedoc['eval']['ur@10'] = unr_at_k['nr@10']
        basedoc['eval']['ur@20'] = unr_at_k['nr@20']
        basedoc['eval']['nr@01'] = nr_at_k['nr@01']
        basedoc['eval']['nr@05'] = nr_at_k['nr@05']
        basedoc['eval']['nr@10'] = nr_at_k['nr@10']
        basedoc['eval']['nr@20'] = nr_at_k['nr@20']
        basedoc['events'] = name
        basedoc['salience'] = sal
        basedoc['score'] = score
        outp_pred.write(json.dumps(basedoc))
        outp_pred.write("\n")
    except:
        print("ERROR", key)
    baseline = inp_pred.readline()
outp_pred.close()
inp_pred.close()
