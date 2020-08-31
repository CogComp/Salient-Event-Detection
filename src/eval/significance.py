import random
import json

def randomization_test(l_target, l_base, met):
    total_test = 5000
    diff = sum(l_target) / float(len(l_target)) - sum(l_base) / float(len(l_base))
    cnt = 0.0
    for i in range(total_test):
        l_a, l_b = random_swap(l_target, l_base)
        this_diff = sum(l_a) / float(len(l_a)) - sum(l_b) / float(len(l_b))
        if this_diff > diff:
            cnt += 1.0
    p = cnt / float(total_test)
    print("Metrics: ", met, "Our: ", sum(l_target) / float(len(l_target)), "KCM:  ", sum(l_base) / float(len(l_base)), "Diff: ", diff, "P-Value", p)
    return p


def random_swap(l_target, l_base):
    l_a = list(l_target)
    l_b = list(l_base)

    for i in range(len(l_target)):
        if random.randint(0, 1):
            l_a[i], l_b[i] = l_b[i],l_a[i]
    return l_a, l_b

def cal_sig_old_new_format():
    f1 = open('/shared/djjindal/ASD/data/full/preds/m1/base_supfull_U.json', 'r')
    f2 = open('/shared/djjindal/ASD/data/full/preds/m2/base_frame2_supfull_U.json', 'r')
    l_target, l_base = dict(), dict()
    i = 0
    line = f1.readline()
    model_dict = json.loads(f2.readline())
    metric_list = ["p@01", "p@05", "p@10", "p@20", "up@01", "up@05", "up@10", "up@20", "ur@01", "ur@05","ur@10", "ur@20", "nr@01", "nr@05", "nr@10", "nr@20"]
    for met in metric_list:
        l_target[met] = []
        l_base[met] = []
        
    while(line):
        base_json = json.loads(line)
        key = base_json['docno']
        if key in model_dict:
            model_json = model_dict[key]
            for met in metric_list:
                l_target[met].append(model_json[met])
                l_base[met].append(base_json["eval"][met])
            i+=1
        line = f1.readline()
    print("Running randomization_test on ", i, "samples")
    for met in metric_list:
        print(met, sum(l_base[met]) / float(len(l_base[met])))
        print(met, sum(l_target[met]) / float(len(l_target[met])))
        if( sum(l_target[met]) / float(len(l_target[met])) > sum(l_base[met]) / float(len(l_base[met]))):
            pval = randomization_test(l_target[met], l_base[met], met)
        

def cal_sig_new_format():
    f1 = open('/shared/djjindal/ASD/data/full/preds/m3/base_frame_sl_unem_supfull_U.json', 'r')
    f2 = open('/shared/djjindal/ASD/data/full/preds/m4/base_frame_sl_nem_ps_supfull_U.json', 'r')
    l_target, l_base = dict(), dict()
    i = 0
    model1_dict = json.loads(f1.readline())
    model2_dict = json.loads(f2.readline())
    metric_list = ["up@01", "up@05", "up@10","ur@01", "ur@05","ur@10"]
    for met in metric_list:
        l_target[met] = []
        l_base[met] = []
        
    for key in model1_dict:
        try:
            base_json = model1_dict[key]
            model_json = model2_dict[key]
            for met in metric_list:
                l_target[met].append(model_json[met])
                l_base[met].append(base_json[met])
        except:
            print("Error ", key)
    print("Running randomization_test on ", len(l_base[met]), "samples")
    for met in metric_list:
        print(met, sum(l_base[met]) / float(len(l_base[met])))
        print(met, sum(l_target[met]) / float(len(l_target[met])))
        if( sum(l_target[met]) / float(len(l_target[met])) > sum(l_base[met]) / float(len(l_base[met]))):
            pval = randomization_test(l_target[met], l_base[met], met)
        
cal_sig_new_format()        
        
        