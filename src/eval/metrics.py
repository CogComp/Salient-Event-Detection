class Eval:
    l_depth = [1, 5, 10, 20]

    def p_at_uniq_k(self, event, score, label):
        h_p = {}
        l_d_b = list(zip(score, label, event))
        l_d_b = sorted(l_d_b, key = lambda item: item[2]) #Sort by event
        l_score, l_label, l_event = [], [], []
        old, oldmax = None, -1000
        c = -1
        for s,l,e in l_d_b:
            if(e!=old):
                l_score.append(s)
                l_label.append(l)
                l_event.append(e)
                old = e
                oldmax = s
                c += 1 
            elif(s > oldmax):
                l_score[c] = s
                oldmax = s        

        l_d = list(zip(l_score, l_label))
        l_d = sorted(l_d, key = lambda item: -item[0]) #Sort by score
        correct = 0
        for p in range(max(self.l_depth)):
            label = 0
            if p < len(l_d):
                label = l_d[p][1]
            if label > 0:
                correct += 1
            depth = p + 1
            if depth in self.l_depth:
                res = float(correct) / depth
                h_p['p@%02d' % depth] = res
        return l_score, l_label, l_event, h_p
    
    def p_at_norm_uniq_k(self, event, score, label):
        h_p = {}
        l_d_b = list(zip(score, label, event))
        l_d_b = sorted(l_d_b, key=lambda item: item[2]) #Sort by event
        l_score, l_label, l_event = [], [], []
        old, oldmax = None, -1000
        c = -1
        for s,l,e in l_d_b:
            if(e!=old):
                l_score.append(s)
                l_label.append(l)
                l_event.append(e)
                old = e
                oldmax = s
                c += 1
            elif(s>oldmax):
                l_score[c] = s
                oldmax = s        

        l_d = list(zip(l_score, l_label))
        l_d = sorted(l_d, key = lambda item: -item[0]) #Sort by score
        correct = 0
        total_z = max(1, sum([max(0, min(label, 1)) for label in l_label]))
        for p in range(max(self.l_depth)):
            label = 0
            if p < len(l_d):
                label = l_d[p][1]
            if label > 0:
                correct += 1
            depth = p + 1
            if depth in self.l_depth:
                res = float(correct) / min(total_z, depth)
                h_p['p@%02d' % depth] = res
        return l_score, l_label, l_event, h_p
    
    def p_at_k(self, l_score, l_label):
        h_p = {}
        l_d = list(zip(l_score, l_label))
        l_d = sorted(l_d, key = lambda item: -item[0])
        correct = 0
        for p in range(max(self.l_depth)):
            label = 0
            if p < len(l_d):
                label = l_d[p][1]
            if label > 0:
                correct += 1
            depth = p + 1
            if depth in self.l_depth:
                res = float(correct) / depth
                h_p['p@%02d' % depth] = res
        return h_p

    def r_at_k(self, l_score, l_label):
        h_r = {}
        l_d = list(zip(l_score, l_label))
        l_d = sorted(l_d, key = lambda item: -item[0])
        correct = 0
        total_z = max(1, sum([max(0, min(label, 1).item()) for label in l_label]))
        for p in range(max(self.l_depth)):
            label = 0
            if p < len(l_d):
                label = l_d[p][1]
            if label > 0:
                correct += 1
            depth = p + 1
            if depth in self.l_depth:
                res = float(correct) / total_z
                h_r['r@%02d' % depth] = res
        return h_r

    def norm_r_at_k(self, l_score, l_label):
        h_r = {}
        l_d = list(zip(l_score, l_label))
        l_d = sorted(l_d, key=lambda item: -item[0])
        correct = 0
        total_z = max(1, sum([max(0, min(label, 1).item()) for label in l_label]))
        for p in range(max(self.l_depth)):
            label = 0
            if p < len(l_d):
                label = l_d[p][1]
            if label > 0:
                correct += 1
            depth = p + 1
            if depth in self.l_depth:
                res = float(correct) / min(total_z, depth)
                h_r['nr@%02d' % depth] = res
        return h_r
    
    def int_norm_r_at_k(self, l_score, l_label):
        h_r = {}
        l_d = list(zip(l_score, l_label))
        l_d = sorted(l_d, key=lambda item: -item[0])
        correct = 0
        total_z = max(1, sum([max(0, min(label, 1)) for label in l_label]))
        for p in range(max(self.l_depth)):
            label = 0
            if p < len(l_d):
                label = l_d[p][1]
            if label > 0:
                correct += 1
            depth = p + 1
            if depth in self.l_depth:
                res = float(correct) / min(total_z, depth)
                h_r['nr@%02d' % depth] = res
        return h_r
   
    def precision(self, l_score, l_label):
        z = 0
        c = 0
        for score, label in zip(l_score, l_label):
            if score > 0:
                z += 1
                if label > 0:
                    c += 1
        return {'precision': float(c) / max(z, 1.0)}

    def recall(self, l_score, l_label):
        z = 0
        c = 0
        for score, label in zip(l_score, l_label):
            if label > 0:
                z += 1
                if score > 0:
                    c += 1
        return {'recall': float(c) / max(z, 1.0)}

    def accuracy(self, l_score, l_label):
        c = 0
        for score, label in zip(l_score, l_label):
            if label > 0:
                if score > 0:
                    c += 1
        z = len(l_score)
        return {'accuracy': float(c) / max(z, 1.0)}

    def auc(self, l_score, l_label):
        l_label = [max(0, item) for item in l_label]  # binary
        l_label = [min(1, item) for item in l_label]
        if min(l_label) == 1:
            auc_score = 1
        elif max(l_label) == 0:
            auc_score = 0
        else:
            auc_score = roc_auc_score(l_label, l_score)
        return {'auc': auc_score}
