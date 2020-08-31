import torch
from transformers import BertTokenizer, get_linear_schedule_with_warmup
import math

def get_loc(event):
    loc = [-1]*10
    loc[0], loc[1] = event['loc'][0], event['loc'][1] 
    try:
        args = event['arg_loc']
        if('ARG0' in args):
            loc[2], loc[3] = args['ARG0'][0], args['ARG0'][1]
        if('ARG1' in args):
            loc[4], loc[5] = args['ARG1'][0], args['ARG1'][1]
        if('LOC' in args):
            loc[6], loc[7] = args['LOC'][0], args['LOC'][1]
        if('TMP' in args):
            loc[8], loc[9] = args['TMP'][0], args['TMP'][1]
        return loc
    except KeyError:
        return loc
    
def get_event_features_from_doc(doc):
    events = doc['event']['bodyText']
    event_count = len(events)
    event_feature_list = []
    ps = True
    for idx in range(0, event_count):
        feats = events[idx]['feature']['featureArray']
                
        ps, corref = 0, 0
        if 'parent_score' in events[idx]:
            ps, corref = events[idx]['parent_score']/event_count, events[idx]['corref_score']/event_count
        else:
            ps=False
        
        event_sent_loc = feats[-1]
        event_trig_freq = feats[-2]
        max_arg_mc, is_arg_sal, is_arg_ne = 0, 0, 0
        if 'entityFeatures' in events[idx]:
            max_arg_mc = max(0, events[idx]['entityFeatures'][0]-3)
            is_arg_sal = events[idx]['entityFeatures'][1]
            is_arg_ne = events[idx]['entityFeatures'][2]
        event_feature_list.append([event_sent_loc , 
                                   event_trig_freq, 
                                   max_arg_mc, 
                                   is_arg_sal, 
                                   is_arg_ne, 
                                   ps, 
                                   corref, 
                                   event_trig_freq/event_count])
    return event_feature_list

def get_events_from_doc(doc):
    events = doc['event']['bodyText']
    event_count = len(events)
    event_sur_list = []
    event_frame_list = []
    event_loc_list = []
    event_lab_list = []
    for idx in range(0, event_count):
        loc = get_loc(events[idx])    
        event_sur_list.append(events[idx]['surface'])
        event_frame_list.append(events[idx]['frame_name'])
        event_loc_list.append(loc)
        event_lab_list.append(events[idx]['salience'])
    return event_sur_list, event_loc_list, event_lab_list, event_frame_list

'''
    Output: b*512, b*512, n*1, n*10
'''
def get_tokenized_input(tokenizer, line_num, docbody, event_sur_list, event_frame_list=None):
    body_tokens = docbody.lower().split()
    input_ids = []
    attention_masks = []
    relative_locs = []
    
    body_token_len = len(body_tokens)
    batches = math.ceil(body_token_len/400)
    for batch in range(0, batches):
        left, right = batch*400 , min((batch+1)*400, body_token_len)
        context_tokens = body_tokens[left:right]
        context = ' '.join(context_tokens)
        encoded_dict = tokenizer.encode_plus(context, # Sentence to encode. 
                                             add_special_tokens = True, # Add '[CLS]' and '[SEP]' 
                                             max_length = 512, # Pad & truncate all sentences. 
                                             pad_to_max_length = True,
                                             attention_mask = True, # Construct attn. masks. 
                                             tensors = 'pt', # Return pytorch tensors. 
                                            )
        input_ids.append(encoded_dict['input_ids']) 
        attention_masks.append(encoded_dict['attention_mask']) # And its attention mask (simply differentiates padding from non-padding). 
        
    frame_input_ids = []
    frame_attention_masks = []
    if(event_frame_list != None):
        for frame in event_frame_list:
            encoded_dict = tokenizer.encode_plus(frame, # Sentence to encode. 
                                                 add_special_tokens = True, # Add '[CLS]' and '[SEP]' 
                                                 max_length = 16, # Pad & truncate all sentences. 
                                                 pad_to_max_length = True,
                                                 attention_mask = True, # Construct attn. masks. 
                                                 tensors = 'pt', # Return pytorch tensors. 
                                                )
            frame_input_ids.append(encoded_dict['input_ids']) 
            frame_attention_masks.append(encoded_dict['attention_mask'])         

    return input_ids, attention_masks, frame_input_ids, frame_attention_masks


def my_collate(data):
    def merge_2d_float(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths), sequences[0].shape[1]).float()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths
    
    def merge_2d(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths), sequences[0].shape[1]).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    def merge_1d(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    data.sort(key=lambda x: len(x[0]), reverse=True)

    # seperate source and target sequences
    input_ids, attention_masks, labels, locs, features, frame_input_ids, frame_attention_masks, docno = zip(*data)
    input_ids, input_lengths = merge_2d(input_ids)
    attention_masks, attention_masks_lengths = merge_2d(attention_masks)
    frame_input_ids, frame_input_lengths = merge_2d(frame_input_ids)
    frame_attention_masks, frame_attention_masks_lengths = merge_2d(frame_attention_masks)
    labels, labels_lengths = merge_1d(labels)
    locs, locs_lengths = merge_2d(locs)
    features, feats_lengths = merge_2d_float(features)
    return input_ids, attention_masks, labels, locs, features, frame_input_ids, frame_attention_masks, torch.tensor(labels_lengths), torch.tensor(docno)

