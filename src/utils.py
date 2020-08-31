from transformers import BertTokenizer, get_linear_schedule_with_warmup
import nvidia_smi
import torch

btokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def decode_b(bert_encoded_tokens):
    bert_decoded_context = btokenizer.decode(bert_encoded_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)    
    return bert_decoded_context.lower().split()

def gpu_stats(message):
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

    res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
    print(f'{message} gpu: {res.gpu}%, gpu-mem: {res.memory}%')

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device

def print_active_params(mod):
    for name, param in mod.named_parameters():
        if param.requires_grad:
            print(name)#, param.data)