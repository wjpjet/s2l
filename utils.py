"""Utility Functions"""
import os
import json
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import vocab

def large_randint():
    return random.randint(int(1e5), int(1e6))

def set_all_seeds(rand_seed):
    random.seed(rand_seed)
    np.random.seed(large_randint())
    torch.manual_seed(large_randint())
    torch.cuda.manual_seed(large_randint())

class Config(object):
    def __init__(self, attrs):
        self.__dict__.update(attrs)

    @classmethod
    def load(cls, filename):
        with open(filename, 'r') as f:
            attrs = json.load(f)
        return cls(attrs)

    def dump(self, filename):
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        print('Results will be stored in:', dirname)

        with open(filename, 'w') as f:
            json.dump(vars(self), f, sort_keys=True, indent=2)
            f.write('\n')

    def __repr__(self):
        return json.dumps(vars(self), sort_keys=True, indent=2)

def init_net(net, net_file):
    if net_file:
        net.load_state_dict(torch.load(net_file))
    else:
        net.apply(weights_init)

def assert_zero_grads(params):
    for p in params:
        if p.grad is not None:
            assert_eq(p.grad.data.sum(), 0)

def batch_index_gen(batch_size, size):
    batch_indexer = []
    start = 0
    while start < size:
        end = start + batch_size
        if end > size:
            end = size
        batch_indexer.append((start, end))
        start = end
    return batch_indexer

def convert_to_numpy(data, vocab, padding_length=50):
   
    total_size = len(data)    
    np_s1 = np.empty(shape=(total_size, padding_length), dtype=np.float32)
    np_s2 = np.empty(shape=(total_size, padding_length), dtype=np.float32)
    np_y = np.empty(shape=(total_size), dtype=np.float32)
    s1_len = []
    s2_len = []
    
    for i, (s1, s2, y) in enumerate(data):
        s1 = np.asarray(s1, dtype=np.float32)
        s2 = np.asarray(s2, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        # If less than length 50: Pad to the length of 50 on the right, with zeros
        if len(s1) > padding_length:
            s1 = s1[:padding_length]
            s1[-1] = vocab.stoi['<eos>'] # Assign stop word to end
            s1_len.append(padding_length)
        else:
            s1 = np.lib.pad(s1, (0, padding_length - len(s1)), 'constant')
            s1_len.append(len(s1))
        # Same to S2
        if len(s2) > padding_length:
            s2 = s2[:padding_length]
            s2[-1] = vocab.stoi['<eos>'] # Assign stop word to end
            s2_len.append(padding_length)
        else:
            s2 = np.pad(s2, (0, padding_length - len(s2)), 'constant')
            s2_len.append(len(s2))
            
        # concatenate vectors
        np_s1[i] = s1
        np_s2[i] = s2
        np_y[i] = y
        
    return np_s1, np_s2, np_y, (s1_len, s2_len)

def obtain_data(file_name, vocab_file_name):
    json_data = []
    
    with open(file_name) as data_file:
        for l in data_file:
            json_data.append(json.loads(l))
    
    data = []
    count = 0
    for item in json_data:
        if item.get('gold_label') == '-':
            #count += 1
            continue
        sentence_pair = (item['sentence1'], item['sentence2'], item['gold_label'])
        data.append(sentence_pair)

    stoi = []
    with open(vocab_file_name) as voc_file:
        stoi = voc_file.readlines()
        stoi = [x.strip() for x in stoi] 

    return data, stoi
