import argparse
import torch
import numpy as np
from data_loader import load_data
from train import train

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default='music', help='which dataset to use (music, book, movie, restaurant)')
parser.add_argument('--n_epoch', type=int, default=20, help='the number of epochs')
parser.add_argument('--cf_batch_size', type=int, default=64, help='cf batch size')
parser.add_argument('--kg_batch_size', type=int, default=128, help='ckg batch size')
parser.add_argument('--test_batch_size', type=int, default=2048, help='test batch size')
parser.add_argument('--n_layer', type=int, default=4, help='depth of layer')
parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
parser.add_argument('--l2_weight', type=float, default=1e-5, help='weight of the l2 regularization term')

parser.add_argument('--dim', type=int, default=128, help='dimension of entity and relation embeddings')
parser.add_argument('--triplet_set_size', type=int, default=64, help='the number of triplets in triplet set')
parser.add_argument('--agg', type=str, default='sum', help='the type of aggregator (sum, pool, concat)')

parser.add_argument('--use_cuda', type=bool, default=True, help='whether using gpu or cpu')
parser.add_argument('--show_topk', type=bool, default=False, help='whether showing topk or not')
parser.add_argument('--random_flag', type=bool, default=False, help='whether using random seed or not')

def set_random_seed(np_seed, torch_seed):
    np.random.seed(np_seed)                  
    torch.manual_seed(torch_seed)       
    torch.cuda.manual_seed(torch_seed)      
    torch.cuda.manual_seed_all(torch_seed)  

if not args.random_flag:
    set_random_seed(555, 2020)
    
data_info = load_data(args)
train(args, data_info)
    