# import os.path as osp
# import os

import torch
from sklearn.metrics import roc_auc_score, average_precision_score
# import numpy as np
# import scipy.sparse as ssp
from tqdm import tqdm
import argparse


import torch.nn.functional as F
from utils import set_random_seed, load_TDA_data, Calculate_TDA_feature 
from model import Multi_PHLP
from pytorchtools import EarlyStopping

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def str2none(v):
    if v.lower()=='none':
        return None
    else:
        return str(v)


parser = argparse.ArgumentParser(description='Link Prediction with Persistent-Homology')
#Dataset 
parser.add_argument('--data-name', default='USAir', help='graph name')

#training/validation/test divison and ratio
parser.add_argument('--test-ratio', type=float, default=0.1,
                    help='ratio of test links')
parser.add_argument('--val-ratio', type=float, default=0.05,
                    help='ratio of validation links. If using the splitted data from SEAL,\
                     it is the ratio on the observed links, othewise, it is the ratio on the whole links.')
parser.add_argument('--practical-neg-sample', type=bool, default = False,
                    help='only see the train positive edges when sampling negative')
#setups in preparing the training set 
parser.add_argument('--Max-hops', type=int, default=3,
                    help='number of max hops in sampling subgraph')
parser.add_argument('--starting_hop_restric', type=list, default=[3,100])
# parser.add_argument('--starting-hop-of-max-nodes', type=int, default=3)

#Node labeling settings
parser.add_argument('--node-label', type=str, default='degdrnl',
                    help='whether to use degree drnl labeling')
parser.add_argument('--deg-cut', type=int, default=5)

#Model and Training
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--lr', type=float, default=0.00005,
                    help='learning rate')
parser.add_argument('--weight-decay', type=float, default=0)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--batch-normalize', type=str2bool, default=False)
parser.add_argument('--weight-initialization', type=str2bool, default=False)
parser.add_argument('--hidden-channels', type=int, default=1024)
parser.add_argument('--num-layers', type=int, default=3)
parser.add_argument('--batch-size', type=int, default=1024)
parser.add_argument('--epoch-num', type=int, default=10000)
parser.add_argument('--patience', type=int, default=7)

#Multi Process
parser.add_argument('--num_cpu', type=int, default=1)
parser.add_argument('--multiprocess', type=str2bool, default=True)

args = parser.parse_args()

if (args.data_name in ('Ecoli','PB')):
    args.starting_hop_restric=[2,100]


print ("-"*35+'Dataset and Features'+"-"*35)
print ("{:<10}|{:<10}|{:<10}|{:<10}|{:<25}|{:<10}|{:<10}"\
    .format('Dataset','Test Ratio','Val Ratio','Max hops', 'starting hop restric', 'node label', 'deg cut'))
print ("-"*90)
print ("{:<10}|{:<10}|{:<10}|{:<10}|{:<25}|{:<10}|{:<10}"\
    .format(args.data_name,args.test_ratio,args.val_ratio,str(args.Max_hops),str(args.starting_hop_restric), str(args.node_label), str(args.deg_cut)))
print ("-"*90)

print ("-"*10+'Multi Process'+"-"*10)
print ("{:<15}|{:<15}"\
    .format('Number of cpu','Multi process'))
print ("-"*33)

if args.multiprocess == True:
    str_multiprocess = 'True'
else:
    str_multiprocess = 'False'

print ("{:<15}|{:<15}"\
    .format(args.num_cpu,str_multiprocess))
print ("-"*33)

print('<<Begin calculating Persistent Homological feature>>')
Calculate_TDA_feature(**vars(args))

train_loader, val_loader, test_loader = load_TDA_data(args.data_name, args.batch_size)
print('<<Completed>>')