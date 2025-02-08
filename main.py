import torch
import os
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import argparse
import re
import torch.nn.functional as F
from utils import set_random_seed, load_TDA_data, Calculate_TDA_feature 
from model import Multi_PHLP, PHLP
from pytorchtools import EarlyStopping

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # GPU number

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

cur_dir = os.path.dirname(os.path.realpath(__file__))

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

def parse_list(argument):
    try:
        items = re.split('[,\[\]]', argument)
        return [int(x) for x in items if x]
    except ValueError:
        raise argparse.ArgumentTypeError("List items need to be integers")


parser = argparse.ArgumentParser(description='Link Prediction with Persistent-Homology')
#Dataset 
parser.add_argument('--data-name', default='USAir', help='graph name')

#training/validation/test divison and ratio
parser.add_argument('--test-ratio', type=float, default=0.1,
                    help='ratio of test links')
parser.add_argument('--val-ratio', type=float, default=0.05,
                    help='ratio of validation links. If using the splitted data from SEAL,\
                     it is the ratio on the observed links, othewise, it is the ratio on the whole links')
parser.add_argument('--practical-neg-sample', type=bool, default = False,
                    help='only see the train positive edges when sampling negative')
#setups in preparing the training set 
parser.add_argument('--Max-hops', type=int, default=3,
                    help='number of max hops in sampling subgraph')
parser.add_argument('--starting-hop-restric', type=parse_list, default=[3,100])
# parser.add_argument('--starting-hop-of-max-nodes', type=int, default=3)

#Node labeling settings
parser.add_argument('--node-label', type=str, default='degdrnl',
                    help='whether to use degree drnl labeling')
parser.add_argument('--deg-cut', type=int, default=5)
parser.add_argument('--centor-nodes', type=str, default='Target',
                    help='Choose whether the centor_nodes be to "Target" or "Random"')

#Persistent Homology settings
parser.add_argument('--onedim-PH', type=str2bool, default=False,
                    help='whether to use 1 dimensional persistent homology')
parser.add_argument('--multi-angle', type=str2bool, default=False,
                    help='whether to use Multi-angle PHLP')
parser.add_argument('--angle-hop', type=parse_list, default=[2,2],
                    help='whether to use Multi-angle PHLP')

#Model and Training
parser.add_argument('--seed', type=int, default=123,
                    help='random seed (default: 1)')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate')
parser.add_argument('--weight-decay', type=float, default=0)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--num-layers', type=int, default=3)
parser.add_argument('--batch-size', type=int, default=1024)
parser.add_argument('--epoch-num', type=int, default=10000)
parser.add_argument('--patience', type=int, default=20)

#Multi Process
parser.add_argument('--num-cpu', type=int, default=1)
parser.add_argument('--multiprocess', type=str2bool, default=True)

args = parser.parse_args()

if (args.data_name in ('Ecoli','PB')):
    args.starting_hop_restric=[2,100]
if args.data_name == 'Power':
    args.Max_hops=7
    args.centor_nodes='Random'
if args.data_name == 'Ecoli' and args.onedim_PH:
    args.Max_hops=2


print ("-"*40+'Dataset and Features'+"-"*40)
print ("{:<10}|{:<10}|{:<10}|{:<10}|{:<25}|{:<10}|{:<10}|{:<10}"\
    .format('Dataset','Test Ratio','Val Ratio','Max hops', 'starting hop restric', 'node label', 'deg cut', 'seed'))
print ("-"*100)
print ("{:<10}|{:<10}|{:<10}|{:<10}|{:<25}|{:<10}|{:<10}|{:<10}"\
    .format(args.data_name,args.test_ratio,args.val_ratio,str(args.Max_hops),str(args.starting_hop_restric), str(args.node_label), str(args.deg_cut), str(args.seed) ))
print ("-"*100)

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
Calculate_TDA_feature(args)

train_loader, val_loader, test_loader = load_TDA_data(args)
print('<<Completed>>')

sample = next(iter(train_loader))
set_random_seed(args.seed)
if args.multi_angle:
    hidden_channels = sample.delta_TDA_feature.size(2)
    model = Multi_PHLP(num_multi=sample.delta_TDA_feature.size(1), hidden_channels_PI=hidden_channels, num_layers=args.num_layers, dropout=args.dropout)
else:
    hidden_channels = sample.delta_TDA_feature.size(1)
    model = PHLP(hidden_channels_PI=hidden_channels, num_layers=args.num_layers, dropout=args.dropout)

print ("-"*40+'Model and Training'+"-"*45)
print ("{:<14}|{:<13}|{:<8}|{:<11}|{:<7}|{:<16}|{:<17}|{:<10}"\
    .format('Learning Rate','Weight Decay', 'Dropout', 'Batch Size','Epoch',\
        'Hidden Channels', 'number of layers', 'Patience'))
print ("-"*103)

print ("{:<14}|{:<13}|{:<8}|{:<11}|{:<7}|{:<16}|{:<17}|{:<10}"\
    .format(args.lr,args.weight_decay, str(args.dropout), str(args.batch_size),\
        args.epoch_num, hidden_channels, args.num_layers, args.patience))
print ("-"*103)

def train(loader, multi_angle=False):
    model.train()
    
    total_loss = 0
    for data in tqdm(loader, desc="train"):
        dealta_feature = data.delta_TDA_feature.to(device)
                
        optimizer.zero_grad()
        loss = 0
        link_labels = data.y.to(device)
        if multi_angle:
            link_prob, link_probs = model(dealta_feature, each_result=True)
            for i in range(link_probs.size(1)):
                loss += F.binary_cross_entropy(link_probs[:,i], link_labels.to(torch.float))
            loss += F.binary_cross_entropy(link_prob, link_labels.to(torch.float))
        else:
            link_prob = model(dealta_feature)
            loss += F.binary_cross_entropy(link_prob, link_labels.to(torch.double))
        loss.backward()
        optimizer.step()
        total_loss += loss * data.y.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def test(loader, multi_angle=False, data_type='test'):
    model.eval()
        
    total_loss = 0
    y_pred, y_true = [], []
    for data in tqdm(loader, desc='test:'+data_type):
        dealta_feature = data.delta_TDA_feature.to(device)
                
        optimizer.zero_grad()
        link_prob = model(dealta_feature)
        link_labels = data.y.to(device)
        if multi_angle:
            total_loss += F.binary_cross_entropy(link_prob, link_labels.to(torch.float)) * data.y.size(0)
        else:
            total_loss += F.binary_cross_entropy(link_prob, link_labels.to(torch.double)) * data.y.size(0)
        y_pred.append(link_prob.view(-1).cpu())
        y_true.append(data.y.view(-1).cpu().to(torch.float))
    loss = total_loss / len(loader.dataset)
    val_pred, val_true = torch.cat(y_pred), torch.cat(y_true)
    
    return roc_auc_score(val_true, val_pred), average_precision_score(val_true, val_pred), loss

model = model.to(device)
parameters = list(model.parameters())
optimizer = torch.optim.Adam(params=parameters, lr=args.lr, weight_decay=args.weight_decay)
early_stopping = EarlyStopping(patience = args.patience)

Best_Val_fromAUC = 0
Best_Val_loss = 10000
Final_Test_AUC_fromAUC=0

file_dir_ =  cur_dir + '/data/result'   
if not os.path.exists(file_dir_):
    os.makedirs(file_dir_)
    
if args.onedim_PH:
    if args.multi_angle:
        filename = file_dir_ + "/Multi_PHLP_{}_seed{}.txt".format(args.data_name, args.seed)
    else:
        filename = file_dir_ + "/PHLP_{}_seed{}.txt".format(args.data_name, args.seed)
else:
    if args.multi_angle:
        filename = file_dir_ + "/Multi_PHLP(0dim)_{}_seed{}.txt".format(args.data_name, args.seed)
    else:
        filename = file_dir_ + "/PHLP(0dim)_{}_seed{}.txt".format(args.data_name, args.seed)
    
f = open(filename, 'w')
f.write(filename + '\n')
f.close()

for epoch in range(args.epoch_num):
    loss_epoch = train(train_loader, multi_angle=args.multi_angle)
    val_auc, val_ap, val_loss = test(val_loader, multi_angle=args.multi_angle, data_type='val')
    if val_auc > Best_Val_fromAUC:
        test_auc, test_ap, test_loss = test(test_loader, multi_angle=args.multi_angle, data_type='test')
        Best_Val_fromAUC = val_auc
        Final_Test_AUC_fromAUC = test_auc
    if epoch%10 == 0:
        f = open(filename, 'a')
        f.write(f'Epoch: {epoch:03d}, Loss : {loss_epoch:.4f}, ValLoss : {val_loss:.4f},\
                    Test AUC: {test_auc:.4f}, Picked AUC:{Final_Test_AUC_fromAUC:.4f} \n')
        f.close()
        print(f'Epoch: {epoch:03d}, Loss : {loss_epoch:.4f}, ValLoss : {val_loss:.4f}, \
        Test AUC: {test_auc:.4f}, Picked AUC:{Final_Test_AUC_fromAUC:.4f}')
    early_stopping(val_loss.item(), model)
    if early_stopping.early_stop:
        print("Early stopping")
        f = open(filename, 'a')
        f.write('Early Stopping /n')
        f.write(f'From AUC: Final Test AUC: {Final_Test_AUC_fromAUC:.4f}'+ '\n\n')
        f.close()
        break

print(f'From AUC: Final Test AUC: {Final_Test_AUC_fromAUC:.4f}')
f = open(filename, 'a')
f.write(f'From AUC: Final Test AUC: {Final_Test_AUC_fromAUC:.4f}'+ '\n')
f.close()
