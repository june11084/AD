from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import time
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import utils

parser = argparse.ArgumentParser(description='Anomaly Detection')
# path args
parser.add_argument('train_path', type=str, metavar='TRAIN_PATH', help='load train data path')
parser.add_argument('test_path', type=str, metavar='TEST_PATH', help='load test data path')

parser.add_argument('--log_interval', type=int, default=10, metavar='N', help='print log interval (default: 10 epochs)')
# training args
parser.add_argument('--batch_size', type=int, default=16, metavar='N', help='input batch size for training (default: 16)')
parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='N', help='random seed (default: 1)')
parser.add_argument('--chunk_len', type=int, default=28, metavar='N', help='chunk length (default: 28 for 7 hours)')
parser.add_argument('--stride', type=int, default=1, metavar='N', help='sliding window stride (default: 1)')
parser.add_argument('--num_workers', type=int, default=6, metavar='N', help='number of dataloader workers (default: 6)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='F', help='learning rate (default: 1e-3)')

# train or test
parser.add_argument('--train', type=int, default=0, metavar='B', help='train mode, default=False')
parser.add_argument('--dropout', type=int, default=0, metavar='B', help='whether applying dropout in the testing given trained model')

# path args
parser.add_argument('--check_path', type=str, default='./checkpoint/z.pk', metavar='C', help='save checkpoint path')
parser.add_argument('--load_check', type=str, default='', metavar='C', help='load checkpoint path')

# attack args
parser.add_argument('--attack_bound', type=float, default=1e-2, metavar='F', help='infty bound of attack (default: 0.01)')
parser.add_argument('--attack_savepath', type=str, default='', metavar='C', help='attack save path')

args = parser.parse_args()
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dim = 1

print("start reading data")
# read data and build train dataloader
data_train = np.loadtxt(args.train_path)
data_train = (data_train - min(data_train)) / (max(data_train) - min(data_train)) # standardize data
i = 0
chunks_train = np.empty((0, 1, args.chunk_len))
while i < len(data_train)-args.chunk_len+1:
    tmp = np.reshape(data_train[i:i+args.chunk_len], (1,1,-output_dim)) # 1 * channel_num * length
    chunks_train = np.concatenate((chunks_train, tmp), axis=0)
    i += args.stride
print("finish building data chunks_train")
chunks_train = torch.tensor(chunks_train, dtype=torch.float)
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(chunks_train), batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
input_size = chunks_train.shape[1]


# read data and build test dataloader
data_test = np.loadtxt(args.test_path)
data_test = (data_test - min(data_test)) / (max(data_test) - min(data_test)) # standardize data, CAREFUL!!!!, only use train value to normalize, but can use test data to standardize
i = 0
chunks_test = np.empty((0, 1, args.chunk_len))
while i < len(data_test)-args.chunk_len+1:
    tmp = np.reshape(data_test[i:i+args.chunk_len], (1,1,-output_dim)) # 1 * channel_num * length
    chunks_test = np.concatenate((chunks_test, tmp), axis=0)
    i += args.stride
print("finish building data chunks_test")
chunks_test = torch.tensor(chunks_test, dtype=torch.float)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(chunks_test), batch_size=1, num_workers=args.num_workers, shuffle=False)
print("finish building data loader")

