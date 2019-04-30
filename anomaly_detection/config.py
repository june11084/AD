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
parser.add_argument('--log_interval', type=int, default=10, help='print log interval (default: 10 epochs)')
parser.add_argument('--analysis', type=str, default=None, help='doing analysis, plot, etc')

# training args
parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training (default: 16)')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--chunk_len', type=int, default=28, help='chunk length (default: 28 for 7 hours)')
parser.add_argument('--stride', type=int, default=1, help='sliding window stride (default: 1)')
parser.add_argument('--num_workers', type=int, default=6, help='number of dataloader workers (default: 6)')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
parser.add_argument('--alpha', type=float, default=1e-2, help='regularizer coef (default=0.01)')

# train or test
parser.add_argument('--train', type=str, default=None, help='train mode: "train_teacher" -> train teacher model; "train_student" -> train student model with logvar output; "test_student" -> test student model; "test" -> test or dropout test for teacher model; "load" -> only load results; default = None, do nothing')
parser.add_argument('--dropout', type=str, default=None, help='whether applying dropout in the testing given trained teacher model, default = no dropout')
parser.add_argument('--build_std', type=str, default=None, help='build dataset with std, save to --custom_data')
parser.add_argument('--custom_data', type=str, default=None, help='custom dataset path')

# path args
parser.add_argument('--check_path', type=str, default='./checkpoint/z.pk', help='save checkpoint path')
parser.add_argument('--load_check', type=str, default='', help='load checkpoint path')

# attack args
parser.add_argument('--attack_bound', type=float, default=1e-2, help='infty bound of attack (default: 0.01)')
parser.add_argument('--attack_savepath', type=str, default='', help='attack save path')

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

