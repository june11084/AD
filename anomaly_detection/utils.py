import torch
import matplotlib.pyplot as plt
import numpy as np

def idx2onehot(idx, n):

    assert torch.max(idx).item() < n
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)

    idx = idx.cuda()
    onehot = torch.zeros(idx.size(0), n).cuda()
    onehot.scatter_(1, idx, 1)

    return onehot


# perform anomaly identification based on frac
def cut(test_loss, frac=0.05):
    # perform cut
    k = int(len(test_loss)*frac)
    idx_anomly = sorted(range(len(test_loss)), key=lambda i: test_loss[i])[-k:]
    idx_normal = [i for i in range(len(test_loss)) if i not in idx_anomly]
    return idx_anomly, idx_normal

def plot_hist(test_loss):
    # plot histogram
    plt.hist(test_loss, bins=100)
    plt.savefig(args.fig_path+'loss_hist.pdf', format='pdf')
    # plt.show()

def plot_detect(test_loss, test_data, idx_anomaly, sample_var=None):
    # plot position
    y = np.zeros_like(test_data)
    y[list(map(lambda e: e+27, idx_anomaly))] = np.asarray(test_loss)[list(idx_anomaly)]

    if sample_var is None:
        fig, (ax1, ax2) = plt.subplots(2,1)
        ax1.plot(test_data, 'g')
        ax2.plot(y, 'r')
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(3,1)
        ax1.plot(test_data, 'g')
        ax2.plot(y, 'r')
        ax3.plot(sample_var, 'b')
    plt.savefig(args.fig_path+'detect.pdf', format='pdf')
    # plt.show()

