import torch
import sys
import numpy as np

# hyper params
data_path = sys.argv[1]
loss_file = sys.argv[2]
chunk_len = 28
stride = 1

# load loss file
(orig_loss_list, new_loss_list, data_mod_list) =  torch.load(loss_file)

# read data and build chunks
orig_data = np.loadtxt(data_path)
orig_data = (orig_data - min(orig_data)) / (max(orig_data) - min(orig_data)) # standardize data
i = 0
chunks = np.empty((0, 1, chunk_len))
while i < len(orig_data)-chunk_len+1:
    tmp = np.reshape(orig_data[i:i+chunk_len], (1,1,-1)) # 1 * channel_num * length
    chunks = np.concatenate((chunks, tmp), axis=0)
    i += stride

print("finish building data chunks")

y_total = sum(orig_data[chunk_len-1:])
total_orig_err = sum(map(abs,orig_loss_list))
total_mod_err = sum(map(abs,new_loss_list))

total_orig_err_rate = total_orig_err / y_total
total_mod_err_rate  = total_mod_err  / y_total

print(total_orig_err, total_mod_err, total_mod_err/total_orig_err)
print(total_orig_err_rate, total_mod_err_rate, total_mod_err_rate/total_orig_err_rate)

all_mods = np.zeros((chunks.shape[0], chunks.shape[1], chunks.shape[2]-1))
for i in range(chunks.shape[0]):
    all_mods[i] = data_mod_list[i] - chunks[i,0,:-1]

all_mods = np.reshape(np.array(all_mods), (-1))

# plot histogram
import matplotlib.pyplot as plt
plt.hist(all_mods, bins=100)
plt.show()

