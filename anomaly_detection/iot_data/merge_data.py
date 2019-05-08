import os
import sys
import numpy as np

load_path = sys.argv[1]
save_path = sys.argv[2]

all_data = {}
for folder_name in os.listdir(load_path):
    print(folder_name)
    for file_name in os.listdir(load_path+folder_name):
        if 'png' not in file_name:
            if file_name not in all_data.keys():
                all_data[file_name] = np.loadtxt(load_path+folder_name+'/'+file_name)
            else:
                tmp = np.loadtxt(load_path+folder_name+'/'+file_name)
                all_data[file_name] = np.concatenate((all_data[file_name], tmp), axis=0)

print(all_data.keys())
for key in all_data.keys():
    np.savetxt(save_path+key+'.nptxt', all_data[key])
