import pandas as pd
import sys
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

def convert_time_to_int(s): # convert a "year-month-date hour-minute-second" string to integer
    datetime_object = datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
    return int(time.mktime(datetime_object.timetuple()))

def avg_every_k_minutes(df, k, starting_time): # starting time need to be in format of "year-month-date hour-minute-second"
    starting_unix_time = convert_time_to_int(starting_time)
    interval = k*60
    df['unix_time_interval_id'] = df['unix_time'].map(lambda e: e - e%interval)
    interval_ids = df['unix_time_interval_id'].unique()
    return_list = []
    for e in interval_ids:
        tmp = df.loc[df['unix_time_interval_id']==e]
        return_list.append(tmp['value'].map(float).mean())
    return(return_list)

file_path = sys.argv[1]
save_path = sys.argv[2]
data = pd.read_csv(file_path)

# obtain channels and node_ids
channels = data['channel'].unique()
node_ids = data['node_id'].unique()
print('channels: ', channels)
print('node_ids: ', node_ids)

channel = 'temp'
for node_id in node_ids:

    # get subset data
    examples = data.loc[(data['channel']==channel)&(data['node_id']==node_id)]
    examples['unix_time'] = examples['timestamp'].map(convert_time_to_int) # add a column that convert timestamp to unix_time stamp
    sorted_examples = examples.sort_values('unix_time', ascending=True) # sort based on the unix timestamp

    # round to 15 mins
    # starting_time = '2018-10-01 00:00:00' # for node 54661
    starting_time = '2019-01-20 00:00:00' # for freezer data
    k = 15
    avg_result = avg_every_k_minutes(sorted_examples, k, starting_time)
    np.savetxt(save_path+'/'+str(node_id), avg_result)
    plt.plot(avg_result)
    plt.title(str(node_id))
    plt.savefig(save_path+'/figs/'+str(node_id)+'.png')
    # plt.show()
    # plt.show()
    #
    # print(sorted_examples.head(n=100))
    # plt.plot(sorted_examples['unix_time'].values[::], sorted_examples['value'].values[::])
    # plt.title('node_id: '+str(node_id))
    # plt.show()

