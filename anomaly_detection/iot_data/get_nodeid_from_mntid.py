import pandas as pd
import sys
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pyodbc
import csv

file_path = sys.argv[1]
save_folder = sys.argv[2]
data = pd.read_csv(file_path)
channel = 'temp'

# obtain unique ids
unique_ids = data['device_id'].unique()
print(unique_ids)

# connect to db and build cursor
conn_str = "DRIVER={SQL Server};SERVER=hbsedasqlu1dev1.am.munichre.com,1866;DATABASE=MESHIFY_LANDING_RD;UID=bia_connect;PWD=b1a_Connect"
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()
node_ids = []

for unique_id in unique_ids:
    # send cursor sql query
    # sql = "select * from meshify.folders_reporting" # get folders (accounts)
    # sql = "select * from meshify.nodes_reporting" # get nodes (sensor devices)
    # sql = "select * from meshify.data_points where node_id = "+node_id+" and channel = '"+channel+"'"
    sql = "select * from meshify.nodes where unique_id = '"+unique_id+"'"
    print(sql)
    cursor.execute(sql)
    rows = cursor.fetchall()
    try:
        node_ids.append(rows[0][1])
    except:
        print(rows, ' empty!')
print('finish getting node_id, start fetching data')

for node_id in node_ids:
    sql = "select * from meshify.data_points where node_id = "+str(node_id)+" and channel = '"+channel+"'"
    cursor.execute(sql)
    
    # write to csv file
    print('write node '+str(node_id)+' to csv ...')
    filename = save_folder + str(node_id) + '_data'
    with open(filename, 'w') as file:
        f = csv.writer(file, lineterminator='\n')
        f.writerow([d[0] for d in cursor.description]) # write the head
        rows = cursor.fetchall()
        for row in rows:
            try:
                f.writerow(row) # write the remaining
            except:
                # some rows contain characters that are not recognized by csvwriter
                tmp = (list(map(lambda e: e.encode("utf-8") if type(e)==str else e, row)))
                print(tmp)
                f.writerow(tmp)
        file.close()
    print('finish writing')

# close the db
cursor.close()
conn.close()


