import pyodbc
import csv
import sys

node_id = sys.argv[1]
channel = sys.argv[2]
filename = sys.argv[3]

# connect to db and build cursor
conn_str = "DRIVER={SQL Server};SERVER=hbsedasqlu1dev1.am.munichre.com,1866;DATABASE=MESHIFY_LANDING_RD;UID=bia_connect;PWD=b1a_Connect"
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

# # print all tables with "reporting" in their names
# for row in cursor.tables():
#     if 'reporting' in row.table_name:
#         print(row.table_name)

# some sql template
#begin_datetime = '2018-01-01'
#end_datetime = '2018-03-31'
#qry_str = "select rawData, dataType, messageDate from " + table + " where sensorID=" + str(sensorid) + " and messageDate>='" + begin_datetime + "' and messageDate<='" + end_datetime + "'"
#a = pd.read_sql(qry_str, conn)

# send cursor sql query
# sql = "select * from meshify.folders_reporting" # get folders (accounts)
# sql = "select * from meshify.nodes_reporting" # get nodes (sensor devices)
sql = "select * from meshify.data_points where node_id = "+node_id+" and channel = '"+channel+"'"
cursor.execute(sql)

# write to csv file
print('write to csv ...')
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