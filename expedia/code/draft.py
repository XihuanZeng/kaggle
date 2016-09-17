"""
path = '/home/xihuan/Downloads/expedia/data/split_data/train/f0.csv'
f = open(path, 'rb')
header = f.readline().strip().split(',')
user_idx = header.index('user_id')
date_idx = header.index('date_time')
cnt = 0
users = []
while 1:
    line = f.readline()
    cnt += 1
    if cnt % 10000 == 0:
        print 'read %s lines' % cnt
    if line == '':
        break
    arr = line.split(',')
    if arr[date_idx][:4] != '2013':
        continue
    users.append(arr[user_idx])

users = list(set(users))
users.sort()
"""


"""
def generate_libffm_kth_model(data, dict_list, categorical_features, k):
    f = open('../data/libffm_data/train.txt', 'wb')
    g = open('../data/libffm_data/validate.txt', 'wb')
    for index, row in data.iterrows():
        is_booking = row['is_booking']
        if is_booking == 0:
            continue
        cluster = row['hotel_cluster']
        writable = ''
        for j in range(len(dict_list)):
            writable += '%s:%s:1 ' % (j, dict_list[j][str(row[categorical_features[j]])])
        if cluster == k:
            if row['month'] <= 10:
                f.write('1' + ' ' + writable + '\n')
            else:
                g.write('1' + ' ' + writable + '\n')
        else:
            if row['month'] <= 10:
                f.write('0' + ' ' + writable + '\n')
            else:
                g.write('0' + ' ' + writable + '\n')
    f.close()
    g.close()
"""
import pandas as pd
data = pd.read_csv('/home/xihuan/gitrepos/kaggle/expedia/data/model_input/train/train0.csv')
data = pd.read_csv('/home/xihuan/gitrepos/kaggle/expedia/data/tr.csv')
data.columns.values

test = pd.read_csv('/home/xihuan/gitrepos/kaggle/expedia/data/model_input/test/test0.csv')
test.columns.values

"""
array(['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1', 'date_time',
       'site_name', 'posa_continent', 'user_location_country',
       'user_location_region', 'user_location_city',
       'orig_destination_distance', 'user_id', 'is_mobile', 'is_package',
       'channel', 'srch_ci', 'srch_co', 'srch_adults_cnt',
       'srch_children_cnt', 'srch_rm_cnt', 'srch_destination_id',
       'srch_destination_type_id', 'is_booking', 'cnt', 'hotel_continent',
       'hotel_country', 'hotel_market', 'hotel_cluster', 'year', 'month',
       'y', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9',
       'c10', 'c11', 'libffm_score'], dtype=object)
"""


# data = pd.read_csv('/home/xihuan/Downloads/coupon/coupon_list_train.csv')



import itertools
import numpy as np
from scipy import stats
import pylab as pl
from sklearn import svm, linear_model, cross_validation



import json
with open('../data/dict/train_count_dict.json', 'rb') as f:
    train_count_dict = json.load(f)
f.close()

train_count_dict['1']

f = open('../data/dict/train_count_dict.json', 'rb')
json.load(f)









