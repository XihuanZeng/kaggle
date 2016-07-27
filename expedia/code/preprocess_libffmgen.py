# this script is to generate data input for libffm
# namely <label> <field1>:<index1>:<value1> <field2>:<index2>:<value2> ...
# the feature used are the categorical features

import math
import os
import json
import pickle
import pandas as pd
import argparse


def generate_libffm_kth_model(data, dict_list, categorical_features, k):
    f = open('../data/libffm_data/train.txt', 'wb')
    for index, row in data.iterrows():
        is_booking = row['is_booking']
        if is_booking == 0:
            continue
        cluster = row['hotel_cluster']
        writable = ''
        for j in range(len(dict_list)):
            writable += '%s:%s:1 ' % (j, dict_list[j][str(row[categorical_features[j]])])
        if cluster == k:
            f.write('1' + ' ' + writable + '\n')
        else:
            f.write('0' + ' ' + writable + '\n')
    f.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster',  type=int)
    args = parser.parse_args()
    k = args.cluster
    data = pd.read_csv('../data/ho.csv')
    categorical_features = ['site_name', 'posa_continent', 'user_location_country', 'user_location_region', 'user_location_city',
                            'is_mobile', 'is_package', 'channel', 'srch_destination_id', 'hotel_continent', 'hotel_country',
                            'hotel_market']
    dict_list = []
    for categorical_feature in categorical_features:
        g = json.load(open('../data/dict/%s.json' % categorical_feature))
        dict_list.append(g)
    generate_libffm_kth_model(data, dict_list, categorical_features, k)

if __name__ == '__main__':
    main()

# ../libffm/ffm-train -t 20 ../data/libffm_data/train.txt libffm_model1