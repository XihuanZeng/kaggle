# this script is to generate data input for libffm
# namely <label> <field1>:<index1>:<value1> <field2>:<index2>:<value2> ...
# the feature used are the categorical features

import math
import os
import json
import pickle
import pandas as pd
import argparse

def create_feature_dict(dataset, dict_dir, feature_name, offset):
    """
    create one-hot encoding for each categorical level and given a universal id corresponding to that variable
    :param feature_name: e.g name of categorical feature such as 'user_id', 'user_location_city'
    :param offset:
    :return: if offset = 100, the categorical feature has 2 levels, 0 and 1. then a dict(1:10000, 2:10001) will return
    """
    levels = dataset[feature_name].unique()
    levels_dict = dict()
    for i in range(len(levels)):
        levels_dict[levels[i]] = offset + i
    with open(os.path.join(dict_dir, '%s.json' % feature_name), 'wb') as f:
        json.dump(levels_dict, f)
    f.close()
    return len(levels)


def generate_libffm_kth_model(dataset, dict_dir, categorical_features, k, output_file):
    # this will run only once
    if not set(['%s.json' % feature for feature in categorical_features]) < set(os.listdir(dict_dir)):
        offset = 0
        for feature in categorical_features:
            offset += create_feature_dict(dataset, dict_dir, feature, offset)

    f = open(output_file, 'wb')
    dict_list = []
    for categorical_feature in categorical_features:
        g = json.load(os.path.join(dict_dir, '%s.json' % categorical_feature))
        dict_list.append(g)
    for index, row in dataset.iterrows():
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
    generate_libffm_kth_model(data, categorical_features, k, output_file = '../data/libffm_data/train.txt')

if __name__ == '__main__':
    main()

# ../libffm/ffm-train -t 20 ../data/libffm_data/train.txt libffm_model1