# this script create 3 datasets, hold-out, train and validation.
# hold out set has all book and click record prior to 2014-01-01
# train set has book and click record within 2014-01-01 to 2014-08-31 that belongs to user that appears in hold-out set
# test set has book and click record after 2014-09-01 that belongs to user that appears in hold-out set

import math
import os
import json
import pickle
import pandas as pd


def create_feature_dict(dataset, feature_name, offset):
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
    with open('../data/dict/%s.json' % feature_name, 'wb') as f:
        json.dump(levels_dict, f)
    f.close()
    return len(levels)


def main():
    #dataset = pd.read_csv('../data/train.csv')
    dataset = pd.read_csv('/home/xihuan/Downloads/expedia/data/split_data/train/f0.csv')
    dataset['date_time'] = pd.to_datetime(dataset['date_time'], errors = 'ignore')
    dataset['year'] = dataset['date_time'].dt.year
    dataset['month'] = dataset['date_time'].dt.month

    # save the users list
    users = dataset[dataset.year == 2013].user_id.unique()
    users_dict = dict()
    for i in range(len(users)):
        users_dict[users[i]] = i
    with open('../data/users_dict.json', 'wb') as f:
        json.dump(users_dict, f)
    f.close()

    # load destinations.csv
    destinations = pd.read_csv('../data/destinations.csv')
    srch_destination_id = destinations.srch_destination_id

    # train/validation split
    dataset = dataset[(dataset.user_id.isin(users)) & (dataset.srch_destination_id.isin(srch_destination_id))]    # to make life easier
    ho = dataset[(dataset.year == 2013)]
    tr = dataset[(dataset.year == 2014) & (dataset.month >= 1) & (dataset.month <= 8)]
    te = dataset[(dataset.year == 2014) & (dataset.month > 8) & (dataset.month <= 12)]
    ho.to_csv('../data/ho.csv')
    tr.to_csv('../data/tr.csv')
    te.to_csv('../data/te.csv')

    # save categorical data in dict
    offset = len(users)
    categorical_features = ['site_name', 'posa_continent', 'user_location_country', 'user_location_region', 'user_location_city',
                            'is_mobile', 'is_package', 'channel', 'srch_destination_id', 'hotel_continent', 'hotel_country',
                            'hotel_market']
    for feature in categorical_features:
        offset += create_feature_dict(dataset, feature, offset)


if __name__ == '__main__':
    main()





