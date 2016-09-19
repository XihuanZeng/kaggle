# this script add features to each training and test set
# count features: for each cluster i, for each categorical variable x and for each level j within x, we calculate
#                 (num of sample whose hotel_cluster=i and its variable X has level j) / (num of sample whose hotel_cluster=i)
#                 where num of sample is weighted by book:17, click:3
# historical book and click ratio features: book to click ratio of cluster i for users who share the same attribute of booking j
# destination features: features from destinations.csv but the dimension is reduced to 10
import pandas as pd
import json
import os
import pickle
from collections import defaultdict
from sklearn.decomposition import PCA

def create_feature_dict(data, features, normalize = False):
    feature_dict = dict()
    grouped = data.groupby(features)
    for name, group in grouped:
        book = group[group.is_booking == 1]
        book_feature_dict = defaultdict(lambda : 0, dict(book.hotel_cluster.value_counts(normalize = normalize)))
        click = group[group.is_booking == 0]
        click_feature_dict = defaultdict(lambda : 0, dict(click.hotel_cluster.value_counts(normalize = normalize)))
        feature_dict[name] = {'book': book_feature_dict, 'click': click_feature_dict}
    return feature_dict

def add_count_features(dict_dir, model_input_dir, selected_features):
    assert 'train_count_dict.json' in os.listdir(dict_dir), 'train counter is not created'
    assert 'test_count_dict.json' in os.listdir(dict_dir), 'test counter is not created'
    for train_or_test in ['train', 'test']:
        with open(os.path.join(dict_dir, '%s_count_dict.json' % train_or_test), 'rb') as f:
            count_dict = json.load(f)
        f.close()
        for i in range(100):
            # add count features for each cluster
            data = pd.read_csv(os.path.join(model_input_dir, '%s/%s%s.csv' % (train_or_test, train_or_test, i)))
            for j in range(len(selected_features)):
                feature_dict = count_dict[str(i)][selected_features[j]]
                feature_dict = defaultdict(lambda: 0, feature_dict)
                data['c%s' % j] = [feature_dict[str(x)] for x in data[selected_features[j]]]
            data.to_csv(os.path.join(model_input_dir, '%s/%s%s.csv' % (train_or_test, train_or_test, i)))

def add_historical_book_click_feature(train_feature_dict, test_feature_dict, model_input_dir, features, number):
    for k in range(100):
        train_k = pd.read_csv(os.path.join(model_input_dir, 'train', 'train%s.csv' % k))
        test_k = pd.read_csv(os.path.join(model_input_dir, 'test', 'test%s.csv' % k))
        train_k['h%s' % number] = train_k.apply(lambda row: get_key(tuple(row[features]), train_feature_dict, k), axis = 1)
        test_k['h%s' % number] = test_k.apply(lambda row: get_key(tuple(row[features]), test_feature_dict, k), axis = 1)
        train_k.to_csv(os.path.join(model_input_dir, 'train', 'train%s.csv' % k))
        test_k.to_csv(os.path.join(model_input_dir, 'test', 'test%s.csv' % k))


def add_destination_features(data_dir, model_input_dir, ndim):
    dest = pd.read_csv(os.path.join(data_dir, 'destinations.csv'))
    pca = PCA(ndim)
    dest_reduced = pca.fit_transform(dest[[s for s in dest.columns if s.startswith('d')]])
    dest_dict = dict()
    for idx, i in enumerate(dest['srch_destination_id']):
        dest_dict[i] = dest_reduced[idx]
    for k in range(100):
        train_k = pd.read_csv(os.path.join(model_input_dir, 'train', 'train%s.csv' % k))
        tmp = pd.DataFrame(data = [dest_dict[i] for i in train_k['srch_destination_id']],
                           index = train_k['srch_destination_id'],
                           columns = ['d%s' % i for i in range(ndim)])
        for column in tmp.columns:
            train_k[column] = list(tmp[column])

        test_k = pd.read_csv(os.path.join(model_input_dir, 'test', 'test%s.csv' % k))
        tmp = pd.DataFrame(data = [dest_dict[i] for i in test_k['srch_destination_id']],
                           index = test_k['srch_destination_id'],
                           columns = ['d%s' % i for i in range(ndim)])
        for column in tmp.columns:
            test_k[column] = list(tmp[column])

        train_k.to_csv(os.path.join(model_input_dir, 'train', 'train%s.csv' % k))
        test_k.to_csv(os.path.join(model_input_dir, 'test', 'test%s.csv' % k))


data_dir = '../data'
ndim = 3

ho.columns.values

len(ho.srch_destination_id.value_counts())




train_or_test = 'train'
i = 0


ho = pd.read_csv('../data/ho.csv')
tr = pd.read_csv('../data/tr.csv')
te = pd.read_csv('../data/te.csv')

dict_dir = '../data/dict'
# add historical book click ratios as feature
f1 = create_feature_dict(ho, ['srch_destination_id', 'srch_rm_cnt'])
f1[(5059,1)]


model_input_dir = '../data/model_input'
k = 10
train_k = pd.read_csv(os.path.join(model_input_dir, 'train', 'train%s.csv' % k))
train_k.srch_destination_id
f1[(114, 10)]


def get_key(key, feature_dict, cluster):
    try:
        value = feature_dict[key]
    except KeyError:
        return 0
    book = value['book']
    click = value['click']
    try:
        return float(book[cluster]) / click[cluster]
    except ZeroDivisionError:
        return 0

f = create_feature_dict(ho, ['srch_destination_id'])


train_k.apply(lambda row: get_key(row['srch_destination_id'], f, 5), axis = 1)
train_k.apply(lambda row: get_key(tuple(row[['srch_destination_id', 'srch_rm_cnt']]), f1, 5), axis = 1).value_counts()




get_key(1114, f, 29)


f[1114]






f2 = create_feature_dict(pd.concat([ho, tr]),['srch_destination_id'])




feature_dict = dict()
grouped = ho.groupby(['srch_rm_cnt', 'srch_adults_cnt', 'srch_children_cnt'])









for name, group in grouped:
    if len(group) > 10:
        break

