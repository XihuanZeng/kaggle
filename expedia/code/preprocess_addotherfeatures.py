# this script add features to each training and test set
# count features: for each cluster i, for each categorical variable x and for each level j within x, we calculate
#                 (num of sample whose hotel_cluster=i and its variable X has level j) / (num of sample whose hotel_cluster=i)
#                 where num of sample is weighted by book:17, click:3

# historical book and click ratio features: book and click ratio of cluster i that has attribute of jth booking.
#                                  there are two set of attributes
#                                  1. srch_destination_id
#                                  2. (srch_rm_cnt, srch_adults_cnt, srch_children_cnt) combination

# destination features: features from destinations.csv but the dimension is reduced to 3




import pandas as pd
import json
from collections import defaultdict

def create_feature_dict(data, features):
    feature_dict = dict()
    grouped = data.groupby(features)
    for name, group in grouped:
        book = group[group.is_booking == 1]
        book_feature_dict = defaultdict(lambda : 0, dict(book.hotel_cluster.value_counts(normalize = True)))
        click = group[group.is_booking == 0]
        click_feature_dict = defaultdict(lambda : 0, dict(click.hotel_cluster.value_counts(normalize = True)))
        feature_dict[name] = {'book': book_feature_dict, 'click': click_feature_dict}
    return feature_dict



def add_count_features(data, train_or_test,
                       selected_features = ['site_name', 'posa_continent', 'user_location_country', 'user_location_region', 'user_location_city',
                                            'is_mobile', 'is_package', 'channel', 'srch_destination_id', 'hotel_continent', 'hotel_country',
                                            'hotel_market']):
    with open('../data/dict/%s_count_dict.json' % train_or_test, 'rb') as f:
        count_dict = json.load(f)
    f.close()
    for i in range(100):
        # add count features for each cluster
        for j in range(len(selected_features)):
            feature_dict = count_dict[str(i)][selected_features[j]]
            feature_dict = defaultdict(lambda: 0, feature_dict)
            data['c%s' % j] = [feature_dict[str(x)] for x in data[selected_features[j]]]
        data.to_csv('../data/model_input/%s/%s%s.csv' % (train_or_test, train_or_test, i))


def add_historical_book_click_feature(cluster, feature_dict, train_or_test):




def add_destination_features():
    pass



ho.columns.values

len(ho.srch_destination_id.value_counts())




train_or_test = 'train'
i = 0


ho = pd.read_csv('../data/ho.csv')
tr = pd.read_csv('../data/tr.csv')
te = pd.read_csv('../data/te.csv')


# add count features
for i in range(100):
    train_data = pd.read_csv('../data/model_input/train/train%s.csv' % i)
    add_count_features(train_data, 'train')
    test_data = pd.read_csv('../data/model_input/test/test%s.csv' % i)
    add_count_features(test_data, 'test')

# add historical book click ratios as feature
f1 = create_feature_dict(ho,['srch_rm_cnt', 'srch_adults_cnt', 'srch_children_cnt'])
f2



feature_dict = dict()
grouped = ho.groupby(['srch_rm_cnt', 'srch_adults_cnt', 'srch_children_cnt'])









for name, group in grouped:
    if len(group) > 10:
        break

