# this script will enlarge tr.csv and te.csv 100 times.
# for each event(line) in tr/te.csv, this will be expand to 100 event, one for each hotel_cluster.
# each has the same feature but the response is whether the ground truth hotel_cluster is the same as the hotel_cluster of this event
# for each of the 100 set generated from tr.csv, we keep only 10% of the 0 of the new response

import pandas as pd
import json
import random
from collections import defaultdict



def expanding(data, count_dict, selected_features, stage, sample_reduction = True):
    for i in range(100):
        tmp = data[data.is_booking == 1].copy()
        tmp['y'] = [1 if hotel_cluster == i else 0 for hotel_cluster in tmp['hotel_cluster']]
        if sample_reduction:
            positives = tmp['y'][tmp.y == 1].index.tolist()
            negatives = tmp['y'][tmp.y == 0].index.tolist()
            tmp = tmp.loc[positives + random.sample(negatives, int(0.1 * len(negatives)))]
        # add count features
        for j in range(len(selected_features)):
            feature_dict = count_dict[str(i)][selected_features[j]]
            feature_dict = defaultdict(lambda: 0, feature_dict)
            tmp['c%s' % j] = [feature_dict[str(x)] for x in tmp[selected_features[j]]]
        if stage == 'train':
            tmp.to_csv('../data/model_input/train/train%s.csv' % i)
        if stage == 'test':
            tmp.to_csv('../data/model_input/test/test%s.csv' % i)


def main():
    tr = pd.read_csv('../data/tr.csv')
    te = pd.read_csv('../data/te.csv')

    selected_features = ['site_name', 'posa_continent', 'user_location_country', 'user_location_region', 'user_location_city',
                        'is_mobile', 'is_package', 'channel', 'srch_destination_id', 'hotel_continent', 'hotel_country',
                        'hotel_market']

    with open('../data/dict/train_count_dict.json', 'rb') as f:
        train_count_dict = json.load(f)
    f.close()

    with open('../data/dict/test_count_dict.json', 'rb') as f:
        test_count_dict = json.load(f)
    f.close()

    # generate training data
    expanding(tr, train_count_dict, selected_features, 'train')
    # generate test data
    expanding(te, test_count_dict, selected_features, 'test', sample_reduction = False)


if __name__ == '__main__':
    main()