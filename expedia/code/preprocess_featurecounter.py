# this script will produce a dictionary. This will keep the normalized count(sum to 1) of each categorical variable for each hotel_cluster
# this dictionary is used when creating training set
# we still use the magic ratio 3:17, i,e a click count as 3 and a book count as 17


import pandas as pd
import json
import os

def generate_feature_counter(data, selected_features, dict_dir, train_or_test):
    if os.path.exists(os.path.join(dict_dir, '%s_count_dict.json' % train_or_test)):
        f = open(os.path.join(dict_dir, '%s_count_dict.json' % train_or_test), 'rb')
        return json.load(f)
    else:
        count_dict = dict()
        for i in range(100):
            count_dict_cluster_i = dict()
            subset = data[data.hotel_cluster == i]
            for j in selected_features:
                book_count = subset[subset.is_booking == 1][j].value_counts()
                click_count = subset[subset.is_booking == 0][j].value_counts()
                count_dict_cluster_i_feature_j = dict()
                for k in click_count.index:
                    try:
                        count_dict_cluster_i_feature_j[k] = 17 * book_count[k] + 3 * click_count[k]
                    except KeyError:
                        count_dict_cluster_i_feature_j[k] = 3 * click_count[k]
                total = sum(count_dict_cluster_i_feature_j.values())
                for k in count_dict_cluster_i_feature_j.keys():
                    count_dict_cluster_i_feature_j[k] = count_dict_cluster_i_feature_j[k] / float(total)
                count_dict_cluster_i[j] = count_dict_cluster_i_feature_j
            count_dict[i] = count_dict_cluster_i
        with open(os.path.join(dict_dir, '%s_count_dict.json' % train_or_test), 'wb') as f:
            json.dump(count_dict, f)
        return count_dict






