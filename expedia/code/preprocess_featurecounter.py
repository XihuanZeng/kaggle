# this script will produce a dictionary. This will keep the normalized count(sum to 1) of each categorical variable for each hotel_cluster
# this count feature is collected use data only from hold out set which is not part of the training set
# this dictionary is used when creating training set


import pandas as pd
import json

ho = pd.read_csv('../data/ho.csv')
selected_features = ['site_name', 'posa_continent', 'user_location_country', 'user_location_region', 'user_location_city',
                    'is_mobile', 'is_package', 'channel', 'srch_destination_id', 'hotel_continent', 'hotel_country',
                    'hotel_market']
count_dict = dict()
for i in range(100):
    count_dict_cluster_i = dict()
    subset = ho[ho.hotel_cluster == i]
    for j in selected_features:
        count = subset[j].value_counts()
        count_dict_cluster_i_feature_j = dict()
        for k in count.index:
            total = sum(count)
            count_dict_cluster_i_feature_j[k] = count[k] / float(total)
        count_dict_cluster_i[j] = count_dict_cluster_i_feature_j
    count_dict[i] = count_dict_cluster_i

with open('../data/dict/count_dict.json', 'wb') as f:
    json.dump(count_dict, f)
f.close()

