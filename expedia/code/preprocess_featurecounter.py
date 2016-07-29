# this script will produce a dictionary. This will keep the normalized count(sum to 1) of each categorical variable for each hotel_cluster
# this count feature is collected use data only from hold out set which is not part of the training set
# this dictionary is used when creating training set
# we still use the magic ratio 3:17, i,e a click count as 3 and a book count as 17


import pandas as pd
import json

def generate_feature_counter(data, selected_features):
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
    return count_dict


def main():
    selected_features = ['site_name', 'posa_continent', 'user_location_country', 'user_location_region', 'user_location_city',
                        'is_mobile', 'is_package', 'channel', 'srch_destination_id', 'hotel_continent', 'hotel_country',
                        'hotel_market']
    ho = pd.read_csv('../data/ho.csv')
    tr = pd.read_csv('../data/tr.csv')

    # generate count dict for training stage, i,e collect count features from ho.csv
    train_count_dict = generate_feature_counter(ho, selected_features)
    with open('../data/dict/train_count_dict.json', 'wb') as f:
        json.dump(train_count_dict, f)
    f.close()

    # generate count dict for testing stage, i.e collect count features from ho.csv and tr.csv
    test_count_dict = generate_feature_counter(pd.concat([ho, tr]), selected_features)
    with open('../data/dict/test_count_dict.json', 'wb') as f:
        json.dump(test_count_dict, f)
    f.close()


if __name__ == '__main__':
    main()






