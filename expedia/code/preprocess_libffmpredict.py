# this script transfer each training or test dataset to libffm format data

import pandas as pd
import json
from preprocess_libffmgen import generate_libffm_kth_model
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster',  type=int)
    args = parser.parse_args()
    k = args.cluster

    train_data = pd.read_csv('../data/model_input/train/train%s.csv' % k)
    test_data = pd.read_csv('../data/model_input/test/test%s.csv' % k)
    categorical_features = ['site_name', 'posa_continent', 'user_location_country', 'user_location_region', 'user_location_city',
                            'is_mobile', 'is_package', 'channel', 'srch_destination_id', 'hotel_continent', 'hotel_country',
                            'hotel_market']
    generate_libffm_kth_model(train_data, categorical_features, k, output_file = '../data/libffm_data/tmp_train_input.txt')
    generate_libffm_kth_model(test_data, categorical_features, k, output_file = '../data/libffm_data/tmp_test_input.txt')

if __name__ == '__main__':
    main()

# python preprocess_libffmpredict.py --cluster 0
# ./libffm/ffm-predict data/libffm_data/tmp_train_input.txt models/libffm_model0 data/libffm_data/tmp_train_output.txt
# ./libffm/ffm-predict data/libffm_data/tmp_test_input.txt models/libffm_model0 data/libffm_data/tmp_test_output.txt
# python preprocess_libffmaddfeature.py --cluster 0

