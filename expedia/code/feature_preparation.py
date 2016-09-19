import math
import os
import json
import pickle
import pandas as pd
import numpy as np

from preprocess_featurecounter import generate_feature_counter
from preprocess_traingen import expanding
from preprocess_libffmgen import generate_libffm_kth_model
from preprocess_libffmaddfeature import add_libffm_feature
from preprocess_addotherfeatures import add_count_features, create_feature_dict, add_historical_book_click_feature, add_destination_features

class Featured_Dataset():
    CATEGORICAL_FEATURES = ['site_name', 'posa_continent', 'user_location_country', 'user_location_region', 'user_location_city',
                            'is_mobile', 'is_package', 'channel', 'srch_destination_id', 'hotel_continent', 'hotel_country',
                            'hotel_market']

    # this is used to find similar users
    # 'similar' is defined by users sharing the same search attributes, which are elements of SAME_ATTRIBUTES
    # since exact same search queries are rare, if they share some key attributes, we still treat them as similar
    SAME_ATTRIBUTES = [['srch_destination_id'], ['hotel_market'] , ['user_location_country'], ['srch_rm_cnt'],
                       ['srch_children_cnt'], ['month'], ['hotel_market', 'month'], ['hotel_market', 'is_package']]

    def __init__(self, dataset, data_dir, dict_dir, model_input_dir, libffm_dir):
        """
        :param dataset: this is the sampled dataset
        :return:
        """
        self.dataset = dataset
        self.dict_dir = dict_dir
        self.data_dir = data_dir
        self.model_input_dir = model_input_dir
        self.libffm_dir = libffm_dir
        self.holdout_set, self.train_set, self.test_set = self.TrainTestSplit()

    def TrainTestSplit(self):
        if os.path.exists(os.path.join(self.data_dir, 'ho.csv')):
            ho = pd.read_csv(os.path.join(self.data_dir, 'ho.csv'))
        else:
            ho = self.dataset[(dataset.year == 2013) | ((dataset.year == 2014) & (dataset.month <= 4))]

        if os.path.exists(os.path.join(self.data_dir, 'tr.csv')):
            tr = pd.read_csv(os.path.join(self.data_dir, 'tr.csv'))
        else:
            tr = self.dataset[(dataset.year == 2014) & (dataset.month > 4) & (dataset.month <= 10)]

        if os.path.exists(os.path.join(self.data_dir, 'te.csv')):
            te = pd.read_csv(os.path.join(self.data_dir, 'te.csv'))
        else:
            te = self.dataset[(dataset.year == 2014) & (dataset.month > 11) & (dataset.month <= 12)]
        return ho, tr, te

    def GenerateFeatureCounter(self):
        self.train_count_dict = generate_feature_counter(self.holdout_set, Featured_Dataset.CATEGORICAL_FEATURES,
                                                         self.dict_dir, 'train')
        self.test_count_dict = generate_feature_counter(pd.concat([self.holdout_set, self.train_set]), Featured_Dataset.CATEGORICAL_FEATURES,
                                                        self.dict_dir, 'test')

    def Expanding(self, sample_reduction = False):
        if not set(['holdout%s.csv' % i for i in range(100)]) < set(os.listdir(os.path.join(self.model_input_dir), 'holdout')):
            expanding(self.holdout_set, 'holdout', sample_reduction = sample_reduction)

        if not set(['train%s.csv' % i for i in range(100)]) < set(os.listdir(os.path.join(self.model_input_dir), 'train')):
            expanding(self.train_set, 'train', sample_reduction = sample_reduction)

        if not set(['test%s.csv' % i for i in range(100)]) < set(os.listdir(os.path.join(self.model_input_dir), 'test')):
            expanding(self.test_set, 'test', sample_reduction = sample_reduction)

    def GenerateLibffmData_K(self, k):
        generate_libffm_kth_model(pd.read_csv(os.path.join(self.model_input_dir, 'holdout', 'holdout%s.csv' % k)),
                                  self.dict_dir, Featured_Dataset.CATEGORICAL_FEATURES, k, output_file = os.path.join(self.libffm_dir, 'tmp_holdout_input.txt'))

        generate_libffm_kth_model(os.path.join(self.model_input_dir, 'train', 'train%s.csv' % k), self.dict_dir,
                                  Featured_Dataset.CATEGORICAL_FEATURES, k, output_file = os.path.join(self.libffm_dir, 'tmp_train_input.txt'))

        generate_libffm_kth_model(os.path.join(self.model_input_dir, 'test', 'test%s.csv' % k), self.dict_dir,
                                  Featured_Dataset.CATEGORICAL_FEATURES, k, output_file = os.path.join(self.libffm_dir, 'tmp_test_input.txt'))

        generate_libffm_kth_model(pd.concat(pd.read_csv(os.path.join(self.model_input_dir, 'holdout', 'holdout%s.csv' % k)),
                                            pd.read_csv(os.path.join(self.model_input_dir, 'train', 'train%s.csv' % k))),
                                  self.dict_dir, Featured_Dataset.CATEGORICAL_FEATURES, k, output_file = os.path.join(self.libffm_dir, 'tmp_holdout_train_input.txt'))

    def AddLibffmFeature(self, k):
        add_libffm_feature(self.model_input_dir, self.libffm_dir, 'train', k)
        add_libffm_feature(self.model_input_dir, self.libffm_dir, 'test', k)

    def AddOtherFeatures(self):
        # count features
        add_count_features(self.dict_dir, self.model_input_dir, Featured_Dataset.CATEGORICAL_FEATURES)
        # historical book/click ratios, it should run for a while
        for idx, same_attribute in enumerate(Featured_Dataset.SAME_ATTRIBUTES):
            train_feature_dict = create_feature_dict(self.holdout_set, same_attribute)
            test_feature_dict = create_feature_dict(pd.concat(self.holdout_set, self.train_set), same_attribute)
            add_historical_book_click_feature(train_feature_dict, test_feature_dict, self.model_input_dir, same_attribute, idx)
        # add destination features
        add_destination_features(self.data_dir, self.model_input_dir, ndim = 3)

### main
dataset = pd.read_csv('../data/train_samples.csv')
data = Featured_Dataset(dataset)
data.GenerateFeatureCounter()
data.Expanding()

# prepare LibFFM features
for k in range(100):
    # create libffm format data for cluster k for holdout, train, test and holdout+train
    # the reason is we need to fit two models: one fit on holdout set, and predict on train set; the other fit on holdout+train and predict on test set
    data.GenerateLibffmData_K(k)
    # fit the first model
    os.system('../libffm/ffm-train -t 20 %s libffm_model1' % os.path.join(data.libffm_dir, 'tmp_holdout_input.txt'))
    # fit the second model
    os.system('../libffm/ffm-train -t 20 %s libffm_model2' % os.path.join(data.libffm_dir, 'tmp_holdout_train_input.txt'))
    # predict for the train set with libffm_model1
    os.system('../libffm/ffm-predict %s %s %s' % (os.path.join(data.libffm_dir, 'tmp_train_input.txt'),
                                                  os.path.join(data.libffm_dir, 'libffm_model1'),
                                                  os.path.join(data.libffm_dir, 'tmp_train_output.txt')))
    # predict for the test set with libffm_model2
    os.system('../libffm/ffm-predict %s %s %s' % (os.path.join(data.libffm_dir, 'tmp_test_input.txt'),
                                                  os.path.join(data.libffm_dir, 'libffm_model2'),
                                                  os.path.join(data.libffm_dir, 'tmp_test_output.txt')))
    # add LibFFM feature to the model input
    data.AddLibffmFeature(k)


# add other features: count features, destination features, historical book/click ratios
data.AddOtherFeatures()

# Transfer the training set and test set into libsvm format to feed XGBoost

