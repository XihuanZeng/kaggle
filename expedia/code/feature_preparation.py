import math
import os
import json
import pickle
import pandas as pd
import numpy as np

from preprocess_featurecounter import generate_feature_counter




class Featured_Dataset():
    CATEGORICAL_FEATURES = ['site_name', 'posa_continent', 'user_location_country', 'user_location_region', 'user_location_city',
                            'is_mobile', 'is_package', 'channel', 'srch_destination_id', 'hotel_continent', 'hotel_country',
                            'hotel_market']

    def __init__(self, dataset, data_dir, dict_dir):
        """
        :param dataset: this is the sampled dataset
        :return:
        """
        self.dataset = dataset
        self.dict_dir = dict_dir
        self.data_dir = data_dir
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
        
    def Expanding(self):
        pass


# main
dataset = pd.read_csv('../data/train_samples.csv')
data = Featured_Dataset(dataset)
data.GenerateFeatureCounter()