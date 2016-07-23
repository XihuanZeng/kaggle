# sample 10,000 users randomly to conduct pilot study

import os
import numpy as np
import pandas as pd
import random

from utils import get_attributes


def main():
    random.seed(1991)
    datadir = os.path.join('../data')
    sample_size = 10000
    #train_set = pd.read_csv(os.path.join(datadir, 'train.csv'))
    #test_set = pd.read_csv(os.path.join(datadir, 'test.csv'))

    

    users = train_set['user_id'].unique()
    users = random.sample(users, sample_size)
    np.save(os.path.join(datadir, 'split_data', 'users'), users)

    train_set[train_set.user_id.isin(users)].to_csv(os.path.join(datadir, 'split_data', 'train.csv'))
    test_set[test_set.user_id.isin(users)].to_csv(os.path.join(datadir, 'split_data', 'test.csv'))

if __name__ == "__main__":
    main()