import random
import json
import pandas as pd

def main():
    dataset = pd.read_csv('../data/train.csv')
    dataset['date_time'] = pd.to_datetime(dataset['date_time'], errors = 'ignore')
    dataset['year'] = dataset['date_time'].dt.year
    dataset['month'] = dataset['date_time'].dt.month

    # sample 100,000 users
    users = dataset[dataset.year == 2013].user_id.unique()
    users = random.sample(users, 100000)

    # store the selected users
    users_dict = dict()
    for i in range(len(users)):
        users_dict[users[i]] = i
    with open('../data/dict/users_dict.json', 'wb') as f:
        json.dump(users_dict, f)
    f.close()

    # to make life easier, exclude event with unmarked destination
    destinations = pd.read_csv('../data/destinations.csv')
    srch_destination_id = destinations.srch_destination_id
    dataset = dataset[(dataset.user_id.isin(users)) & (dataset.srch_destination_id.isin(srch_destination_id))]
    dataset.to_csv('../data/train_samples.csv')

if __name__ == '__main__':
    main()