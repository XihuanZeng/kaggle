import os
import numpy as np
import argparse
import pandas as pd
import ml_metrics as metrics
from xgboost.sklearn import XGBClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from collections import Counter



parser = argparse.ArgumentParser(description='suffix')
parser.add_argument('--suffix', type = int, default = 0)
args = parser.parse_args()


def load_header(file,type='dict'):
    """
    :param file: path to header file
    :param type: dict or list
    :return: dict or list of header file
    """
    tmp1=open(file,'r')
    tmp2=tmp1.read().strip().split(',')
    if type=='dict':
        return {i:index for index,i in enumerate(tmp2)}
    if type=='list':
        return tmp2

def get_user_info(df):
    """
    :param df: subset of train.csv, booking rows or clicking rows
    :return: if df contains only booking rows, this will create a dict of {user: {hotel_cluster : # of bookings made}}
    """
    user_record={}
    grouped = df.groupby(['user_id'])
    for name, group in grouped:
        user_record[name] = Counter(list(group.hotel_cluster))
    return user_record

def most_popular_n(key, n):
    """
    :param key: attribute name from train_header
    :param n: int
    :return: for instance, if key = 'user_country' and there are 200 countries in total.
             This will return a list of most popular n countries in terms of #booking + #clicking
    """
    return list(train_set[key].value_counts().index[:n])

def ensure_datetime_quality(s):
    """
    :param s: str
    :return: if this str can be covert to date time, s is returned, otherwise 2013-06-01 00:00:00 is returned
    """
    try:
        time = pd.to_datetime(s)
        return s
    except:
        return '2013-06-01 00:00:00'


# path to data files
datadir = os.path.join('../data')
train_data_path = os.path.join(datadir, 'split_data', 'train', '%03d' % args.suffix + '.csv')
test_data_path = os.path.join(datadir, 'split_data', 'test', '%03d' % args.suffix + '.csv')
destination_path = os.path.join(datadir,'destinations.csv')

# load data set: basic operations to get train set, validation set, destinations set
dataset = pd.read_csv(train_data_path)
dataset['date_time'] = pd.to_datetime(map(ensure_datetime_quality, list(dataset.date_time)))
dataset['srch_ci'] = pd.to_datetime(map(ensure_datetime_quality, list(dataset.srch_ci)))
dataset['srch_co'] = pd.to_datetime(map(ensure_datetime_quality, list(dataset.srch_co)))
dataset['year'] = dataset.date_time.dt.year
dataset['month'] = dataset.date_time.dt.month
train_set = dataset[(dataset.year == 2013) | ((dataset.year == 2014) & (dataset.month < 8))]
validation_set = dataset[(dataset.year == 2014) & (dataset.month >= 8)]
validation_set = validation_set[validation_set.is_booking == 1]
destinations_df = pd.read_csv(destination_path)

# get user dictionary: each user corresponds to a 200 dim vec
user_info_df = train_set[['id', 'user_id', 'hotel_cluster','is_booking']]
user_booking_dict = get_user_info(user_info_df[user_info_df.is_booking == 1])
user_clicking_dict = get_user_info(user_info_df[user_info_df.is_booking == 0])

# classes that generate the features
class User_Feature_Generator(BaseEstimator, TransformerMixin):
    """
    This class get 2 columns (user_id, hotel_cluster) from a subset of train,csv.
    It will create a 200-dim vector of each user as the feature binding to that user.
    first 100 elements are counts of bookings on 100 hotel clusters;
    last 100 elements are counts of clickings on 100 hotel clusters.
    """

    def get_transformer_name(self):
        return self.__class__.__name__

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        def get_user_vector(user_id):
            booking_vec = [0] * 100
            clicking_vec = [0] * 100
            try:
                booking_record = user_booking_dict[user_id]
                for cluster_id, count in booking_record.items():
                    booking_vec[cluster_id] = count
            except:
                pass
            try:
                clicking_record = user_clicking_dict[user_id]
                for cluster_id, count in clicking_record.items():
                    clicking_vec[cluster_id] = count
            except:
                pass
            return booking_vec + clicking_vec

        return np.array(map(get_user_vector, list(X.user_id)))


class Destinations_Feature_Generator(BaseEstimator,TransformerMixin):
    """
    This class get input from destinations.csv in the datadir, (optional) will do pca to reduce dimension to ndim
    for each desniation, its corresponding feature will be a vector of length ndim
    """
    def __init__(self, n_components = 10):
        self.n_components = n_components

    def get_transformer_name(self):
        return self.__class__.__name__

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X):
        pca = PCA(n_components = self.n_components)
        dest_small = pca.fit_transform(destinations_df[["d{0}".format(i + 1) for i in range(149)]])
        dest_small = pd.DataFrame(dest_small)
        avg_dest_vector = np.median(np.array(dest_small), axis = 0)
        dest_dict = {}
        for dest_id, dest_vector in zip(destinations_df.srch_destination_id, np.array(dest_small)):
            dest_dict[dest_id] = dest_vector

        def get_dest_vector_dim_reduced(dest_id):
            try:
                return dest_dict[dest_id]
            except:
                return avg_dest_vector

        return np.array(map(get_dest_vector_dim_reduced, X.srch_destination_id))


class OneHot_Feature_Generator(BaseEstimator, TransformerMixin):
    """
    this class generate one-hot features given the key, which is one attribute fron train_header
    """
    def __init__(self, key, num_levels):
        self.key = key
        self.num_levels = num_levels

    def get_transformer_name(self):
        return self.__class__.__name__

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X):
        key_info = X[self.key]
        def OneHot_Encoding(entry):
            feature_vec = [0] * self.num_levels
            try:
                feature_vec[int(entry)] = 1
            except:
                pass
            return feature_vec
        return map(OneHot_Encoding, key_info)


class Feature_Generator(BaseEstimator, TransformerMixin):
    """
    this class simply takes one column from the input as the feature
    """
    def __init__(self, key):
        self.key = key

    def get_transformer_name(self):
        return self.__class__.__name__

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X):
        size = len(X)
        return np.array(X[self.key]).reshape(size, 1)


class Time_Feature_Generator(BaseEstimator, TransformerMixin):
    """
    this class generate the time before checking in, length of stay, season and is_weekday of checking in time as the feature
    time before checking in, lengthy of stay are self-explainable
    season can be overlapping: Spring: Mar-May, Summer: May-Sep, Fall: Aug-Oct, Winter: Nov-Feb
    is_weekday: 1 for Mon-Thu, 0 for Fri-Sun
    """
    def get_transformer_name(self):
        return self.__class__.__name__

    def fit(self, *args, **kwargs):
        return self

    def generate_time_feature(self, row):
        date_time = row.date_time
        srch_ci = row.srch_ci
        srch_co = row.srch_co
        try:
            length_of_stay = (srch_co - srch_ci).days
        except:
            length_of_stay = 2   # this is the median of whole train.csv

        try:
            days_before_check_in = (srch_ci - date_time).days
        except:
            days_before_check_in = 16    # this is the median of whole train.csv

        try:
            season_vec = map(int, [srch_ci.month in i for i in [[3,4,5], [5,6,7,8,9], [8,9,10], [11,12,1,2]]])
        except:
            season_vec = [0, 0, 0, 0]

        try:
            is_weekday = int(srch_ci.weekday() in [1,2,3,4])
        except:
            is_weekday = 0

        return [length_of_stay] + [days_before_check_in] + [is_weekday] + season_vec

    def transform(self, X):
        return np.array([self.generate_time_feature(row) for index,
                         row in X[['date_time', 'srch_ci', 'srch_co']].iterrows()])


class UserLocation_Feature_Generator(BaseEstimator, TransformerMixin):
    """
    this class generate features for each sample point based on user_location_country/region/city
    note that there are lots of levels for each variable.
    by default it take 10 countries, 20 regions, 50 cities
    """
    def __init__(self, num_country = 10, num_region = 20, num_city = 50):
        self.num_country = num_country
        self.num_region = num_region
        self.num_city = num_city

    def get_transformer_name(self):
        return self.__class__.__name__

    def fit(self, *args, **kwargs):
        return self

    def generate_user_location_feature(self, row):
        country = row.user_location_country
        region = row.user_location_region
        city = row.user_location_city
        country_vec = [0] * self.num_country
        region_vec = [0] * self.num_region
        city_vec = [0] * self.num_city
        most_popular_user_location_country = most_popular_n('user_location_country', self.num_country)
        most_popular_user_location_region = most_popular_n('user_location_region', self.num_region)
        most_popular_user_location_city = most_popular_n('user_location_city', self.num_city)
        try:
            country_vec[most_popular_user_location_country.index(country)] = 1
        except:
            pass

        try:
            region_vec[most_popular_user_location_region.index(region)] = 1
        except:
            pass

        try:
            city_vec[most_popular_user_location_city.index(city)] = 1
        except:
            pass
        return country_vec + region_vec + city_vec

    def transform(self, X):
        return np.array([self.generate_user_location_feature(row) for index, row in X[['user_location_country', 'user_location_region', 'user_location_city']].iterrows()])


class HotelLocation_Feature_Generator(BaseEstimator, TransformerMixin):
    """
    this class generate features for each sample point based on hotel_continent, hotel_country and hotel_market
    note that there are lots of levels for each variable.
    by default it takes all 6 continents, 10 countries and 100 hotel markets
    """
    def __init__(self, num_country = 10, num_market = 100):
        self.num_country = num_country
        self.num_market = num_market

    def get_transformer_name(self):
        return self.__class__.__name__

    def fit(self, *args, **kwargs):
        return self

    def generate_hotel_location_feature(self, row):
        continent = row.hotel_continent
        country = row.hotel_country
        market = row.hotel_market
        continent_vec = [0] * 7
        country_vec = [0] * self.num_country
        market_vec = [0] * self.num_market
        most_popular_hotel_country = most_popular_n('hotel_country', self.num_country)
        most_popular_hotel_market = most_popular_n('hotel_market', self.num_market)
        try:
            continent_vec[continent] = 1
        except:
            pass

        try:
            country_vec[most_popular_hotel_country.index(country)] = 1
        except:
            pass

        try:
            market_vec[most_popular_hotel_market.index(market)] = 1
        except:
            pass

        return continent_vec + country_vec + market_vec

    def transform(self, X):
        return np.array([self.generate_hotel_location_feature(row) for index, row in X[['hotel_continent', 'hotel_country', 'hotel_market']].iterrows()])


# create pipeline, use the above classes to generate features, while using a RandomForestClassifier to train the combined features
pipeline = Pipeline([
    ('generate_features', FeatureUnion([
        ('user_feature', User_Feature_Generator()),
        ('dest_feature', Destinations_Feature_Generator()),
        ('user_location_feature', UserLocation_Feature_Generator()),
        ('hotel_location_feature', HotelLocation_Feature_Generator()),
        ('time_feature', Time_Feature_Generator()),
        ('posa_continent_feature', OneHot_Feature_Generator(key = 'posa_continent', num_levels = 5)),
        ('channel_feature', OneHot_Feature_Generator(key = 'channel', num_levels = 11)),
        ('site_name_feature', OneHot_Feature_Generator(key = 'site_name', num_levels = 54)),
        ('srch_destination_type_feature', OneHot_Feature_Generator(key = 'srch_destination_type_id', num_levels = 10)),
        ('is_mobile_feature', Feature_Generator(key = 'is_mobile')),
        ('is_package_feature', Feature_Generator(key = 'is_package')),
        ('srch_rm_cnt_feature', Feature_Generator(key = 'srch_rm_cnt')),
        ('srch_rm_adults_feature', Feature_Generator(key = 'srch_adults_cnt')),
        ('srch_rm_children_feature', Feature_Generator(key = 'srch_children_cnt'))
    ])),
    ('classifier', XGBClassifier())
])


# tunable hyperparameters
parameters = {
    'generate_features__dest_feature__n_components': (80, 90, 100, 120),
    'generate_features__user_location_feature__num_country' : (5, 10, 15, 20),
    'generate_features__user_location_feature__num_region' : (10, 20, 30, 40, 50),
    'generate_features__user_location_feature__num_city' : (10, 20, 30, 40, 50, 70, 100),
    'generate_features__hotel_location_feature__num_country' : (5, 10, 15, 20),
    'generate_features__hotel_location_feature__num_market' : (20, 30, 50, 100),
    'classifier__n_estimators': (100, 200, 300, 400, 500),
    'classifier__max_depth': (5, 8, 10)
}

# random search CV: randomly choose 60 grid points of parameter combination
random_search = RandomizedSearchCV(pipeline, parameters, n_jobs = 4, verbose = 1, n_iter = 60)
random_search.fit(train_set[train_set.is_booking == 1], train_set[train_set.is_booking == 1].hotel_cluster)


# prediction: predict 5 classes with highest probability for each sample using the best model chosen by above CV
def take5(x):
    return list(np.argsort(-x)[:5])

validate_predict = random_search.predict_proba(validation_set)
validate_predict_best5 = map(take5, validate_predict)

# get MAP@5 of validation set
target = [[l] for l in validation_set["hotel_cluster"]]
validate_score = metrics.mapk(target, validate_predict_best5, k=5)
validate_accuracy = accuracy_score(target,  map(np.argmax, validate_predict))
print ('prediction accuracy is : %s' % validate_accuracy)
print ('prediction MAP score is : %s' % validate_score)


# use the best estimator to train the whole dataset
best_est = random_search.best_estimator_
best_est.fit(dataset[dataset.is_booking == 1], dataset[dataset.is_booking == 1].hotel_cluster)
test_set = pd.read_csv(test_data_path)
test_set['date_time'] = pd.to_datetime(map(ensure_datetime_quality, list(test_set.date_time)))
test_set['srch_ci'] = pd.to_datetime(map(ensure_datetime_quality, list(test_set.srch_ci)))
test_set['srch_co'] = pd.to_datetime(map(ensure_datetime_quality, list(test_set.srch_co)))
test_set['year'] = test_set.date_time.dt.year
test_set['month'] = test_set.date_time.dt.month
predict = best_est.predict_proba(test_set)
predict_best5 = map(take5, predict)

# store the result
with open(os.path.join(datadir, 'split_data', 'submission', '%03d' % args.suffix + '.csv'), 'w') as f:
    for i in zip(test_set.id, predict_best5):
        f.write(str(i[0]) + ',' + ' '.join(map(str, i[1])) + '\n')
f.close()

# store the parameters
param = best_est.get_params()
n_estimators = param['classifier__n_estimators']
max_depth = param['classifier__max_depth']
num_dest = param['generate_features__dest_feature__n_components']
hotel_num_country = param['generate_features__hotel_location_feature__num_country']
hotel_num_market = param['generate_features__hotel_location_feature__num_market']
user_num_country = param['generate_features__user_location_feature__num_country']
user_num_region = param['generate_features__user_location_feature__num_region']
user_num_city = param['generate_features__user_location_feature__num_city']

writable = ['%03d' % args.suffix, validate_accuracy, validate_score, n_estimators, max_depth, num_dest,
            hotel_num_country, hotel_num_market, user_num_country, user_num_region, user_num_city]
with open(os.path.join(datadir, 'split_data', 'result', 'result.csv'), 'a') as f:
    f.write(','.join(map(str, writable)) + '\n')