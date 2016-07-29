# this script implement a rule-based method. to be specific, when a query is coming, we first check srch_destination_id and
# use the most popular 5 hotel cluster of that srch_destination_id as the prediction.
# if we cannot get 5, then we move to hotel_market, then hotel_country, then 5 most popular hotel cluster for whole dataset
# we will use the magic ratio 3:17, i,e a click count as 3 and a book count as 17

import pandas as pd
from collections import defaultdict
from ml_metrics import mapk
from heapq import nlargest


ho = pd.read_csv('../data/ho.csv')
tr = pd.read_csv('../data/tr.csv')
te = pd.read_csv('../data/te.csv')

dataset = pd.concat([ho, tr])
best_srch_destination_id = defaultdict(lambda: defaultdict(int))
best_hotel_market = defaultdict(lambda: defaultdict(int))
best_hotel_country = defaultdict(lambda: defaultdict(int))
best_hotel = defaultdict(int)


for index, row in dataset.iterrows():
    srch_destination_id = row['srch_destination_id']
    is_booking = row['is_booking']
    hotel_market = row['hotel_market']
    hotel_country = row['hotel_country']
    hotel_cluster = row['hotel_cluster']
    score = 3 + 17 * is_booking
    if srch_destination_id != '':
        best_srch_destination_id[srch_destination_id][hotel_cluster] += score
    if hotel_market != '':
        best_hotel_market[hotel_market][hotel_cluster] += score
    if hotel_country != '':
        best_hotel_country[hotel_country][hotel_cluster] += score
    best_hotel[hotel_cluster] += score

# calculate MAP@5 for the rule-based method
prediction = []
for index, row in te.iterrows():
    srch_destination_id = row['srch_destination_id']
    is_booking = row['is_booking']
    hotel_market = row['hotel_market']
    hotel_country = row['hotel_country']
    top_clusters = nlargest(5, best_srch_destination_id[srch_destination_id],
                            key = best_srch_destination_id[srch_destination_id].get)
    if len(top_clusters) <= 5:
        item = nlargest(5, best_hotel_market[hotel_market],
                        key = best_hotel_market[hotel_market].get)
        for i in item:
            if i not in top_clusters:
                top_clusters.append(i)

    if len(top_clusters) < 5:
        item = nlargest(5, best_hotel_country[hotel_country],
                        key = best_hotel_country[hotel_market].get)
        for i in item:
            if i not in top_clusters:
                top_clusters.append(i)

    if len(top_clusters) < 5:
        item = nlargest(5, best_hotel, key = best_hotel.get)
        for i in item:
            if i not in top_clusters:
                top_clusters.append(i)

    prediction.append(top_clusters[:5])

ground_truth = [[l] for l in te['hotel_cluster']]
print 'the rule-based method has MAP5 %s' % mapk(ground_truth, prediction, k = 5)   # 0.25024279168333224








