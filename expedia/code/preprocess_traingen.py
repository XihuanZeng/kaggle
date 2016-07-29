# this script will enlarge tr.csv and te.csv 100 times.
# for each event(line) in tr/te.csv, this will be expand to 100 event, one for each hotel_cluster.
# each has the same feature but the response is whether the ground truth hotel_cluster is the same as the hotel_cluster of this event
# for each of the 100 set generated from tr.csv, we keep only 10% of the 0 of the new response

import pandas as pd
import json

tr = pd.read_csv('../data/tr.csv')

for i in

