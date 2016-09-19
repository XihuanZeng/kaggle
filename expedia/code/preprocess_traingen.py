# this script will enlarge tr.csv and te.csv 100 times.
# for each event(line) in tr/te.csv, this will be expand to 100 event, one for each hotel_cluster.
# each has the same feature but the response is whether the ground truth hotel_cluster is the same as the hotel_cluster of this event

import os
import pandas as pd
import json
import random
from collections import defaultdict

def expanding(data, stage, model_input_dir, sample_reduction = True):
    for i in range(100):
        tmp = data[data.is_booking == 1].copy()
        tmp['y'] = [1 if hotel_cluster == i else 0 for hotel_cluster in tmp['hotel_cluster']]
        if sample_reduction:
            positives = tmp['y'][tmp.y == 1].index.tolist()
            negatives = tmp['y'][tmp.y == 0].index.tolist()
            tmp = tmp.loc[positives + random.sample(negatives, int(0.1 * len(negatives)))]
        tmp.to_csv(os.path.join(model_input_dir, '%s' % stage, '%s%s.csv' % (stage, i)))
