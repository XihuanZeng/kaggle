import sys
import json
import pandas as pd
import numpy as np
import os
from IO import read_input_file
from DataPreparation import sub_complex_answers
from FeatureExtractor import add_qa_features

# set path
base_dir = os.path.join('..')
corpus_dir = os.path.join('../corpus')
input_dir = os.path.join('../input')
train_file = os.path.join(input_dir, 'training_set.tsv')
validate_file = os.path.join(input_dir, 'training_set.tsv')
test_file = os.path.join(input_dir, 'test_set.tsv')

# read data(train, valid, test) and add trivial features
train_q = read_input_file(train_file, sep='\t', max_rows=1000000)
sub_complex_answers(train_q)
add_qa_features(train_q)

valid_q = read_input_file(test_file, sep='\t' , max_rows=1000000)
sub_complex_answers(valid_q)
add_qa_features(valid_q)

test_q = read_input_file(test_file, sep='\t' , max_rows=1000000)
sub_complex_answers(test_q)
add_qa_features(test_q)

# provide binary features
train_b = prp_binary_dataf(train_q)
valid_b = prp_binary_dataf(valid_q)
test_b = prp_binary_dataf(test_q)




import nltk
asc = []
for q in train_q.question:
    try:
        a = nltk.word_tokenize(q)
    except:
        asc.append(q)

AsciiConvertor.convert(asc[0])
print asc[0]
asc[0]

from nltk.corpus import stopwords
len(stopwords.words('english'))