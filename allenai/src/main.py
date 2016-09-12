import sys
import json
import pandas as pd
import numpy as np
import os
from IO import read_input_file, load_from_pkl, save_to_pkl
from DataPreparation import sub_complex_answers, prp_binary_dataf
from FeatureExtractor import FeatureExtractor, add_qa_features

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

# prepare corpus features
# Here we suppose all the raw data is all stored in corpus dir, the raw data is:
# wiki: enwiki-20160113-pages-articles.xml
# ck12: OEBPS dir that contains files extracted from Concepts_b_v8_vdt.epub
# ck12: CK-12-Biology-Concepts_b_v143_e4x_s1.text: the downloaded version is pdf, use online converter generate text
# ck12: CK-12-Chemistry-Basic_b_v143_vj3_s1.text
# ck12: CK-12-Earth-Science-Concepts-For-High-School_b_v114_yui_s1.text
# ck12: CK-12-Life-Science-Concepts-For-Middle-School_b_v126_6io_s1.text
# ck12: CK-12-Physical-Science-Concepts-For-Middle-School_b_v119_bwr_s1.text
# ck12: CK-12-Physics-Concepts-Intermediate_b_v56_ugo_s1.text
data_pkl_file = None
norm_scores_default = False
if data_pkl_file is None:
    fext = FeatureExtractor(base_dir = base_dir, recalc = False, norm_scores_default = norm_scores_default, print_level = 2)

    # prepare word set, which is to derive all the unique 1-gram and 2-gram from train, valid and test
    fext.prepare_word_sets(corpus_dir = corpus_dir, train_b = train_b, valid_b = None, test_b = None)

    # prepare ck12html corpus: this function will go into OEBPS dir, find all the x.html file where x is a number
    # extract all the text while ignore sections such as 'explore more', 'review', 'practice', 'references'
    fext.prepare_ck12html_corpus(corpus_dir = corpus_dir)

    #
    fext.prepare_ck12text_corpus(corpus_dir = corpus_dir)






