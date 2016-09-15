import os
import numpy as np
from IO import read_input_file, load_from_pkl
from DataPreparation import sub_complex_answers, prp_binary_dataf
from FeatureExtractor import FeatureExtractor, add_qa_features
from sklearn.linear_model import LogisticRegression

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

    # prepare ck12html corpus: this function will go into CK12/OEBPS dir, find all x.html file where x is a number
    # extract all the text while ignore sections such as 'explore more', 'review', 'practice', 'references'
    fext.prepare_ck12html_corpus(corpus_dir = corpus_dir)

    # prepare ck12text corpus: this function will go into CK12 dir, find all .text file, which are 6 textbooks
    # extract relevant text from all Chapters of each book
    fext.prepare_ck12text_corpus(corpus_dir = corpus_dir)

    # prepare simplewiki corpus: this function will go into simplewiki dir, find the simplewiki-20151102-pages-articles.xml
    # extract text from all categories found if the page contains at least some uncommon words from train_b and test_b
    fext.prepare_simplewiki_corpus(corpus_dir, train_b, valid_b)

    # prepare Lucene indexing: this will create Lucene indexing in lucene_idx[1-3] for the corpus created by previous functions
    fext.prepare_lucene_indexes(corpus_dir = corpus_dir)

    # generate features for the train, valid and test/
    # there are 2 types of features:
    # 1. Basic feature that only looks at the dataset
    # 2. Lucene features that returns the score produced by Lucene index
    # prepare basic features
    fext.prepare_features(dataf_q=train_q, dataf_b=train_b, train_df=train_b, cache_dir='funcs_train')
    fext.prepare_features(dataf_q=valid_q, dataf_b=valid_b, train_df=train_b, cache_dir='funcs_valid')
    fext.prepare_features(dataf_q=test_q, dataf_b=test_b, train_df=train_b, cache_dir='funcs_test')

# train the data with Logistic Regression
model = LogisticRegression()
train_cache_dir = os.path.join(base_dir, 'funcs_train')
for file in os.listdir(train_cache_dir):
    if file.endswith('pkl'):
        train_b[file[:-4]] = load_from_pkl(os.path.join(base_dir, 'funcs_train', file))
model.fit(train_b[[x for x in train_b.columns if x not in ['ID', 'answer', 'question', 'correct', 'q_num_words', 'ans_name']]], train_b['correct'])

# predict answer for the test question
test_cache_dir = os.path.join(base_dir, 'funcs_test')
for file in os.listdir(test_cache_dir):
    if file.endswith('pkl'):
        test_b[file[:-4]] = load_from_pkl(os.path.join(base_dir, 'funcs_test', file))
raw_result = model.predict_proba(test_b[[x for x in train_b.columns if x not in ['ID', 'answer', 'question', 'correct', 'q_num_words', 'ans_name']]])
a = map(lambda x: x[1], raw_result)
b = np.array(a)
b = b.reshape((len(a)/4, 4))
result_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
result = map(lambda x: result_dict[np.argmax(x)], b)

# save result as text
with open(os.path.join(base_dir, 'submission.txt'), 'wt') as f:
    for i in result:
        f.write(str(i) + '\n')
f.close()