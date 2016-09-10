from sklearn import preprocessing
from sklearn import preprocessing
import numpy as np
from nltk import word_tokenize
from Parser import SimpleWordParser

def add_qa_features(train):
    '''
    Add simple features computed for each question
    These features are:
    1. Does the question contains 'which'
    2. Does the question contains '___'
    3. Does the question contains 'not', 'except', 'least'
    4. Number of words in question
    5. Average Number of words for answers of this question
    '''
    parser = SimpleWordParser()
    train['q_which']     = np.array([('which' in qst.lower().split(' ')) for qst in train['question']])
    train['q____']       = np.array([('___' in qst) for qst in train['question']])
    not_words_weights = {'NOT':1, 'EXCEPT':1, 'LEAST':1}    # note the 'not' words can have unequal weights
    train['q_not']       = np.array([np.max([not_words_weights.get(w,0) for w in qst.split(' ')]) for qst in train['question']])
    train['q_num_words'] = np.array([parser.parse(qst) for qst in train['question']])
    train['a_num_words'] = np.array([np.mean([len(parser.parse(ans)) for ans in anss]) for anss in np.array(train[['answerA','answerB','answerC','answerD']])])


