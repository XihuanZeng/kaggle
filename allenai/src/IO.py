import cPickle
import os
import pandas as pd

def read_input_file(path, max_rows=999999999, index_col=0, sep=','):
    """
    :param path: path to the tsv file
    :param max_rows: max number of rows to read
    :param index_col: which column is the index column
    :param sep: separator
    :return:
    """
    dataf = pd.read_table(path, index_col=index_col, nrows=max_rows, sep=sep)
    if 'correctAnswer' in dataf.columns:
        dataf = dataf[[(ca in ['A','B','C','D']) for ca in dataf['correctAnswer']]]
    dataf['ID'] = dataf.index
    return dataf

def save_to_pkl(filename, data):
    """
    :param filename: path to pkl file
    :param data:
    :return:
    """
    with open(filename, "wb") as f:
        cPickle.dump(data, f, cPickle.HIGHEST_PROTOCOL)

def load_from_pkl(filename):
    if not os.path.exists(filename):
        return None
    with open(filename, "rb") as f:
        data = cPickle.load(f)
    return data

def create_dirs(dirs):
    '''
    Make sure the given directories exist. If not, create them
    '''
    for dir in dirs:
        if not os.path.exists(dir):
            print 'Creating directory %s' % dir
            os.mkdir(dir)