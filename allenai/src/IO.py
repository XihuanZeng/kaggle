import cPickle
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

