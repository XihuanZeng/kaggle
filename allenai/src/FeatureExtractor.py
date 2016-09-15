import scipy
import sys
import re
import gc
import os
import numpy as np
from Parser import SimpleWordParser
from bs4 import BeautifulSoup as bs
from HtmlReader import HtmlReader
from TextReader import TextReader
from WikiCorpusBuilder import WikiCorpusBuilder
from LuceneCorpus import LuceneCorpus
from nltk.stem.snowball import EnglishStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import PorterStemmer
from IO import load_from_pkl, save_to_pkl, create_dirs
from AnswerFunc import *


import lucene
from java.io import File, StringReader
from org.apache.lucene.analysis.core import WhitespaceAnalyzer
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, StoredField, StringField, TextField
from org.apache.lucene.search.similarities import BM25Similarity
from org.apache.lucene.index import IndexWriter, IndexWriterConfig, DirectoryReader, MultiFields, Term
from org.apache.lucene.queryparser.classic import MultiFieldQueryParser, QueryParser
from org.apache.lucene.search import BooleanClause, IndexSearcher, TermQuery
from org.apache.lucene.store import MMapDirectory, SimpleFSDirectory
from org.apache.lucene.util import BytesRefIterator, Version



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
    train['q_num_words'] = np.array([len(parser.parse(qst)) for qst in train['question']])
    train['a_num_words'] = np.array([np.mean([len(parser.parse(ans)) for ans in anss]) for anss in np.array(train[['answerA','answerB','answerC','answerD']])])



class FeatureExtractor(object):
    '''
    This is the main class that runs the various search functions and prepares the features.
    Each feature is a score (or value) for the relevant question,answer pair.
    '''

    def __init__(self, base_dir, recalc=False, norm_scores_default=False, print_level=1):
        """
        :param base_dir:
        :param recalc:
        :param norm_scores_default:
        :param print_level:
        :return:
        """
        self.base_dir = base_dir
        self.recalc = recalc
        self.norm_scores_default = norm_scores_default
        self.print_level = print_level

    def _words_to_names(self, words):
        names = []
        for word in words:
            if len(word) == 0:
                return ''
            names.append(word[0].upper() + word[1:])
        return names

    def prepare_word_sets(self, corpus_dir, train_b, valid_b, test_b):
        if self.print_level > 0:
            print '-> Preparing word sets'
        word_sets_file = '%s/word_sets.pkl' % corpus_dir
        print (word_sets_file)
        # if not exist, will create from traning set and store
        # word_sets contains all one gram and two grams after removing stopwords
        self.word_sets = load_from_pkl(word_sets_file)
        if self.word_sets is None:
            # Prepare list of words (and pairs) that appear in training set
            # note that if tuples = [1], then parser,parse('one two three') -> ['one', 'two', 'three]
            # if tuples = [2], then parser.parse('one two three') -> ['one two', 'two three']
            # if tuples = [1,2], then parser,parse('one two three) -> ['one', 'two', 'three', 'one two', 'two three']
            parser = SimpleWordParser(tuples=[1,2])
            words = set()
            for exam in [train_b, valid_b, test_b]:
                if exam is not None:
                    words.update(np.concatenate([self._words_to_names(parser.parse(qst)) for qst in exam['question']]))
                    words.update(np.concatenate([self._words_to_names(parser.parse(ans)) for ans in exam['answer']]))
            words.difference_update(['']) # ignore empty word
            words = sorted(words)
            if self.print_level > 1:
                print '%d word sets: ...%s...' % (len(words), words[::5000])
            self.word_sets = words
            save_to_pkl(word_sets_file, self.word_sets)

    def prepare_ck12html_corpus(self, corpus_dir):
        self.ck12html_corpus = '%s/CK12/OEBPS/ck12.txt' % corpus_dir
        if not os.path.exists(self.ck12html_corpus):
            # Doc per HTML section (h1-4)
            htmlr = HtmlReader(min_chars_per_line=1, min_words_per_section=20)
            locdic = htmlr.read(htmldir='%s/CK12/OEBPS' % corpus_dir,
                                outfile=self.ck12html_corpus,
                                ignore_sections=set(['explore more.*', 'review', 'practice', 'references']),
                                stop_words=None, pos_words=set([]), corpus_words=None,
                                min_pos_words_in_page_name=0, min_pos_words_in_section=0, action='write')


    def prepare_ck12text_corpus(self, corpus_dir):
        self.ck12text_corpus = '%s/CK12/ck12_text.txt' % corpus_dir
        if not os.path.exists(self.ck12text_corpus):
            textr = TextReader(min_chars_per_line=1, min_words_per_section=25)
            locdic = textr.read(dir='%s/CK12' % corpus_dir,
                                outfile=self.ck12text_corpus,
                                # see "Peoples-Physics-Book-Basic_b_v10_zgo_s1.text" for instance
                                # each chapter is begin with 'CHAPTER'
                                first_line_regexp='^(CHAPTER)',
                                action='write')

    def prepare_simplewiki_corpus(self, corpus_dir, train_b, valid_b):
        # some explanations of the parameters, note that by modifying these numbers, you get different wiki corpus
        # here for simplicity, I only show one possible combination
        # common_words_min_frac = 1.0, meaning no words are treated as common words
        # uncommon words_max_frac = 0,05 meaning there are 403716 uncommon words in this setting
        # min_pos_words_in_page_name=0 meaning an eligible page must have 0 pos words(words that appears in train_b and valid_b), cuz we only want relevant wiki pages
        # min_pos_words_in_section=5 meaning eligible section must have 5 pos words
        # use_all_pages_match_pos_word=True
        # use_all_pages_match_answer=True
        # always_use_first_section=True
        self.simplewiki_corpus = '%s/simplewiki/simplewiki_1.0000_0.0500_0_5_True_True_True_corpus.txt' % corpus_dir
        if not os.path.exists(self.simplewiki_corpus):
            wkb = WikiCorpusBuilder(wiki_name='simplewiki', wiki_dir='%s/simplewiki'%corpus_dir,
                                    wiki_file='simplewiki-20151102-pages-articles.xml', debug_flag=False)
            # Create 2 files all_categories.pkl and parent_categories.pkl, if exist, will just load,
            # They are stored in wkb.all_categories and wkb.parent_categories
            # we scan the wiki file find all categories that has <title>Categories:xxx</title> and their parent Catetories
            # details can be found in read_categories method in WikiReader.py
            wkb.read_categories(reread=False)
            # Create 2 files 'use_categories.pkl' and 'pages_in_categories.pkl'
            # for all singlewiki corpus, target_categories = None, important_categories=['Earth', 'Cellular respiration', 'DNA', 'Units of length', 'History of science',
            #                                                           'Evolutionary biology', 'Nonmetals', 'Health', 'Charles Darwin']
            # important_categories are science-related categories, if not found in target_catefories, which is generated from above method, will give an alert
            # it will all read_pages_in_categories in Cardal_WikiReader.py
            wkb.read_pages_in_categories(target_categories=None, max_cat_depth=9999,
                                         important_categories=['Earth', 'Cellular respiration', 'DNA', 'Units of length', 'History of science',
                                                               'Evolutionary biology', 'Nonmetals', 'Health', 'Charles Darwin'], reread=False)
            # this will read all the text from wiki file
            # parse useful pure text and build a dict of words
            # depends on the common words and uncommon words fraction, we pick up common words and uncommon words
            # we also add common words to stop words
            # we finally save common_words.pkl, uncommon_words.pkl and stop_words.pkl to corpus dir
            wkb.find_common_words(wiki_common_words_min_frac=1.0, wiki_uncommon_words_max_frac=0.05, use_wiki_stop_words=False, reread=False)
            # note that wkb.create_corpus function returns a value, this is just the location of corpus name
            # this will create the corpus file as well as exams_words.pkl(all words that appear in train_b and valid_b),
            # positive_words.pkl(all words in exam that are also uncommon in wiki),
            # and all_answers.pkl(this is a set, each element is a tuple of words within that answer)
            self.simplewiki_corpus = wkb.create_corpus(train_b, valid_b, min_pos_words_in_page_name=0, min_pos_words_in_section=5,
                                                        only_first_section_per_page=False, use_all_pages_match_pos_word=True, use_all_pages_match_answer=True,
                                                        always_use_first_section=True, max_read_lines=9990000000, reread=False)


    def prepare_lucene_indexes(self, corpus_dir):
        self.lucene_dir1, self.lucene_parser1, self.lucene_corpus1 = None, None, None
        self.lucene_dir2, self.lucene_parser2, self.lucene_corpus2 = None, None, None
        self.lucene_dir3, self.lucene_parser3, self.lucene_corpus3 = None, None, None

        # Lucene Index 1: ck12html
        self.lucene_dir1 = '%s/lucene_idx1' % corpus_dir
        self.lucene_parser1 = SimpleWordParser(word_func=EnglishStemmer().stem, split_words_regexp='[\-\+\*\/\,\;\:\(\)]', min_word_length=1)
        self.lucene_corpus1 = LuceneCorpus(index_dir=self.lucene_dir1, filenames=[self.ck12html_corpus], parser=self.lucene_parser1)
        if not os.path.exists(self.lucene_dir1):
             self.lucene_corpus1.prp_index()

        # Lucene Index 2: ck12text
        self.lucene_dir2 = '%s/lucene_idx2' % corpus_dir
        self.lucene_parser2 = SimpleWordParser(word_func=LancasterStemmer().stem, split_words_regexp='[\-\+\*\/\,\;\:\(\)]', min_word_length=1)
        self.lucene_corpus2 = LuceneCorpus(index_dir=self.lucene_dir2, filenames=[self.ck12text_corpus], parser=self.lucene_parser2)
        if not os.path.exists(self.lucene_dir2):
             self.lucene_corpus2.prp_index()

        # Lucene Index 3: simplewiki
        self.lucene_dir3 = '%s/lucene_idx3' % corpus_dir
        self.lucene_parser3 = SimpleWordParser(word_func=PorterStemmer().stem, split_words_regexp='[\-\+\*\/\,\;\:\(\)]', min_word_length=1)
        self.lucene_corpus3 = LuceneCorpus(index_dir=self.lucene_dir3, filenames=[self.simplewiki_corpus],
                                           parser=self.lucene_parser3, similarity=BM25Similarity())
        if not os.path.exists(self.lucene_dir3):
             self.lucene_corpus3.prp_index()


    def prepare_features(self, dataf_q, dataf_b, train_df, cache_dir):
        """
        :param dataf_q: this is not used in my implementation
        :param dataf_b:
        :param train_df: note that we create word counter for training set only, that's why we need to pass this even we are creating features for valid set
        :param cache_dir:
        :return:

        Basic Features:
        1. AnswerInQuestionFunc(): calculate the set of all words of parsed from question, for each ans, calculate the fraction of
        (# intersection of answer words and question words) / (# of total answer words). A little variation is when we parse, we can set
        word stemming on

        2. AnswersInAnswersFunc(): calculate for each answer, the avg ratio of its words appears in other questions. Will not use this feature

        3. AnswerCountFunc(count_type='count', parser): it will use the parser to parse all the answer(if use_question = True, will do the same for questions),
           after all ans are parsed, this will build a counter dict for all the unique words, for each ans, the mean count of words in that ans is calculated as feature

        4. AnswerCountFunc(count_type='correct', parser): same as above, but here only words from correct answers are used to build counter dict

        5. AnswersLengthFunc(log_flag=False): this will gives relative length(# of char) of each ans to mean(answers for the same question)

        Lucene Feature:
        6. AnswersLuceneSearchFunc(lucene_corpus, max_docs, weight_func, score_func): for each ans, we search qst+ans in lucene_corpus, retrievel max_docs number
        of documents, for the score returned by each documents, we can apply an optional score_func as transformation, then we use weight_func to weight
        all these scores and sum as the feature of this ans
        """
        self.cache_dir = '%s/%s' % (self.base_dir, cache_dir)
        create_dirs([self.cache_dir])
        stemmer1 = PorterStemmer()
        stem1 = stemmer1.stem
        check_same_question = not set(dataf_b['ID']).isdisjoint(train_df['ID'])
        stemmed_parser  = SimpleWordParser(word_func=stem1, ignore_special_words=True , min_word_length=1)

        func_name = 'ans_in_qst_stem'
        self.add_answer_func(dataf_b, func=AnswersInQuestionFunc(parser=stemmed_parser), name=func_name)

        func_name = 'ans_in_ans_stem'
        self.add_answer_func(dataf_b, func=AnswersInAnswersFunc(parser=stemmed_parser), name=func_name)

        func_name = 'ans_words_stem_count'
        self.add_answer_func(dataf_b, func=AnswerCountFunc(train_df, check_same_question=check_same_question, count_type='count', parser=stemmed_parser, single_words=True), name=func_name)

        func_name = 'ans_length_ratio'
        self.add_answer_func(dataf_b, func=AnswersLengthFunc(log_flag=False), name=func_name)

        func_name = 'ans_num_words'
        self.add_answer_func(dataf_b, func=AnswersNumWordsFunc(), name=func_name)

        func_name = 'luc_stem_1'
        self.add_answer_func(dataf_b,
                             func=AnswersLuceneSearchFunc(lucene_corpus=self.lucene_corpus1, parser=self.lucene_parser1,
                             max_docs=250, weight_func=lambda n: 1.0/(5.0+np.arange(n))**2, score_func=lambda s: (s/10.0)**3,
                             norm_scores= True),
                             name= func_name)

        func_name = 'luc_stem_2'
        self.add_answer_func(dataf_b,
                             func=AnswersLuceneSearchFunc(lucene_corpus=self.lucene_corpus2, parser=self.lucene_parser2,
                             max_docs=250, weight_func=lambda n: 1.0/(5.0+np.arange(n))**2, score_func=lambda s: (s/10.0)**3,
                             norm_scores= True),
                             name= func_name)

        func_name = 'luc_stem_3'
        self.add_answer_func(dataf_b,
                             func=AnswersLuceneSearchFunc(lucene_corpus=self.lucene_corpus3, parser=self.lucene_parser3,
                             max_docs=250, weight_func=lambda n: 1.0/(5.0+np.arange(n))**2, score_func=lambda s: (s/10.0)**3,
                             norm_scores= True),
                             name= func_name)

    def _cache_filename(self, fname):
        return '%s/%s.pkl' % (self.cache_dir, fname)

    def _read_from_cache(self, fname):
        filename = self._cache_filename(fname)
        #print 'Loading from cache %s' % filename
        return load_from_pkl(filename)

    def _save_to_cache(self, fname, data):
        filename = self._cache_filename(fname)
        print 'Saving to cache %s' % filename
        return save_to_pkl(filename, data)

    def _is_in_cache(self, name):
        if self.cache_dir is None:
            return False
        exists = True
        if np.isscalar(name):
            exists = os.path.exists(self._cache_filename(name))
        else:
            for n in name:
                exists = exists and os.path.exists(self._cache_filename(n))
        return exists

    def add_answer_func(self, train_b, func, name, question_ids=None):
        '''
        Run a score function on each set of question and answers
        '''
        if (not self.recalc) and (self.cache_dir is not None) and (self._is_in_cache(name)):
            if np.isscalar(name):
                train_b[name] = self._read_from_cache(name)
            else:
                for n in name:
                    train_b[n] = self._read_from_cache(n)
            return

        groups = train_b.groupby('ID').groups
        for i,(idx,inds) in enumerate(groups.iteritems()):
            assert len(set(train_b.irow(inds)['question']))==1
            if (question_ids is not None) and (idx not in question_ids): continue
            question = train_b.iloc[inds[0]]['question']
            answers = np.array(train_b.iloc[inds]['answer'])
            if 'correct' in train_b.columns:
                print '\n-----> #%d : correct = %s' % (i, ', '.join(['%d'%c for c in np.array(train_b.iloc[inds]['correct'])]))
                sys.stdout.flush()
            vals = func(question, answers)
            if question_ids is not None:
                print 'vals = %s' % str(vals)
            for val,ind in zip(vals, inds):
                if np.isscalar(val):
                    train_b.set_value(ind, name, val)
                else:
                    assert len(val)==len(name)
                    for v,n in zip(val,name):
                        train_b.set_value(ind, n, v)

        if (self.cache_dir is not None) and (question_ids is None):
            if np.isscalar(name):
                self._save_to_cache(name, np.array(train_b[name]))
            else:
                for n in name:
                    self._save_to_cache(n, np.array(train_b[n]))






