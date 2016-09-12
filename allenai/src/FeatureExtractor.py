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



from IO import load_from_pkl, save_to_pkl

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
                                first_line_regexp='^(C HAPTER)', # see "Peoples-Physics-Book-Basic_b_v10_zgo_s1.text"
                                action='write')

    def prepare_ck12text_sent_corpus(self, corpus_dir):
        self.ck12text_sent_corpus = '%s/CK12/ck12_text_sentences.txt' % corpus_dir
        if not os.path.exists(self.ck12text_sent_corpus):
            textr = SentenceReader(min_chars_per_line=1, min_words_per_section=10)
            locdic = textr.read(dir='%s/CK12' % corpus_dir,
                                outfile=self.ck12text_sent_corpus,
                                sentence_sep='. ',
                                action='write')



    def prepare_corpuses(self, corpus_dir, train_b, valid_b, prp_wiki_corpuses=True):
        '''
        Prepare all the corpus files we shall be using. This needs to be done only once.
        '''
        if self.print_level > 0:
            print '-> Preparing corpuses'

        # Prepare CK-12 HTML corpus
        self.ck12html_corpus = '%s/CK12/OEBPS/ck12.txt' % corpus_dir
        if not os.path.exists(self.ck12html_corpus):
            # Doc per HTML section (h1-4)
            htmlr = HtmlReader(min_chars_per_line=1, min_words_per_section=20)
            locdic = htmlr.read(htmldir='%s/CK12/OEBPS' % corpus_dir,
                                outfile=self.ck12html_corpus,
                                ignore_sections=set(['explore more.*', 'review', 'practice', 'references']),
                                stop_words=None, pos_words=set([]), corpus_words=None,
                                min_pos_words_in_page_name=0, min_pos_words_in_section=0, action='write')

        # Prepare CK-12 HTML paragraph corpus
        # this is almost the same as last one, but section_regexp is slightly different which contains tags <p>...</p>
        # the last one we concatnate different paragraph of same title together, but here we split different paragraphs to different sections under same title
        self.ck12html_para_corpus = '%s/CK12/OEBPS/ck12_paragraphs.txt' % corpus_dir
        if not os.path.exists(self.ck12html_para_corpus):
            # Doc per HTML paragraph
            htmlr = HtmlReader(min_chars_per_line=1, min_words_per_section=25)
            locdic = htmlr.read(htmldir='%s/CK12/OEBPS' % corpus_dir,
                                outfile=self.ck12html_para_corpus,
                                ignore_sections=set(['explore more', 'review', 'references']),
                                section_regexp='(?:<p[^\>]*>)|(?:<h[1-4][^>]*>([^<]+)<[\/]h[1-4]>)',
                                stop_words=None, pos_words=set([]), corpus_words=None,
                                min_pos_words_in_page_name=0, min_pos_words_in_section=0, action='write')

        # Prepare CK-12 text corpus
        # more details see Cardal_TextReader
        # this will create num_textbook pages, each one has around 5000 sections which are text paragraphs
        self.ck12text_corpus = '%s/CK12/ck12_text.txt' % corpus_dir
        if not os.path.exists(self.ck12text_corpus):
            textr = TextReader(min_chars_per_line=1, min_words_per_section=25)
            locdic = textr.read(dir='%s/CK12' % corpus_dir,
                                outfile=self.ck12text_corpus,
                                first_line_regexp='^(C HAPTER)', # see "Peoples-Physics-Book-Basic_b_v10_zgo_s1.text"
                                action='write')


        # Prepare CK-12 text sentences corpus
        # more details see Cardal_SentenceReader
        # this is almost the same as above, but to make each sentence(if satisfy some conditions) as a section
        self.ck12text_sent_corpus = '%s/CK12/ck12_text_sentences.txt' % corpus_dir
        if not os.path.exists(self.ck12text_sent_corpus):
            textr = SentenceReader(min_chars_per_line=1, min_words_per_section=10)
            locdic = textr.read(dir='%s/CK12' % corpus_dir,
                                outfile=self.ck12text_sent_corpus,
                                sentence_sep='. ',
                                action='write')


        # Prepare Utah OER corpus
        self.oer_corpus = '%s/UtahOER/oer_text.txt' % corpus_dir
        if not os.path.exists(self.oer_corpus):
            textr = TextReader(min_chars_per_line=1, min_words_per_section=10)
            locdic = textr.read(dir='%s/UtahOER' % corpus_dir,
                                outfile=self.oer_corpus,
                                first_line_regexp='^.*Table of [cC]ontents*',
                                action='write')



        # Prepare Saylor+OpenStax corpus
        self.saylor_corpus = '%s/Saylor/saylor_text.txt' % corpus_dir
        if not os.path.exists(self.saylor_corpus):
            textr = TextReader(min_chars_per_line=1, min_words_per_section=20)
            locdic = textr.read(dir='%s/Saylor' % corpus_dir,
                                outfile=self.saylor_corpus,
                                first_line_regexp='^.*(CHAPTER|Chapter) 1.*',
                                action='write')


        # Prepare AI2 data corpus
        # SimpleLineReader - read a corpus that is a simple text file, each line is treated as a separate section
        # details can be found in the read_ai2_data method in Cardal_CorpusPreparation.py
        # code that generate ai2_summary.txt:
        # read_ai2_data(dirname, '/home/xihuan/Downloads/allenAI/Cardal/Kaggle_AllenAIscience/corpus/AI2_data/ai2_summary.txt')
        # each line is a question, followed by ans, possibly followed by justification
        self.ai2_corpus = '%s/AI2_data/ai2_corpus.txt' % corpus_dir
        if not os.path.exists(self.ai2_corpus):
            textr = SimpleLineReader(min_chars_per_line=1, min_words_per_section=2)
            locdic = textr.read(filenames=['%s/AI2_data/ai2_summary.txt' % corpus_dir],
                                outfile=self.ai2_corpus,
                                action='write')


        # Prepare StudyStack corpus
        # go to the studystack.com, click the study cards, click export, which will output a file, each line is question and ans
        self.sstack_corpus = '%s/StudyStack/studystack_corpus.txt' % corpus_dir
        if not os.path.exists(self.sstack_corpus):
            textr = SimpleLineReader(min_chars_per_line=1, min_words_per_section=2)
            locdic = textr.read(filenames=['%s/StudyStack/sstack_data.text' % corpus_dir],
                                outfile=self.sstack_corpus,
                                action='write')


        # Prepare StudyStack corpus #2 (small)
        self.sstack_corpus2 = '%s/StudyStack/studystack_corpus2.txt' % corpus_dir
        if not os.path.exists(self.sstack_corpus2):
            textr = SimpleLineReader(min_chars_per_line=1, min_words_per_section=2)
            locdic = textr.read(filenames=['%s/StudyStack/sstack_data2.text' % corpus_dir],
                                outfile=self.sstack_corpus2,
                                action='write')

        # Prepare StudyStack corpus #3 (small+)
        self.sstack_corpus3 = '%s/StudyStack/studystack_corpus3.txt' % corpus_dir
        if not os.path.exists(self.sstack_corpus3):
            textr = SimpleLineReader(min_chars_per_line=1, min_words_per_section=2)
            locdic = textr.read(filenames=['%s/StudyStack/sstack_data3.text' % corpus_dir],
                                outfile=self.sstack_corpus3,
                                action='write')

        # Prepare StudyStack corpus #4 (small-medium)
        self.sstack_corpus4 = '%s/StudyStack/studystack_corpus4.txt' % corpus_dir
        if not os.path.exists(self.sstack_corpus4):
            textr = SimpleLineReader(min_chars_per_line=1, min_words_per_section=2)
            locdic = textr.read(filenames=['%s/StudyStack/sstack_data4.text' % corpus_dir],
                                outfile=self.sstack_corpus4,
                                action='write')

        # Prepare quizlet corpus
        self.quizlet_corpus = '%s/quizlet/quizlet_corpus.txt' % corpus_dir
        if not os.path.exists(self.quizlet_corpus):
            textr = SimpleLineReader(min_chars_per_line=1, min_words_per_section=2)
            locdic = textr.read(filenames=['%s/quizlet/quizlet_data.text' % corpus_dir],
                                outfile=self.quizlet_corpus,
                                action='write')


        # Prepare SimpleWiki corpus #2
        # this wiki_name = 'simplewiki' is actually wiki_type, which is an argument when intializing WikiReader object
        """
        this is the format, <PAGE> tag is the title of a single wiki page
        <SECTION> tag under <PAGE> is the different sections belongs to this wiki page

        <PAGE>alan turing __7
        <SECTION> __1
        alan turing alan mathison turing order british empire obe frs london 23 june 1912 wilmslow cheshire june 1954 english people english mathematician computer scientist born maida vale london
        <SECTION>career __2
        turing one people worked first computers first person think using computer things hard person created turing machine 1936 machine imagination imaginary included idea computer program turing interested artificial intelligence proposed turing test say machine could called intelligent computer could said think human talking could not tell machine during world war ii turing worked break germany german ciphers secret messages using cryptanalysis helped break codes enigma machine enigma machine after solved german codes 1945 1947 turing worked design ace computer ace automatic computing engine national physical laboratory presented paper 19 february 1946 paper first detailed design stored-program computer although possible build ace delays starting project late 1947 returned cambridge sabbatical year cambridge pilot ace built without ran first program 10 may 1950
        <SECTION>private life __3
        turing homosexual man 1952 admitted sex man england time homosexual acts illegal turing conviction convicted choose between going jail taking hormones lower sex drive decided take hormones after punishment became erectile dysfunction impotent also grew breasts may 2012 private member's bill put before house lords grant turing statutory pardon july 2013 government supported royal pardon granted 24 december 2013
        <SECTION>death __4
        1954 after suffering two years turing died cyanide poisoning cyanide came either apple poisoned cyanide water cyanide reason confusion police never tested apple cyanide treatment forced now believed wrong against medical ethics international laws human rights august 2009 petition asking british government apologise turing punishing homosexual started petition received thousands signatures prime minister united kingdom prime minister gordon brown acknowledged petition called turing's treatment appalling
        <SECTION>other websites __6
        jack copeland 2012 alan turing codebreaker saved millions lives bbc news technology
        """
        self.simplewiki_corpus2 = '%s/simplewiki/simplewiki_1.0000_0.0500_0_5_True_True_True_corpus.txt' % corpus_dir
        if not os.path.exists(self.simplewiki_corpus2):
            wkb = WikiCorpusBuilder(wiki_name='simplewiki', wiki_dir='%s/simplewiki'%corpus_dir, wiki_file='simplewiki-20151102-pages-articles.xml', debug_flag=False)
            # create 2 files all_categories.pkl and parent_categories.pkl, if exist, will just load, they are stored in wkb.all_categories and wkb.parent_categories
            # we scan the wiki file find all categories that has <title>Categories:xxx</title> and their parent Catetories
            # details can be found in read_categories method in Cardal_WikiReader.py
            wkb.read_categories(reread=False)
            # create a file pages_in_categories.pkl, which stores a vector that contains all page names for all categories
            # for all singlewiki corpus, target_categories = None, important_categories=['Earth', 'Cellular respiration', 'DNA', 'Units of length', 'History of science',
            #                                                           'Evolutionary biology', 'Nonmetals', 'Health', 'Charles Darwin']
            # important_categories are science-related categories, if not found in target_catefories, which is generated from above method, will give an alert
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
            # this will create the corpus file as well as exams_words.pkl(all words that appear in train_b and valid_b),positive_words.pkl(all words in exam that are also uncommon in wiki),
            # and all_answers.pkl(this is a set, each element is a tuple of words within that answer)
            self.simplewiki_corpus2 = wkb.create_corpus(train_b, valid_b, min_pos_words_in_page_name=0, min_pos_words_in_section=5,
                                                        only_first_section_per_page=False, use_all_pages_match_pos_word=True, use_all_pages_match_answer=True,
                                                        always_use_first_section=True, max_read_lines=9990000000, reread=False)



        # Prepare SimpleWiki corpus #3
        # I did not go through the details of making this corpus, it should be very similar to the last one, with minor changes on the parameters
        self.simplewiki_corpus3 = '%s/simplewiki/simplewiki_1.0000_0.1000_0_3_True_True_False_corpus.txt' % corpus_dir
        if not os.path.exists(self.simplewiki_corpus3):
            wkb = WikiCorpusBuilder(wiki_name='simplewiki', wiki_dir='%s/simplewiki'%corpus_dir, wiki_file='simplewiki-20151102-pages-articles.xml', debug_flag=False)
            wkb.read_categories(reread=False)
            wkb.read_pages_in_categories(target_categories=None, max_cat_depth=9999,
                                         important_categories=['Earth', 'Cellular respiration', 'DNA', 'Units of length', 'History of science',
                                                               'Evolutionary biology', 'Nonmetals', 'Health', 'Charles Darwin'], reread=False)
            wkb.find_common_words(wiki_common_words_min_frac=1.0, wiki_uncommon_words_max_frac=0.1, use_wiki_stop_words=False, reread=False)
            self.simplewiki_corpus3 = wkb.create_corpus(train_b, valid_b, min_pos_words_in_page_name=0, min_pos_words_in_section=3,
                                                        only_first_section_per_page=False, use_all_pages_match_pos_word=True, use_all_pages_match_answer=True,
                                                        always_use_first_section=False, max_read_lines=9990000000, reread=False)

        # Prepare SimpleWiki corpus - page names
        self.simplewiki_corpus_pn = '%s/simplewiki/simplewiki_1.0000_0.0100_0_3_True_True_False_pn46669_corpus.txt' % corpus_dir
        if not os.path.exists(self.simplewiki_corpus_pn):
            wkb = WikiCorpusBuilder(wiki_name='simplewiki', wiki_dir='%s/simplewiki'%corpus_dir, wiki_file='simplewiki-20151102-pages-articles.xml', debug_flag=False)
            wkb.read_categories(reread=False)
            wkb.read_pages_in_categories(target_categories=None, max_cat_depth=9999,
                                         important_categories=['Earth', 'Cellular respiration', 'DNA', 'Units of length', 'History of science',
                                                               'Evolutionary biology', 'Nonmetals', 'Health', 'Charles Darwin'], reread=False)
            wkb.find_common_words(wiki_common_words_min_frac=1.0, wiki_uncommon_words_max_frac=0.01, use_wiki_stop_words=False, reread=False)
            self.simplewiki_corpus_pn = wkb.create_corpus(train_b=train_b, valid_b=valid_b, min_pos_words_in_page_name=0, min_pos_words_in_section=3,
                                                          only_first_section_per_page=False, use_all_pages_match_pos_word=True, use_all_pages_match_answer=True,
                                                          pages_to_use=self.word_sets,
                                                          always_use_first_section=False, max_read_lines=9990000000, reread=False)




        # Prepare wikibooks corpus
        self.wikibooks_corpus = '%s/wikibooks/wikibooks_1.0000_0.0200_0_10_True_True_False_corpus.txt' % corpus_dir
        if not os.path.exists(self.wikibooks_corpus):
            wkb = WikiCorpusBuilder(wiki_name='wikibooks', wiki_dir='%s/wikibooks'%corpus_dir, wiki_file='enwikibooks-20151102-pages-articles.xml', debug_flag=False)
            wkb.read_categories(reread=False)
            wkb.read_pages_in_categories(target_categories=None, max_cat_depth=9999, important_categories=[], reread=False)
            wkb.find_common_words(wiki_common_words_min_frac=1.0, wiki_uncommon_words_max_frac=0.02, use_wiki_stop_words=False, reread=False)
            self.wikibooks_corpus = wkb.create_corpus(train_b, valid_b, min_pos_words_in_page_name=0, min_pos_words_in_section=10,
                                                      only_first_section_per_page=False,
                                                      use_all_pages_match_pos_word=True, use_all_pages_match_answer=True,
                                                      always_use_first_section=False, max_read_lines=99900000000, reread=False)



        wiki_target_categories = set([#'Nature','Medicine',
                                      'Biology','Chemistry','Physics','Astronomy','Earth',
                                      'Genetics', 'Geology', 'Health', 'Science', 'Anatomy', 'Physiology', 'Solar System',
                                      'Water', 'Meteorology',
                                      'Water in the United States', 'Agriculture in the United States', 'Environment of the United States',
                                      #'Water', 'Physical chemistry', 'Physical phenomena', 'Human homeostasis', 'Body fluids'
                                      ])
        wiki_important_categories = []


        # Prepare wiki corpus #3 - only 1st section per page
        self.wiki_corpus3 = '%s/wiki/wiki_1.0000_0.0200_0_5_True_True_False_corpus.txt' % corpus_dir
        if not os.path.exists(self.wiki_corpus3):
            wkb = WikiCorpusBuilder(wiki_name='wiki', wiki_dir='%s/wiki'%corpus_dir, wiki_file='enwiki-20160113-pages-articles.xml', debug_flag=False)
            wkb.read_categories(reread=False)
            wkb.read_pages_in_categories(target_categories=wiki_target_categories, max_cat_depth=3, important_categories=wiki_important_categories, reread=False)
            wkb.find_common_words(wiki_common_words_min_frac=1.0, wiki_uncommon_words_max_frac=0.02, use_wiki_stop_words=False, max_read_lines=50000000, reread=False)
            self.wiki_corpus3 = wkb.create_corpus(train_b, valid_b, min_pos_words_in_page_name=0, min_pos_words_in_section=5,
                                                  only_first_section_per_page=True,
                                                  use_all_pages_match_pos_word=True, use_all_pages_match_answer=True,
                                                  always_use_first_section=False, max_read_lines=99900000000, reread=False)



        # Prepare wiki corpus - page names
        self.wiki_corpus_pn = '%s/wiki/wiki_0.5000_0.1000_0_5_True_True_False_pn46669_corpus.txt' % corpus_dir
        if not os.path.exists(self.wiki_corpus_pn):
            wkb = WikiCorpusBuilder(wiki_name='wiki', wiki_dir='%s/wiki'%corpus_dir, wiki_file='enwiki-20160113-pages-articles.xml', debug_flag=False)
            wkb.read_categories(reread=False)
            wkb.read_pages_in_categories(target_categories=None, max_cat_depth=9999, important_categories=wiki_important_categories, reread=False)
            wkb.find_common_words(wiki_common_words_min_frac=0.5, wiki_uncommon_words_max_frac=0.1, use_wiki_stop_words=False, max_read_lines=50000000, reread=False)
            self.wiki_corpus_pn = wkb.create_corpus(train_b, valid_b, min_pos_words_in_page_name=0, min_pos_words_in_section=5,
                                                    only_first_section_per_page=False,
                                                    use_all_pages_match_pos_word=True, use_all_pages_match_answer=True,
                                                    pages_to_use=self.word_sets,
                                                    always_use_first_section=False, max_read_lines=99900000000, reread=False)



        wkb = None
        locdic = None
        gc.collect()

        # Prepare Lucene indexes
        # this part has to use jython
        self.lucene_dir1, self.lucene_parser1, self.lucene_corpus1 = None, None, None
        self.lucene_dir2, self.lucene_parser2, self.lucene_corpus2 = None, None, None
        self.lucene_dir3, self.lucene_parser3, self.lucene_corpus3 = None, None, None
        self.lucene_dir4, self.lucene_parser4, self.lucene_corpus4 = None, None, None
        self.lucene_dir5, self.lucene_parser5, self.lucene_corpus5 = None, None, None
        self.lucene_dir6, self.lucene_parser6, self.lucene_corpus6 = None, None, None
        self.lucene_dir7, self.lucene_parser7, self.lucene_corpus7 = None, None, None
        # This condition is here since I don't have PyLucene on my Windows system
        if (len(sys.argv) >= 3) and (sys.argv[1] == 'prep') and (int(sys.argv[2]) >= 21):
            self.lucene_dir1 = '%s/lucene_idx1' % corpus_dir
            self.lucene_parser1 = SimpleWordParser(word_func=EnglishStemmer().stem, split_words_regexp='[\-\+\*\/\,\;\:\(\)]', min_word_length=1)
            self.lucene_corpus1 = LuceneCorpus(index_dir=self.lucene_dir1, filenames=[self.sstack_corpus, self.quizlet_corpus], parser=self.lucene_parser1)
            # note that for each section(not each page), we add the whole section to Lucene index, we store the text and makes it searchable
            # the section text are parsed using lucene_parser1, which in this case use EnglishStemmer to stem words
            # we have made our corpus consistent so that we can easily search the section we want to index and easily split the text.
            if not os.path.exists(self.lucene_dir1):
                 self.lucene_corpus1.prp_index()

            self.lucene_dir2 = '%s/lucene_idx2' % corpus_dir
            self.lucene_parser2 = SimpleWordParser(word_func=LancasterStemmer().stem, split_words_regexp='[\-\+\*\/\,\;\:\(\)]', min_word_length=1)
            self.lucene_corpus2 = LuceneCorpus(index_dir=self.lucene_dir2, filenames=[self.sstack_corpus3, self.quizlet_corpus, self.ck12text_corpus,
                                                                                      self.wiki_corpus_pn, self.simplewiki_corpus_pn], parser=self.lucene_parser2)
            if not os.path.exists(self.lucene_dir2):
                 self.lucene_corpus2.prp_index()

            self.lucene_dir3 = '%s/lucene_idx3' % corpus_dir
            self.lucene_parser3 = SimpleWordParser(word_func=PorterStemmer().stem, split_words_regexp='[\-\+\*\/\,\;\:\(\)]', min_word_length=1)
            self.lucene_corpus3 = LuceneCorpus(index_dir=self.lucene_dir3, filenames=[self.sstack_corpus4, self.quizlet_corpus, self.oer_corpus, self.saylor_corpus,
                                                                                      self.ck12html_para_corpus, self.ai2_corpus],
                                               parser=self.lucene_parser3, similarity=BM25Similarity())
            # note that for the first time we have similarities, so we do not use the default similarity, which is normalized TF-IDF
            # instead we use BM25 Similarity
            if not os.path.exists(self.lucene_dir3):
                 self.lucene_corpus3.prp_index()

            self.lucene_dir4 = '%s/lucene_idx4' % corpus_dir
            self.lucene_parser4 = SimpleWordParser(word_func=EnglishStemmer().stem, split_words_regexp='[\-\+\*\/\,\;\:\(\)]', min_word_length=1)
            self.lucene_corpus4 = LuceneCorpus(index_dir=self.lucene_dir4, filenames=[self.sstack_corpus, self.quizlet_corpus, self.ck12html_corpus],
                                               parser=self.lucene_parser4, similarity=BM25Similarity())
            if not os.path.exists(self.lucene_dir4):
                 self.lucene_corpus4.prp_index()

            self.lucene_dir5 = '%s/lucene_idx5' % corpus_dir
            self.lucene_parser5 = SimpleWordParser(word_func=LancasterStemmer().stem, split_words_regexp='[\-\+\*\/\,\;\:\(\)]', min_word_length=1)
            self.lucene_corpus5 = LuceneCorpus(index_dir=self.lucene_dir5, filenames=[self.wiki_corpus3, self.simplewiki_corpus3],
                                               parser=self.lucene_parser5, similarity=None)
            if not os.path.exists(self.lucene_dir5):
                 self.lucene_corpus5.prp_index()

            self.lucene_dir6 = '%s/lucene_idx6' % corpus_dir
            self.lucene_parser6 = SimpleWordParser(word_func=EnglishStemmer().stem, split_words_regexp='[\-\+\*\/\,\;\:\(\)]', min_word_length=1)
            self.lucene_corpus6 = LuceneCorpus(index_dir=self.lucene_dir6, filenames=[self.ck12text_corpus, self.saylor_corpus, self.oer_corpus, self.ai2_corpus],
                                               parser=self.lucene_parser6, similarity=BM25Similarity())
            if not os.path.exists(self.lucene_dir6):
                 self.lucene_corpus6.prp_index()

            self.lucene_dir7 = '%s/lucene_idx7' % corpus_dir
            self.lucene_parser7 = SimpleWordParser(word_func=PorterStemmer().stem, split_words_regexp='[\-\+\*\/\,\;\:\(\)]', min_word_length=1)
            self.lucene_corpus7 = LuceneCorpus(index_dir=self.lucene_dir7, filenames=[self.sstack_corpus2, self.wiki_corpus_pn, self.simplewiki_corpus_pn,
                                                                                      self.ck12html_para_corpus, self.oer_corpus],
                                               parser=self.lucene_parser7, similarity=None)
            if not os.path.exists(self.lucene_dir7):
                 self.lucene_corpus7.prp_index()

        print '-> Finished preparing corpuses'

    ALL_FEATURE_TYPES = {# "BASIC" features extracted from the questions and answers, w/o external corpus:
                         'BASIC': 0,
                         # Features computed using my search functions:
                         'ck-hp_saylor.triplets.1': 1, 'ck-hp_saylor_oer.triplets.1': 2,  'qz_ck-ts.1': 3,
                         'st2_qz_oer_ck-hp.1': 4, 'st2_qz_oer_ck-t.triplets.1': 5,
                         'st2_qz_wk-pn_oer_ck-h.pairs.1': 6, 'st_qz.1': 7, 'st_qz.pairs2.1': 8, 'st_qz_ai.1': 9,
                         'st_qz_saylor_ck-t.a1_vs_a2.1': 10, 'sw-pn_qz.1': 11,
                         'sw-pn_ss_ck-t_ai.1': 12, 'sw2_ck-ts.1': 13, 'tr_st_qz.1': 14, 'tr_st_qz.2': 15, 'wk3_sw3.1': 16,
                         'wk-pn_sw-pn_wb.a1_vs_a2.1': 17, 'st_qz.triplets13.1': 18, 'st_qz.Z': 19, 'wk-pn_sw-pn.1': 20,
                         # Features computed using PyLucene:
                         'lucene.1': 21, 'lucene.2': 22, 'lucene.3': 23, 'lucene.4': 24, 'lucene.5': 25, 'lucene.6': 26, 'lucene.7': 27,
                         }

    def prepare_features(self, dataf_q, dataf_b, train_df, aux_b, cache_dir, ftypes=None):
        '''
        Compute one or more features by running the relevant search function.
        aux_b - an additional binary data source with possibly same questions as in dataf_b, to save computations
        '''
        if ftypes is not None:
            assert (len(ftypes) > 0) and (set(ftypes).issubset(FeatureExtractor.ALL_FEATURE_TYPES.values())), \
                    'Feature types should be non-empty subset of:\n%s' % FeatureExtractor.ALL_FEATURE_TYPES
        self.cache_dir = '%s/%s' % (self.base_dir, cache_dir)
        create_dirs([self.cache_dir])

        if self.print_level > 0:
            print '-> Preparing features, cache dir = %s' % cache_dir

        locdic = None
        stemmer1 = PorterStemmer()
        stem1 = stemmer1.stem
        stemmer2 = LancasterStemmer()
        stem2 = stemmer2.stem
        stemmer3 = EnglishStemmer()
        stem3 = stemmer3.stem

        tag_weights1 = {'NN':1.5,'NNP':1.5,'NNPS':1.5,'NNS':1.5, 'VB':1.3,'VBD':1.3,'VBG':1.3,'VBN':1.3,'VBP':1.3,'VBZ':1.3,
                        'JJ':1.0,'JJR':1.0,'JJS':1.0, 'RB':1.0,'RBR':1.0,'RBS':1.0,'RP':1.0}
        tag_weight_func1 = lambda tag: tag_weights1.get(tag, 0.8)

        tag_weights2 = {'NN':2.0,'NNP':2.0,'NNPS':2.0,'NNS':2.0, 'VB':1.5,'VBD':1.5,'VBG':1.5,'VBN':1.5,'VBP':1.5,'VBZ':1.5,
                        'JJ':1.0,'JJR':1.0,'JJS':1.0, 'RB':0.8,'RBR':0.8,'RBS':0.8,'RP':0.8}
        tag_weight_func2 = lambda tag: tag_weights2.get(tag, 0.5)

        swr = '[\-\+\*\/\,\;\:\(\)]' # split_words_regexp

        if 'correctAnswer' in dataf_q.columns:
            targets_q = dict(zip(dataf_q.index,dataf_q['correctAnswer']))
            targets_b = np.array(dataf_b['correct'])
        else:
            targets_q, targets_b = None, None

        train_b = dataf_b

        # ==================================================================================================================================
        # Compute funcs with various combinations of corpora, parsers, and score params
        # ==================================================================================================================================
        ds_funcs = {
                    # 1: 0.4448
                    'ck-hp_saylor.triplets.1': {'corpora': [self.ck12html_para_corpus, self.saylor_corpus],
                                                'parser': SimpleWordParser(word_func=stem3, tuples=[1,2,3], split_words_regexp=swr),
                                                'num_words_qst': [None]+range(2,60), 'num_words_ans': [None]+range(2,40), 'score': 'weights', 'prob_type': 'tf-idf', 'tf_log_flag': True,
                                                'score_params': {'coeffs': lambda n: 1.0/(1.0+np.arange(n)), 'calc_over_vs_under': True,
                                                                 'minword1_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))**0.4, 'minword2_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))**0.7},
                                                'recalc': False, 'skip': False},
                    # 2: 0.4500
                    'ck-hp_saylor_oer.triplets.1': {'corpora': [self.ck12html_para_corpus, self.saylor_corpus, self.oer_corpus],
                                                    'parser': SimpleWordParser(word_func=stem2, tuples=[1,2,3], min_word_length=1),
                                                    'num_words_qst': [None]+range(2,60), 'num_words_ans': [None]+range(2,40), 'score': 'weights', 'prob_type': 'tf-idf', 'tf_log_flag': True,
                                                    'score_params': {'coeffs': lambda n: 1.0/(1.0+np.arange(n))**0.7, 'calc_over_vs_under': True,
                                                                     'minword1_coeffs': lambda mw,nw: np.sqrt((1.0+mw)/(1.0+nw)), 'minword2_coeffs': lambda mw,nw: np.sqrt((1.0+mw)/(1.0+nw))},
                                                    'recalc': False, 'skip': False},
                    # 3: 0.4164
                    'qz_ck-ts.1': {'corpora': [self.quizlet_corpus, self.ck12text_sent_corpus],
                                   'parser': NltkTokenParser(word_func=lambda word,tag: stem3(word), word_func_requires_tag=False,
                                                             tuples=[1], tag_weight_func=tag_weight_func1),
                                   'num_words_qst': [None]+range(2,30), 'num_words_ans': [None]+range(2,20), 'score': 'hg', 'prob_type': 'tf-idf', 'tf_log_flag': False,
                                   'score_params': {'minword1_coeffs': lambda mw,nw: np.sqrt((1.0+mw)/(1.0+nw)), 'minword2_coeffs': lambda mw,nw: np.sqrt((1.0+mw)/(1.0+nw))},
                                   'recalc': False, 'skip': False},
                    # 4: 0.4720
                    'st2_qz_oer_ck-hp.1': {'corpora': [self.sstack_corpus2, self.quizlet_corpus, self.oer_corpus, self.ck12html_para_corpus],
                                           'parser': SimpleWordParser(word_func=None, min_word_length=1),
                                           'num_words_qst': [None]+range(2,30), 'num_words_ans': [None]+range(2,20), 'score': 'weights', 'prob_type': 'tf-idf', 'tf_log_flag': False,
                                           'score_params': {'norm': lambda w: w**2/(np.sum(w**2) + 0.0),
                                                            'minword1_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))**1.4, 'minword2_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))},
                                           'recalc': False, 'skip': False},
                    # 5: 0.5256
                    'st2_qz_oer_ck-t.triplets.1': {'corpora': [self.sstack_corpus2, self.quizlet_corpus, self.oer_corpus, self.ck12text_corpus],
                                                   'parser': SimpleWordParser(word_func=stem3, tuples=[1,2,3], split_words_regexp=swr, min_word_length=1),
                                                   'num_words_qst': [None]+range(2,60), 'num_words_ans': [None]+range(2,40), 'score': 'weights', 'prob_type': 'tf-idf', 'tf_log_flag': False,
                                                   'score_params': {'coeffs': lambda n: 2.0/(2.0+np.arange(n))**0.7, 'calc_over_vs_under': True,
                                                                    'minword1_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))**0.4, 'minword2_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))**0.6},
                                                   'recalc': False, 'skip': False},
                    # 6: 0.5388
                    'st2_qz_wk-pn_oer_ck-h.pairs.1': {#'corpora': [self.sstack_corpus2, self.quizlet_corpus, self.wiki_corpus_pn, self.oer_corpus, self.ck12html_corpus],
                                                      'corpora': [self.sstack_corpus4, self.quizlet_corpus, self.wiki_corpus_pn, self.oer_corpus, self.ck12html_corpus],
                                                      'parser': SimpleWordParser(word_func=stem3, tuples=[1,2], split_words_regexp=swr, min_word_length=1),
                                                      'num_words_qst': [None]+range(2,50), 'num_words_ans': [None]+range(2,30), 'score': 'weights', 'prob_type': 'tf-idf', 'tf_log_flag': False,
                                                      'score_params': {'coeffs': lambda n: 1.0/(1.0+np.arange(n))**0.8, 'calc_over_vs_under': True,
                                                                       'minword1_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))**0.3, 'minword2_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))**0.7},
                                                      'recalc': False, 'skip': False},
                    # 7: 0.5580
                    'st_qz.1': {'corpora': [self.sstack_corpus, self.quizlet_corpus],
                                'parser': SimpleWordParser(word_func=stem3, tuples=[1]),
                                'num_words_qst': [None]+range(2,30), 'num_words_ans': [None]+range(2,20), 'score': 'weights', 'prob_type': 'tf-idf', 'tf_log_flag': True,
                                'score_params': {'coeffs': lambda n: 1.0/(1.0+np.arange(n)), 'calc_over_vs_under': True,
                                                 'minword1_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw)), 'minword2_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))},
                                'recalc': False, 'skip': False},
                    # 8: 0.3340
                    'st_qz.pairs2.1': {'corpora': [self.sstack_corpus, self.quizlet_corpus],
                                       'parser': SimpleWordParser(word_func=stem3, tuples=[2], min_word_length=1),
                                       'num_words_qst': [None]+range(2,30), 'num_words_ans': [None]+range(2,20), 'score': 'weights', 'prob_type': 'tf-idf', 'tf_log_flag': True,
                                       'score_params': {'coeffs': lambda n: (2.0/(2.0+np.arange(n)))**1.4, 'calc_over_vs_under': True,
                                                        'minword1_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))**0.8, 'minword2_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))**0.9},
                                       'recalc': False, 'skip': False},
                    # 9: 0.5184
                    'st_qz_ai.1': {'corpora': [self.sstack_corpus, self.quizlet_corpus, self.ai2_corpus],
                                   'parser': NltkTokenParser(word_func=lambda word,tag: stem2(word), word_func_requires_tag=False,
                                                             tuples=[1], tag_weight_func=tag_weight_func1),
                                   'num_words_qst': [None]+range(2,30), 'num_words_ans': [None]+range(2,20), 'score': 'weights', 'prob_type': 'tf-idf', 'tf_log_flag': False,
                                   'score_params': {'coeffs': lambda n: 1.0/(1.0+np.arange(n))**0.3, 'calc_over_vs_under': True,
                                                    'norm': lambda w: w**2/(np.sum(w**2) + 0.0),
                                                    'minword1_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))**2, 'minword2_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))**1.5},
                                   'recalc': False, 'skip': False},
                    # 10: 0.3080
                    'st_qz_saylor_ck-t.a1_vs_a2.1': {'corpora': [self.sstack_corpus, self.quizlet_corpus, self.saylor_corpus, self.ck12text_corpus],
                                                     'parser': SimpleWordParser(word_func=stem2, tuples=[1]),
                                                     'num_words_qst': [None]+range(2,30), 'num_words_ans': [None]+range(2,20), 'score': 'weights', 'prob_type': 'tf-idf', 'tf_log_flag': True,
                                                     'score_params': {'coeffs': lambda n: 1.0/(1.0+np.arange(n)), 'calc_over_vs_under': True,
                                                                      'minword1_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw)), 'minword2_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))},
                                                     'apairs': {'sim_scores_comb_weights': ([10, 3, 1], [1, 3, 10]), 'search_type': 'a1_vs_a2'},
                                                     'recalc': False, 'skip': False},
                    # 11: 0.4628
                    'sw-pn_qz.1': {'corpora': [self.simplewiki_corpus_pn, self.quizlet_corpus],
                                   'parser': SimpleWordParser(word_func=stem3, tuples=[1], split_words_regexp=swr),
                                   'num_words_qst': [None]+range(2,30), 'num_words_ans': [None]+range(2,20), 'score': 'weights', 'prob_type': 'tf-idf', 'tf_log_flag': True,
                                   'score_params': {'coeffs': lambda n: 1.0/(1.0+np.arange(n))**2.0,
                                                    'minword1_coeffs': lambda mw,nw: np.sqrt((1.0+mw)/(1.0+nw)), 'minword2_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))**0.8},
                                   'recalc': False, 'skip': False},
                    # 12: 0.4620
                    'sw-pn_ss_ck-t_ai.1': {'corpora': [self.simplewiki_corpus_pn, self.sstack_corpus, self.ck12text_corpus, self.ai2_corpus],
                                           'parser': SimpleWordParser(word_func=stem2, tuples=[1], split_words_regexp=swr),
                                           'num_words_qst': [None]+range(2,30), 'num_words_ans': [None]+range(2,20), 'score': 'weights', 'prob_type': 'tf-idf', 'tf_log_flag': True,
                                           'score_params': {'coeffs': lambda n: 1.0/(1.0+np.arange(n))**0.3,
                                                            'minword1_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))**1.8, 'minword2_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))**1.5},
                                           'recalc': False, 'skip': False},
                    # 13: 0.4192
                    'sw2_ck-ts.1': {'corpora': [self.simplewiki_corpus2, self.ck12text_sent_corpus],
                                    'parser': SimpleWordParser(word_func=stem2, tuples=[1], split_words_regexp=swr),
                                    'num_words_qst': [None]+range(2,30), 'num_words_ans': [None]+range(2,20), 'score': 'weights', 'prob_type': 'tf-idf', 'tf_log_flag': False,
                                    'score_params': {'minword1_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw)), 'minword2_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))**1.8,
                                                     'norm': lambda w: w**2/(np.sum(w**2) + 0.0)},
                                    'recalc': False, 'skip': False},
                    # 14: 0.5468
                    'tr_st_qz.1': {'corpora': [self.sstack_corpus, self.quizlet_corpus],
                                   'parser': SimpleWordParser(word_func=stem1, tuples=[1], split_words_regexp=swr, min_word_length=1),
                                   'num_words_qst': [None]+range(2,30), 'num_words_ans': [None]+range(2,15), 'score': 'weights', 'prob_type': 'tf-idf', 'tf_log_flag': True,
                                   'score_params': {'coeffs': lambda n: np.sqrt(1.0/(1.0+np.arange(n))), 'calc_over_vs_under': True,
                                                    'minword1_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))**1.5, 'minword2_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))**2},
                                   'recalc': False, 'skip': False, 'train': True},
                    # 15: 0.5088
                    'tr_st_qz.2': {'corpora': [self.sstack_corpus, self.quizlet_corpus],
                                   'parser': NltkTokenParser(word_func=None, word_func_requires_tag=False, tuples=[1], tag_weight_func=tag_weight_func1),
                                   'num_words_qst': [None]+range(2,30), 'num_words_ans': [None]+range(2,15), 'score': 'weights', 'prob_type': 'tf-idf', 'tf_log_flag': False,
                                   'score_params': {'coeffs': lambda n: np.ones(n), 'calc_over_vs_under': True,
                                                    'norm': lambda w: w**2/(np.sum(w**2) + 0.0),
                                                    'minword1_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))**0.75, 'minword2_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))},
                                   'recalc': False, 'skip': False, 'train': True},
                    # 16: 0.4028
                    'wk3_sw3.1': {'corpora': [self.wiki_corpus3, self.simplewiki_corpus3],
                                  'parser': SimpleWordParser(word_func=stem1, tuples=[1], split_words_regexp=swr),
                                  'num_words_qst': [None]+range(3,40), 'num_words_ans': [None]+range(2,20), 'score': 'weights', 'prob_type': 'tf-idf', 'tf_log_flag': True,
                                  'score_params': {'coeffs': lambda n: (1.0/(1.0+np.arange(n)))**1.2},
                                  'recalc': False, 'skip': False},
                    # 17: over 0.2872, under 0.2908
                    'wk-pn_sw-pn_wb.a1_vs_a2.1': {'corpora': [self.wiki_corpus_pn, self.simplewiki_corpus_pn, self.wikibooks_corpus],
                                                  'parser': SimpleWordParser(word_func=stem3, tuples=[1], split_words_regexp=swr),
                                                  'num_words_qst': [None]+range(1,30), 'num_words_ans': [None]+range(1,20), 'score': 'weights', 'prob_type': 'tf-idf', 'tf_log_flag': False,
                                                  'score_params': {'coeffs': lambda n: 1.0/(1.0+np.arange(n))**0.7, 'calc_over_vs_under': True,
                                                                   'minword1_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))**1.5, 'minword2_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))**1.5},
                                                  'apairs': {'sim_scores_comb_weights': ([12, 2, 1], [1, 2, 12]), 'search_type': 'a1_vs_a2'},
                                                  'recalc': False, 'skip': False},
                    # 18: 0.5500
                    'st_qz.triplets13.1': {'corpora': [self.sstack_corpus, self.quizlet_corpus],
                                           'parser': SimpleWordParser(word_func=stem1, tuples=[1,3], min_word_length=1, split_words_regexp=swr),
                                           'num_words_qst': [None]+range(2,60), 'num_words_ans': [None]+range(2,30), 'score': 'weights', 'prob_type': 'tf-idf', 'tf_log_flag': True,
                                           'score_params': {'coeffs': lambda n: 3.0/(3.0+np.arange(n)), 'calc_over_vs_under': True,
                                                            'minword1_coeffs': lambda mw,nw: ((3.0+mw)/(3.0+nw))**1.2, 'minword2_coeffs': lambda mw,nw: ((3.0+mw)/(3.0+nw))**1.5},
                                           'recalc': False, 'skip': False},
                    # 19: 0.45126
                    'st_qz.Z': {'corpora': [self.sstack_corpus, self.quizlet_corpus],
                                'parser': SimpleWordParser(word_func=stem3, split_words_regexp=swr),
                                'norm_scores': True,
                                'recalc': False, 'skip': False, 'zscore': True},
                    # 20: 0.3884
                    'wk-pn_sw-pn.1': {'corpora': [self.wiki_corpus_pn, self.simplewiki_corpus_pn],
                                      'parser': SimpleWordParser(word_func=None, tuples=[1], split_words_regexp=swr),
                                      'num_words_qst': [None]+range(2,40), 'num_words_ans': [None]+range(2,30), 'score': 'weights', 'prob_type': 'tf-idf', 'tf_log_flag': False,
                                      'score_params': {'norm': lambda w: w**1.5/(np.sum(w**1.5) + 0.0)},
                                      'recalc': False, 'skip': False},
                    # 21: 0.5424
                    'lucene.1': {'lucene_corpus': self.lucene_corpus1,
                                 'parser': self.lucene_parser1,
                                 'max_docs': 1000, 'weight_func': lambda n: 1.0/(1.0+np.arange(n))**1.5, 'score_func': None, 'norm_scores': True,
                                 'recalc': False, 'skip': False, 'lucene': True},
                    # 22: 0.5304
                    'lucene.2': {'lucene_corpus': self.lucene_corpus2,
                                 'parser': self.lucene_parser2,
                                 'max_docs': 500, 'weight_func': lambda n: 1.0/(1.0+np.arange(n))**1.2, 'score_func': lambda s: s**1.5, 'norm_scores': True,
                                 'recalc': False, 'skip': False, 'lucene': True},
                    # 23: 0.5388
                    'lucene.3': {'lucene_corpus': self.lucene_corpus3,
                                 'parser': self.lucene_parser3,
                                 'max_docs': 3000, 'weight_func': lambda n: 1.0/(3.0+np.arange(n))**1.4, 'score_func': lambda s: (s/100.0)**5, 'norm_scores': True,
                                 'recalc': False, 'skip': False, 'lucene': True},
                    # 24: 0.5500
                    'lucene.4': {'lucene_corpus': self.lucene_corpus4,
                                 'parser': self.lucene_parser4,
                                 'max_docs': 2500, 'weight_func': lambda n: 1.0/(1.0+np.arange(n))**2.2, 'score_func': lambda s: (s/100.0)**4, 'norm_scores': True,
                                 'recalc': False, 'skip': False, 'lucene': True},
                    # 25: 0.4292
                    'lucene.5': {'lucene_corpus': self.lucene_corpus5,
                                 'parser': self.lucene_parser5,
                                 'max_docs': 750, 'weight_func': lambda n: 1.0/(2.0+np.arange(n))**1.6, 'score_func': lambda s: (s/10.0)**2.5, 'norm_scores': True,
                                 'recalc': False, 'skip': False, 'lucene': True},
                    # 26: 0.4672
                    'lucene.6': {'lucene_corpus': self.lucene_corpus6,
                                 'parser': self.lucene_parser6,
                                 'max_docs': 800, 'weight_func': lambda n: 1.0/(5.0+np.arange(n))**2, 'score_func': lambda s: (s/10.0)**3, 'norm_scores': True,
                                 'recalc': False, 'skip': False, 'lucene': True},
                    # 27: 0.4684
                    'lucene.7': {'lucene_corpus': self.lucene_corpus7,
                                 'parser': self.lucene_parser7,
                                 'max_docs': 250, 'weight_func': lambda n: 1.0/(10.0+np.arange(n)), 'score_func': lambda s: (s+2.0)**3.4, 'norm_scores': True,
                                 'recalc': False, 'skip': False, 'lucene': True},
                   }

        check_same_question = not set(train_b['ID']).isdisjoint(train_df['ID'])

        for fn,params in sorted(ds_funcs.iteritems()):
            if params['skip']: continue
            if (ftypes is not None) and (FeatureExtractor.ALL_FEATURE_TYPES[fn] not in ftypes): continue
            if params.has_key('zscore') or params.has_key('lucene'):
                func_name = fn
            else:
                func_name = ['%s_over'%fn, '%s_under'%fn]
            if self.print_level > 1:
                print 'Computing features: %s' % str(func_name)
            if params.has_key('corpora'):
                locdic = lambda: CorpusReader.build_locdic_from_outfile(filename=params['corpora'], parser=params['parser'],
                                                                        min_word_docs_frac=0, max_word_docs_frac=1, min_word_count_frac=0, max_word_count_frac=1)
            else:
                locdic = None
            norm_scores = params['norm_scores'] if params.has_key('norm_scores') else self.norm_scores_default
            self.recalc = params['recalc']
            #print 'recalc = %s' % self.recalc
            if params.has_key('train'):
                assert params['train']
                self.add_answer_func(train_b, aux_b,
                                     func=AnswersTrainDoubleSearchFunc(train_df[train_df['correct']==1], check_same_question=check_same_question,
                                                                       base_locdic=locdic, parser=params['parser'],
                                                                       num_words_qst=params['num_words_qst'], num_words_ans=params['num_words_ans'],
                                                                       score=params['score'], score_params=params['score_params'], norm_scores=norm_scores,
                                                                       prob_type=params['prob_type'], tf_log_flag=params['tf_log_flag']),
                                     name=func_name)
            elif params.has_key('train0'):
                assert params['train0'] and (locdic is None)
                self.add_answer_func(train_b, aux_b,
                                     func=AnswersTrainDoubleSearchFunc(train_df[train_df['correct']==0], check_same_question=check_same_question,
                                                                       parser=params['parser'],
                                                                       num_words_qst=params['num_words_qst'], num_words_ans=params['num_words_ans'],
                                                                       score=params['score'], score_params=params['score_params'], norm_scores=norm_scores,
                                                                       prob_type=params['prob_type'], tf_log_flag=params['tf_log_flag']),
                                     name=func_name)
            elif params.has_key('apairs'):
                self.add_answer_func(train_b, aux_b,
                                     func=AnswersPairsDoubleSearchFunc(locdic=locdic, parser=params['parser'],
                                                                       num_words_qst=params['num_words_qst'], num_words_ans=params['num_words_ans'],
                                                                       score=params['score'], score_params=params['score_params'], norm_scores=norm_scores,
                                                                       prob_type=params['prob_type'], tf_log_flag=params['tf_log_flag'],
                                                                       sim_scores_comb_weights=params['apairs']['sim_scores_comb_weights'], search_type=params['apairs']['search_type']),
                                     name=func_name)

            elif params.has_key('zscore'):
                self.add_answer_func(train_b, aux_b,
                                     func=AnswersWordZscoreFunc(locdic=locdic, parser=params['parser'], norm_scores=norm_scores),
                                     name=func_name)
            elif params.has_key('lucene'):
                self.add_answer_func(train_b, aux_b,
                                     func=AnswersLuceneSearchFunc(lucene_corpus=params['lucene_corpus'], parser=params['parser'],
                                                                  max_docs=params['max_docs'], weight_func=params['weight_func'], score_func=params['score_func'],
                                                                  norm_scores=norm_scores),
                                     name=func_name)
            else:
                self.add_answer_func(train_b, aux_b,
                                     func=AnswersDoubleSearchFunc(locdic=locdic, parser=params['parser'],
                                                                  num_words_qst=params['num_words_qst'], num_words_ans=params['num_words_ans'],
                                                                  score=params['score'], score_params=params['score_params'], norm_scores=norm_scores,
                                                                  prob_type=params['prob_type'], tf_log_flag=params['tf_log_flag']),
                                     name=func_name)
            if ((self.print_level > 1) or self.recalc) and (targets_q is not None):
                if params.has_key('zscore') or params.has_key('lucene'):
                    print ' AUC of %s: %.4f' % (func_name, calc_auc(targets_b, train_b[func_name], two_sides=True))
                    print ' Accuracy of %s: %.4f' % (func_name, calc_accuracy(targets_q, get_predictions_from_binary_dataf(train_b, func_name, direction='max')))
                else:
                    print ' AUC of %s: %.4f' % (func_name[0], calc_auc(targets_b, train_b[func_name[0]], two_sides=True))
                    print ' Accuracy of %s: %.4f' % (func_name[0], calc_accuracy(targets_q, get_predictions_from_binary_dataf(train_b, func_name[0], direction='max')))
                    print ' AUC of %s: %.4f' % (func_name[1], calc_auc(targets_b, train_b[func_name[1]], two_sides=True))
                    print ' Accuracy of %s: %.4f' % (func_name[1], calc_accuracy(targets_q, get_predictions_from_binary_dataf(train_b, func_name[1], direction='max')))
            gc.collect()
            self.recalc = False

        # ==================================================================================================================================
        # Compute the "basic" features, ie, those obtained from the questions+answers, without any external corpus
        # ==================================================================================================================================
        if (ftypes is None) or (FeatureExtractor.ALL_FEATURE_TYPES['BASIC'] in ftypes):
            simple_parser   = SimpleWordParser(word_func=None , ignore_special_words=False, min_word_length=1)
            pairs_parser    = SimpleWordParser(word_func=None , ignore_special_words=False, min_word_length=1, tuples=[1,2])
            stemmed_parser  = SimpleWordParser(word_func=stem1, ignore_special_words=True , min_word_length=1)
            stemmed_parserB = SimpleWordParser(word_func=stem3, ignore_special_words=True , min_word_length=2, split_words_regexp=swr)
            stemmed_parserC = SimpleWordParser(word_func=stem2, ignore_special_words=True , min_word_length=2, split_words_regexp=swr)
            stemmed_parser2 = SimpleWordParser(word_func=stem2, ignore_special_words=False, min_word_length=1)
            stemmed_parser3 = SimpleWordParser(word_func=stem3, ignore_special_words=False, min_word_length=1)
            stemmed_pairs_parser3 = SimpleWordParser(word_func=stem3, ignore_special_words=False, min_word_length=1, tuples=[1,2], split_words_regexp=swr)

            func_name = 'ans_in_qst'
            self.add_answer_func(train_b, aux_b, func=AnswersInQuestionFunc(), name=func_name)
            func_name = 'ans_in_qst_stem'
            self.add_answer_func(train_b, aux_b, func=AnswersInQuestionFunc(parser=stemmed_parser), name=func_name)

            func_name = 'ans_in_ans'
            self.add_answer_func(train_b, aux_b, func=AnswersInAnswersFunc(), name=func_name)
            func_name = 'ans_in_ans_stem'
            self.add_answer_func(train_b, aux_b, func=AnswersInAnswersFunc(parser=stemmed_parser), name=func_name)

            func_name = 'ans_count'
            self.add_answer_func(train_b, aux_b, func=AnswerCountFunc(train_df, check_same_question=check_same_question, count_type='count'), name=func_name) # 0.2532 (0.5313)
            func_name = 'ans_words_stem_count'
            self.add_answer_func(train_b, aux_b, func=AnswerCountFunc(train_df, check_same_question=check_same_question, count_type='count', parser=stemmed_parser, single_words=True), name=func_name)
            func_name = 'ans_words_stem_count_nonorm'
            self.add_answer_func(train_b, aux_b, func=AnswerCountFunc(train_df, check_same_question=check_same_question, count_type='count', parser=stemmed_parserC, single_words=True, norm_scores=False), name=func_name)
            func_name = 'ans_qst_words_stem_count'
            self.add_answer_func(train_b, aux_b, func=AnswerCountFunc(train_df, check_same_question=check_same_question, count_type='count', parser=stemmed_parserB, single_words=True, use_questions=True), name=func_name)

            func_name = 'ans_correct'
            self.add_answer_func(train_b, aux_b, func=AnswerCountFunc(train_df, check_same_question=check_same_question, count_type='correct'), name=func_name)
            func_name = 'ans_words_stem_correct'
            self.add_answer_func(train_b, aux_b, func=AnswerCountFunc(train_df, check_same_question=check_same_question, count_type='correct', parser=stemmed_parser, single_words=True), name=func_name)
            func_name = 'ans_words_stem_correct_nonorm'
            self.add_answer_func(train_b, aux_b, func=AnswerCountFunc(train_df, check_same_question=check_same_question, count_type='correct', parser=stemmed_parserB, single_words=True, norm_scores=False), name=func_name)
            func_name = 'ans_qst_words_stem_correct'
            self.add_answer_func(train_b, aux_b, func=AnswerCountFunc(train_df, check_same_question=check_same_question, count_type='correct', parser=stemmed_parser3, single_words=True, use_questions=True), name=func_name)
            func_name = 'ans_words_stem_pairs_correct'
            self.add_answer_func(train_b, aux_b, func=AnswerCountFunc(train_df, check_same_question=check_same_question, count_type='correct', parser=stemmed_pairs_parser3, single_words=True), name=func_name)

            func_name = 'ans_pval'
            self.add_answer_func(train_b, aux_b, func=AnswerCountFunc(train_df, check_same_question=check_same_question, count_type='pval'), name=func_name)
            func_name = 'ans_stem_pval'
            self.add_answer_func(train_b, aux_b, func=AnswerCountFunc(train_df, check_same_question=check_same_question, count_type='pval', parser=stemmed_parser, single_words=False), name=func_name)
            func_name = 'ans_words_stem_pval'
            self.add_answer_func(train_b, aux_b, func=AnswerCountFunc(train_df, check_same_question=check_same_question, count_type='pval', parser=stemmed_parser3, single_words=True), name=func_name)
            func_name = 'ans_words_pairs_zscore'
            self.add_answer_func(train_b, aux_b, func=AnswerCountFunc(train_df, check_same_question=check_same_question, count_type='zscore', parser=pairs_parser, single_words=True), name=func_name)
            func_name = 'ans_words_stem_pval'
            self.add_answer_func(train_b, aux_b, func=AnswerCountFunc(train_df, check_same_question=check_same_question, count_type='pval', parser=stemmed_parser3, single_words=True), name=func_name)
            func_name = 'ans_words_stem_zscore'
            self.add_answer_func(train_b, aux_b, func=AnswerCountFunc(train_df, check_same_question=check_same_question, count_type='zscore', parser=stemmed_parser2, single_words=True), name=func_name)

            func_name = 'ans_corr_vs_qst_count'
            self.add_answer_func(train_b, aux_b, func=AnswerCountFunc(train_df, check_same_question=check_same_question, count_type='ans_vs_qst', parser=stemmed_parser, single_words=True, norm_scores=False), name=func_name)

            func_name = 'ans_length'
            self.add_answer_func(train_b, aux_b, func=AnswersLengthFunc(log_flag=True ), name=func_name)
            func_name = 'ans_length_ratio'
            self.add_answer_func(train_b, aux_b, func=AnswersLengthFunc(log_flag=False), name=func_name)
            func_name = 'ans_num_words'
            self.add_answer_func(train_b, aux_b, func=AnswersNumWordsFunc(), name=func_name)

            func_name = 'is_BC'
            self.add_answer_func(train_b, aux_b, func=AnswersIsBCFunc(), name=func_name)
            func_name = 'BCDA'
            self.add_answer_func(train_b, aux_b, func=AnswersBCDAFunc(), name=func_name)

            func_name = 'is_numerical'
            self.add_answer_func(train_b, aux_b, func=AnswersIsNumericalFunc(), name=func_name)

        print '-> Finished preparing features (types: %s)' % ('all' if ftypes is None else ', '.join(['%s'%ft for ft in ftypes]))

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

    def add_answer_func(self, train_b, aux_b, func, name, question_ids=None):
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

        if aux_b is None:
            aux_ids = set()
        else:
            aux_ids = set(aux_b['ID'])
            print 'Given %d aux IDs' % len(aux_ids)
        groups = train_b.groupby('ID').groups
        for i,(idx,inds) in enumerate(groups.iteritems()):
            assert len(set(train_b.irow(inds)['question']))==1
            if (question_ids is not None) and (idx not in question_ids): continue
            question = train_b.irow(inds[0])['question']
            is_dummy = train_b.irow(inds[0])['is_dummy']
            answers = np.array(train_b.irow(inds)['answer'])
            if 'correct' in train_b.columns:
                print '\n-----> #%d : correct = %s' % (i, ', '.join(['%d'%c for c in np.array(train_b.irow(inds)['correct'])]))
                sys.stdout.flush()
    #         print 'applying func to: %s' % str(np.array(train_b.irow(inds)['answer']))

            # Check if there's an identical question in aux_b
            if idx in aux_ids:
                same_qst = aux_b[aux_b['ID']==idx]
                assert question == np.unique(same_qst['question'])[0]
                assert set(answers) == set(same_qst['answer'])
                assert np.all(answers == np.array(same_qst['answer']))
                print 'Found same question in aux (ID %s):\n%s' % (idx, str(same_qst))
                vals = []
                for ai,ans in enumerate(answers): # for ans in answers:
                    if np.isscalar(name):
                        #vals.append(float(same_qst[same_qst['answer']==ans][name]))
                        vals.append(float(same_qst.irow(ai)[name]))
                    else:
                        #vals.append(np.array(same_qst[same_qst['answer']==ans][name]).flatten())
						vals.append(np.array(same_qst.irow(ai)[name]).flatten())
                print ' -> vals: %s' % vals
            else:
                # Check if it's a dummy question
                if is_dummy > 1:
                    # No need to waste time on dummy questions...
                    if np.isscalar(name):
                        vals = [-1] * len(inds)
                    else:
                        vals = [np.ones(len(name)) * (-1)] * len(inds)
                else:
                    # Compute func
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






