import pandas as pd
import gc
import os

from IO import save_to_pkl, load_from_pkl
from NLPUtils import SpecialWords
from Parser import SimpleWordParser
from LocationDict import build_training_location_dictionary
from WikiReader import WikiReader



class WikiCorpusBuilder(object):
    ALL_CATEGORIES_FILE      = 'all_categories.pkl'
    PARENT_CATEGORIES_FILE   = 'parent_categories.pkl'
    USE_CATEGORIES_FILE      = 'use_categories.pkl'
    PAGES_IN_CATEGORIES_FILE = 'pages_in_categories.pkl'
    COMMON_WORDS_FILE        = 'common_words.pkl'
    UNCOMMON_WORDS_FILE      = 'uncommon_words.pkl'
    STOP_WORDS_FILE          = 'stop_words.pkl'
    EXAMS_WORDS_FILE         = 'exams_words.pkl'
    POSITIVE_WORDS_FILE      = 'positive_words.pkl'
    ANSWERS_FILE             = 'all_answers.pkl'
    CORPUS_FILE              = 'corpus.txt'

    def __init__(self, wiki_name, wiki_dir, wiki_file, debug_flag=False):
        self.wiki_name = wiki_name
        self.wiki_dir  = wiki_dir
        self.wiki_file = wiki_file
        # wkb will have an WikiReader object as private variable
        self.wikir = WikiReader(wiki_name, debug_flag=debug_flag)

    # Create 2 files all_categories.pkl and parent_categories.pkl, if exist, will just load,
    # They are stored in wkb.all_categories and wkb.parent_categories
    # we scan the wiki file find all categories that has <title>Categories:xxx</title> and their parent Catetories
    # details can be found in read_categories method in WikiReader.py
    def read_categories(self, reread=False):
        # this function create 'all_categories.pkl' and 'parent_categories.pkl'
        # there are 29586 categories and 27923 parent categories
        print '=> Reading categories for %s' % self.wiki_name
        categories_file = '%s/%s_%s' % (self.wiki_dir, self.wiki_name, WikiCorpusBuilder.ALL_CATEGORIES_FILE)
        parents_file    = '%s/%s_%s' % (self.wiki_dir, self.wiki_name, WikiCorpusBuilder.PARENT_CATEGORIES_FILE)
        gc.collect()
        if reread or (not os.path.exists(categories_file)) or (not os.path.exists(parents_file)):
            # if it is the 1st time run this code, will end up in this block and create this 2 category files
            # it will call the WikiReader to get all the category names from wiki file by scanning through it and match category regex
            self.wikir.read_sub_categories(wikifile='%s/%s' % (self.wiki_dir, self.wiki_file), max_read_lines=99900000000)
            save_to_pkl(categories_file, self.wikir.all_categories)
            save_to_pkl(parents_file, self.wikir.parent_categories)
        else:
            self.wikir.all_categories = load_from_pkl(categories_file)
            self.wikir.parent_categories = load_from_pkl(parents_file)
        print 'There are a total of %d categories' % len(self.wikir.all_categories)

    # Create 2 files 'use_categories.pkl' and 'pages_in_categories.pkl'
    # for all singlewiki corpus, target_categories = None, important_categories=['Earth', 'Cellular respiration', 'DNA', 'Units of length', 'History of science',
    #                                                           'Evolutionary biology', 'Nonmetals', 'Health', 'Charles Darwin']
    # important_categories are science-related categories, if not found in target_catefories, which is generated from above method, will give an alert
    # it will all read_pages_in_categories in Cardal_WikiReader.py
    def read_pages_in_categories(self, target_categories, max_cat_depth, important_categories, reread=False):
        print '=> Reading pages in target categories for %s' % self.wiki_name
        self.target_categories = target_categories
        self.max_cat_depth = max_cat_depth
        use_categories_file      = '%s/%s_%s' % (self.wiki_dir, self.wiki_name, WikiCorpusBuilder.USE_CATEGORIES_FILE)
        pages_in_categories_file = '%s/%s_%s' % (self.wiki_dir, self.wiki_name, WikiCorpusBuilder.PAGES_IN_CATEGORIES_FILE)
        if reread or (not os.path.exists(use_categories_file)) or (not os.path.exists(pages_in_categories_file)):
            if self.target_categories is None:
                # generated from the above method
                self.use_categories = self.wikir.all_categories
            else:
                # this block check that target categories(which we think are very relevant) are all included in our search category
                self.use_categories = set([cat for cat in self.wikir.all_categories
                                           if self.wikir.search_categories(cat, self.target_categories, max_depth=self.max_cat_depth) >= 0])
            save_to_pkl(use_categories_file, self.use_categories)

            self.pages_in_categories = self.wikir.read_pages_in_categories(wikifile='%s/%s' % (self.wiki_dir, self.wiki_file),
                                                                           use_categories=self.use_categories, max_read_lines=99900000000)
            save_to_pkl(pages_in_categories_file, self.pages_in_categories)
        else:
            self.use_categories = load_from_pkl(use_categories_file)
            self.pages_in_categories = load_from_pkl(pages_in_categories_file)

        print 'Using %d categories related to %s target categories with depth <= %d' % \
                (len(self.use_categories), 'x' if self.target_categories is None else len(self.target_categories), self.max_cat_depth)
        print 'Missing important categories: %s' % str([cat for cat in important_categories if cat not in self.use_categories])
        print 'There are %d pages in the %d categories' % (len(self.pages_in_categories), len(self.use_categories))


    # this will read all the text from wiki file
    # parse useful pure text and build a dict of words
    # depends on the common words and uncommon words fraction, we pick up common words and uncommon words
    # we also add common words to stop words
    # we finally save common_words.pkl, uncommon_words.pkl and stop_words.pkl to corpus dir
    def find_common_words(self, wiki_common_words_min_frac=0.2, wiki_uncommon_words_max_frac=0.01, use_wiki_stop_words=True,
                          max_read_lines=100000000, reread=False):
        print '=> Finding common/uncommon words'
        self.wiki_common_words_min_frac = wiki_common_words_min_frac
        self.wiki_uncommon_words_max_frac = wiki_uncommon_words_max_frac
        self.use_wiki_stop_words = use_wiki_stop_words
        # the 3 files not exist at begining, need to create once
        common_words_file   = '%s/%s_%.4f_%s'   % (self.wiki_dir, self.wiki_name, self.wiki_common_words_min_frac, WikiCorpusBuilder.COMMON_WORDS_FILE)
        uncommon_words_file = '%s/%s_%.4f_%s'   % (self.wiki_dir, self.wiki_name, self.wiki_uncommon_words_max_frac, WikiCorpusBuilder.UNCOMMON_WORDS_FILE)
        stop_words_file     = '%s/%s_%.4f_%s%s' % (self.wiki_dir, self.wiki_name, self.wiki_common_words_min_frac, 'wsw_' if self.use_wiki_stop_words else '', WikiCorpusBuilder.STOP_WORDS_FILE)
        # Read first X lines from Wiki corpus, and get the set of Wiki stop-words (words that appear in many documents),
        # as well as the "uncommon" words (words that appear in a small fraction of the documents)
        if reread or (not os.path.exists(common_words_file)) or (not os.path.exists(uncommon_words_file)) or (not os.path.exists(stop_words_file)):
            # this line creates a locdic variable (Cardal_LocationDict object)
            # by calling the read function, it actually read the wiki file with action = 'locdic', this will create a location dict
            # for each page, and for each section in each page, we read all its section text, and perform the add_words function in Cardal_LocationDict
            # the input for this function are have page_name, section_name, section_number, section_text
            # the add_words function: 1st arg is page_name + page_id, 2nd arg is section_name + section_id, 3rd arg is the section_text
            # this will also compute the count of all parsed words
            wiki_locdic = self.wikir.read(wikifile='%s/%s' % (self.wiki_dir, self.wiki_file),
                                          outfile='%s/%s_locdic1.txt' % (self.wiki_dir, self.wiki_name), # ignored...
                                          #only_first_section_per_page=True, max_read_lines=max_read_lines,
                                          only_first_section_per_page=False, max_sections_per_page=1, max_read_lines=max_read_lines,
                                          stop_words=SpecialWords.ignore_words, pos_words=set(),
                                          min_pos_words_in_page_name=0, min_pos_words_in_section=0, action='locdic')
            # there are 2 fraction thresholds for common words and uncommon words
            # depends on the threshold, these 2 values could be different
            self.wiki_common_words   = set([word for dc,word in wiki_locdic.sort_words_by_num_docs() if dc>(self.wiki_common_words_min_frac  *wiki_locdic.get_num_docs())])
            self.wiki_uncommon_words = set([word for dc,word in wiki_locdic.sort_words_by_num_docs() if dc<(self.wiki_uncommon_words_max_frac*wiki_locdic.get_num_docs())])
            # we add common words to stopwords
            self.stop_words = set(SpecialWords.ignore_words).union(self.wiki_common_words)
            if self.use_wiki_stop_words:
                self.stop_words.update(WikiReader.WIKI_STOP_WORDS)
            wiki_locdic = None
            gc.collect()
            save_to_pkl(common_words_file  , self.wiki_common_words)
            save_to_pkl(uncommon_words_file, self.wiki_uncommon_words)
            save_to_pkl(stop_words_file    , self.stop_words)
        else:
            self.wiki_common_words   = load_from_pkl(common_words_file)
            self.wiki_uncommon_words = load_from_pkl(uncommon_words_file)
            self.stop_words          = load_from_pkl(stop_words_file)

        print 'There are %d common words (>%.4f docs)'   % (len(self.wiki_common_words), self.wiki_common_words_min_frac)
        print 'There are %d uncommon words (<%.4f docs)' % (len(self.wiki_uncommon_words), self.wiki_uncommon_words_max_frac)
        print 'Using %d stop words (%s wiki stop words)' % (len(self.stop_words), 'with' if self.use_wiki_stop_words else 'without')

    def create_corpus(self, train_b, valid_b, min_pos_words_in_page_name, min_pos_words_in_section,
                      only_first_section_per_page=False, max_sections_per_page=99999999,
                      use_all_pages_match_pos_word=True, use_all_pages_match_answer=True, pages_to_use=None, always_use_first_section=False,
                      max_read_lines=99900000000, reread=False):
        print '=> Creating corpus'
        self.min_pos_words_in_page_name   = min_pos_words_in_page_name
        self.min_pos_words_in_section     = min_pos_words_in_section
        self.only_first_section_per_page  = only_first_section_per_page
        self.max_sections_per_page        = max_sections_per_page
        self.use_all_pages_match_pos_word = use_all_pages_match_pos_word
        self.use_all_pages_match_answer   = use_all_pages_match_answer
        self.always_use_first_section     = always_use_first_section
        exams_words_file = '%s/%s_%s' % (self.wiki_dir, self.wiki_name, WikiCorpusBuilder.EXAMS_WORDS_FILE)
        pos_words_file   = '%s/%s_%.4f_%s%s' % (self.wiki_dir, self.wiki_name, self.wiki_common_words_min_frac, 'wsw_' if self.use_wiki_stop_words else '', WikiCorpusBuilder.POSITIVE_WORDS_FILE)
        answers_file     = '%s/%s_%s' % (self.wiki_dir, self.wiki_name, WikiCorpusBuilder.ANSWERS_FILE)
        corpus_file      = '%s/%s_%.4f_%s%.4f_%d_%d_%s_%s_%s' % (self.wiki_dir, self.wiki_name, self.wiki_common_words_min_frac, 'wsw_' if self.use_wiki_stop_words else '',
                                                                 self.wiki_uncommon_words_max_frac, self.min_pos_words_in_page_name, self.min_pos_words_in_section,
                                                                 self.use_all_pages_match_pos_word, self.use_all_pages_match_answer,
                                                                 self.always_use_first_section)
        if pages_to_use is not None:
            corpus_file = '%s_pn%d' % (corpus_file, len(pages_to_use))
        corpus_file = '%s_%s' % (corpus_file, WikiCorpusBuilder.CORPUS_FILE)
        print 'Corpus file: %s' % corpus_file
        gc.collect()

        # Get the corpus of the train+validation sets
        if reread or (not os.path.exists(pos_words_file)) or (not os.path.exists(answers_file)):
            # Get all the words that appear in the exams
            if valid_b is None:
                all_exams = train_b[['ID','question','answer']]
            else:
                all_exams = pd.concat([train_b[['ID','question','answer']], valid_b[['ID','question','answer']]])
            parser = SimpleWordParser()
            exams_locdic = build_training_location_dictionary(all_exams, parser=parser, use_answers=True,
                                                              min_word_docs_frac=0, max_word_docs_frac=1.0, min_word_count_frac=0, max_word_count_frac=1.0,
                                                              ascii_conversion=True)
            self.exams_words = exams_locdic.word_ids.keys()
            # Set the "positive_words" as all the words from the train(+validation) files that are uncommon in Wiki
            self.pos_words = set(self.exams_words).intersection(self.wiki_uncommon_words)
            # Get all the answers (each answer = a set of words)
            self.all_answers = set()
            for answer in all_exams['answer']:
                self.all_answers.add(tuple(sorted(parser.parse(answer))))
            save_to_pkl(exams_words_file, self.exams_words)
            save_to_pkl(pos_words_file, self.pos_words)
            save_to_pkl(answers_file, self.all_answers)
        else:
            self.exams_words = load_from_pkl(exams_words_file)
            self.pos_words   = load_from_pkl(pos_words_file)
            self.all_answers = load_from_pkl(answers_file)

        print 'There are %d positive words (%d wiki uncommon words, %d words from exams)' % (len(self.pos_words), len(self.wiki_uncommon_words), len(self.exams_words))
        print 'There are a total of %d unique answers' % len(self.all_answers)
        print 'Using %d stop words' % (len(self.stop_words))
        if pages_to_use is None:
            use_pages = self.pages_in_categories
        else:
            use_pages = pages_to_use
        print 'Considering %d pages' % len(use_pages)

        if reread or (not os.path.exists(corpus_file)):
            print 'Writing %s corpus to %s' % (self.wiki_name, corpus_file)
            ld = self.wikir.read(wikifile='%s/%s' % (self.wiki_dir, self.wiki_file),
                                 outfile=corpus_file,
                                 only_first_section_per_page=self.only_first_section_per_page, max_sections_per_page=self.max_sections_per_page,
                                 use_pages=use_pages,
                                 max_read_lines=max_read_lines,
                                 stop_words=self.stop_words, pos_words=self.pos_words,
                                 page_name_word_sets=self.all_answers, corpus_words=None, ##set(exams_locdic.word_ids.keys()),
                                 min_pos_words_in_page_name=self.min_pos_words_in_page_name, min_pos_words_in_section=self.min_pos_words_in_section,
                                 use_all_pages_match_pos_word=self.use_all_pages_match_pos_word, use_all_pages_match_sets=self.use_all_pages_match_answer,
                                 always_use_first_section=self.always_use_first_section,
                                 action='write')
            print 'Done writing corpus'

        gc.collect()
        return corpus_file
