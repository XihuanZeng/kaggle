import numpy as np
import gc
from IO import save_to_pkl
from Parser import SimpleWordParser
from LocationDict import LocationDictionary

class CorpusReader(object):
    '''
    CorpusReader - base class for corpus readers
    '''
    PAGE_NAME_PREFIX    = '<PAGE>'
    SECTION_NAME_PREFIX = '<SECTION>'

    PART_NAMES_IGNORE = set(['introduction', 'summary'])

    def __init__(self, min_chars_per_line=50, min_words_per_section=50, debug_flag=False):
        self.min_chars_per_line = min_chars_per_line
        self.min_words_per_section = min_words_per_section
        self.debug_flag = debug_flag
        self._reset(outfile=None, stop_words=None, pos_words=None, page_name_word_sets=None, corpus_words=None,
                    min_pos_words_in_page_name=-1, min_pos_words_in_section=-1,
                    use_all_pages_match_pos_word=False, use_all_pages_match_sets=False, always_use_first_section=False,
                    action=None)
        self.sections_to_use = None

    def _reset(self, outfile, stop_words, pos_words, page_name_word_sets, corpus_words,
               min_pos_words_in_page_name, min_pos_words_in_section, use_all_pages_match_pos_word, use_all_pages_match_sets, always_use_first_section,
               action):
        if (stop_words is not None) and (pos_words is not None) and (len(stop_words.intersection(pos_words)) > 0):
            print 'Stop words contain pos words - removing from pos words: %s' % stop_words.intersection(pos_words)
            pos_words = pos_words.difference(stop_words)
        assert (stop_words is None) or len(stop_words.intersection(pos_words))==0
        self.outfile = outfile
        self.stop_words, self.pos_words, self.page_name_word_sets, self.corpus_words = stop_words, pos_words, page_name_word_sets, corpus_words
        self.min_pos_words_in_page_name, self.min_pos_words_in_section = min_pos_words_in_page_name, min_pos_words_in_section
        self.use_all_pages_match_pos_word, self.use_all_pages_match_sets = use_all_pages_match_pos_word, use_all_pages_match_sets
        self.always_use_first_section = always_use_first_section
        self.action = action
        self._outf, self._locdic = None, None
        self.num_pages, self.num_sections = 0, 0
        self.num_section_action = 0
        self.pages_in_corpus = set() # names of pages that are actually in the corpus

    def set_sections_to_use(self, sections_to_use):
        if sections_to_use is None:
            self.sections_to_use = sections_to_use
        else:
            self.sections_to_use = set(sections_to_use)

    def _start_action(self):
        self.pages_in_corpus = set()
        if self.action == 'write':
            self._outf = open(self.outfile,'w')
        elif self.action == 'locdic':
            self._locdic = LocationDictionary(save_locations=False, doc_name_weight=0)
        else:
            raise ValueError('Unsupported action (%s)' % self.action)

    def _end_action(self):
        if self._outf is not None:
            self._outf.close()
            self._outf = None
        # Write pages_in_corpus
        if self.action == 'write':
            save_to_pkl('%s.pages.pkl' % self.outfile, self.pages_in_corpus)
        gc.collect()

    @staticmethod
    def part_name_from_words(words, number):
        if (len(words) == 1) and (words[0] in CorpusReader.PART_NAMES_IGNORE):
            words = []
        return '%s __%d' % (' '.join(words), number)

    @staticmethod
    def words_from_part_name(part_name):
        words = part_name.split(' ')
        assert words[-1].startswith('__')
        return words[:-1]

    def _add_page(self, page_name, page_name_words):
        self.num_pages += 1
        if self.action == 'write':
            self._outf.write('%s%s\n' % (CorpusReader.PAGE_NAME_PREFIX, CorpusReader.part_name_from_words(page_name_words, self.num_pages)))

    def _check_page_name(self, page_name, page_name_words):
        '''
        Returns True if page should be used; False if it should be skipped
        '''
        if self.use_all_pages_match_sets and (tuple(sorted(page_name_words)) in self.page_name_word_sets):
            return True
        num_pos_words_in_page_name = len(set(page_name_words).intersection(self.pos_words))
        if self.use_all_pages_match_pos_word and (num_pos_words_in_page_name > 0):
            return True
        if num_pos_words_in_page_name >= self.min_pos_words_in_page_name:
            return True
        return False

    def _add_section(self, page_name, page_name_words, section_name, section_name_words, section_number, section_words):
        '''
        Returns 1 if the section was added, 0 otherwise
        Need to check if this is a valid section
        '''
        self.num_sections += 1
        if ((not self.always_use_first_section) or (section_number > 1)) and (len(section_words) < self.min_words_per_section):
            if self.debug_flag:
                print 'section "%s" (%d) too short (%d words)' % (section_name, section_number, len(section_words))
            return 0
        if not self._check_page_name(page_name, page_name_words):
            return 0
        if (self.sections_to_use is not None) and (section_name not in self.sections_to_use):
            if self.debug_flag:
                print 'section "%s" (%d) not in sections_to_use set' % (section_name, section_number)
            return 0
        if self.stop_words is not None:
            section_words = [w for w in section_words if not w in self.stop_words]
        num_pos_words_in_section = len(set(section_words).intersection(self.pos_words))
        if ((not self.always_use_first_section) or (section_number > 1)) and (num_pos_words_in_section < self.min_pos_words_in_section):
            if self.debug_flag:
                print 'section "%s" (%d) has too few pos words (%d)' % (section_name, section_number, num_pos_words_in_section)
            return 0
        if self.debug_flag:
            print 'page "%s" section "%s" (%d) has %d pos words (total %d words)' % (page_name, section_name, section_number, num_pos_words_in_section, len(section_words))
        if self.corpus_words is not None:
            section_words = [w for w in section_words if w in self.corpus_words]
        if self.action == 'write':
            self._outf.write('%s%s\n' % (CorpusReader.SECTION_NAME_PREFIX, CorpusReader.part_name_from_words(section_name_words, section_number)))
            self._outf.write('%s\n' % ' '.join(section_words))
            self.num_section_action += 1
        elif self.action == 'locdic':
            self._locdic.add_words('%s/%s' % (CorpusReader.part_name_from_words(page_name_words, self.num_pages),
                                              CorpusReader.part_name_from_words(section_name_words, section_number)),
                                   page_name_words + section_name_words, section_words)
            self.num_section_action += 1
        self.pages_in_corpus.add(page_name)
        return 1

    @staticmethod
    def build_locdic_from_outfile(filename, parser=SimpleWordParser(),
                                  min_word_docs_frac=0, max_word_docs_frac=0.2, min_word_count_frac=0, max_word_count_frac=0.01,
                                  doc_name_weight=0):
        locdic = LocationDictionary(save_locations=False, doc_name_weight=doc_name_weight)
        locdic.set_search_word_filter(min_word_docs_frac=min_word_docs_frac, max_word_docs_frac=max_word_docs_frac,
                                      min_word_count_frac=min_word_count_frac, max_word_count_frac=max_word_count_frac)
        num_pages, num_sections = 0, 0
        page_name, section_name = None, None
        num_lines = 0
        if type(filename)==str:
            assert file is not None
            filenames = [filename]
        else:
            assert not np.any([(fn is None) for fn in filename])
            filenames = filename # list of file names
        for ifname,fname in enumerate(filenames):
            print 'Building locdic from file #%d: %s' % (ifname, fname)
            with open(fname,'rt') as infile:
                for text in infile:
                    if len(text)==0:
                        print 'Reached EOF'
                        break # EOF
                    if text.startswith(CorpusReader.PAGE_NAME_PREFIX):
                        page_name = text[len(CorpusReader.PAGE_NAME_PREFIX):].strip()
                        section_name = None
                        num_pages += 1
                    elif text.startswith(CorpusReader.SECTION_NAME_PREFIX):
                        section_name = text[len(CorpusReader.SECTION_NAME_PREFIX):].strip()
                        num_sections += 1
                    else:
                        assert (page_name is not None) and (section_name is not None)
                        section_words = parser.parse(text, calc_weights=False)
                        locdic.add_words('F%d/%s/%s' % (ifname, page_name, section_name), CorpusReader.words_from_part_name(page_name) + CorpusReader.words_from_part_name(section_name),
                                         section_words)
                    num_lines += 1
                    if num_lines % 100000 == 0:
                        print ' read %d lines: %d pages, %d sections -> %d words' % (num_lines, num_pages, num_sections, len(locdic.word_ids))
        return locdic