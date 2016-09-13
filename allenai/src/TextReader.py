import time
import re
import os
from CorpusReader import CorpusReader
from Parser import SimpleWordParser



class TextReader(CorpusReader):
    '''
    TextReader - read a "text" corpus
    '''
    def read(self, dir, outfile, stop_words=set(), pos_words=set(),
             first_line_regexp='^CHAPTER',
             ignore_sections=set(), section_end_regexp='^\s*$',
             action='write'):

        self._reset(outfile=outfile, stop_words=stop_words, pos_words=pos_words, page_name_word_sets=set(), corpus_words=None,
                    min_pos_words_in_page_name=0, min_pos_words_in_section=0,
                    use_all_pages_match_pos_word=True, use_all_pages_match_sets=True, always_use_first_section=False,
                    action=action)

        parser = SimpleWordParser(tolower=True, ascii_conversion=True, ignore_special_words=False)

        first_line_re = re.compile(first_line_regexp)
        section_end_re = re.compile(section_end_regexp)

        self._start_action()
        page_name, section_name, section_in_page = None, None, 0
        page_name_words, section_words = [], []
        start_time = time.time()
        # this includes all the .text files which are converted from pdf books
        filenames = ['%s/%s'%(dir,fname) for fname in os.listdir(dir) if fname.endswith('.text')]
        assert len(filenames)>0
        for ifname,fname in enumerate(filenames):
            print 'Reading %s' % fname
            page_name = fname[:-5]
            page_name_words = []
            # 1 file is 1 page
            print 'page name = %s' % page_name
            self._add_page(page_name, page_name_words)
            section_in_page = 0
            section_name, section_name_words = '', []
            with open (fname, 'rb') as myfile:
                found_first_line = False
                text = ''
                # will search for the first_line_re in the file, e.g 'C HAPTER' for CK12-textbooks
                # given that we find first line, if we find a line that contains section_end_re, that is multiple spaces,
                # we write lines we have seen so far to new section
                # it turns out that if we set each paragraph a section, there are more than 5000 sections for 1 page(file)
                # to actually add a section, we also use _add_section method in CorpusReader.py to check if it is a valid section.
                # For instance, we ignore it if section has too few words or if it merely contains figures and formulas
                for line in myfile:
                    line = line.strip()
                    # note that the online pdf to text converter that I used will produce some of the title caption as
                    # 'V IRAL S EXUALLY T RANSMITTED I NFECTIONS', where the space between chars should be substituted
                    line = re.sub('(?<=[A-Z]{1})(\s)(?=[A-Z]{2,})', '', line)
                    if found_first_line:
                        if re.match(section_end_re, line) is not None:
                            # Add previous section
                            section_words = parser.parse(text)
                            self._add_section(page_name, page_name_words, section_name, section_name_words, section_in_page, section_words)
                            section_in_page += 1
                            section_name, section_name_words = '', []
                            text = ''
                        else:
                            text += ' ' + line
                    else:
                        if re.match(first_line_re, line) is not None:
                            found_first_line = True
            assert found_first_line, 'Could not find first line in file %s' % fname
            # Add last section
            section_words = parser.parse(text)
            self._add_section(page_name, page_name_words, section_name, section_name_words, section_in_page, section_words)

        end_time = time.time()
        print 'read_text total time = %.1f secs.' % (end_time-start_time)
        print 'Read %d pages, %d sections; applied action on %d sections' % (self.num_pages, self.num_sections, self.num_section_action)
        self._end_action()

        return self._locdic