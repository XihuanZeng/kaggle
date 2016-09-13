import time
import re
import gc

from CorpusReader import CorpusReader
from Parser import SimpleWordParser



class WikiReader(CorpusReader):
    '''
    WikiReader - read a Wiki corpus
    '''
    # List of words that appear in >10% of (a sample of) Wiki docs, after manual removal of some words
    # List does not include the NLTK stop words
    WIKI_STOP_WORDS = set(['also', 'one', 'known', 'new', 'two', 'may', 'part', 'used', 'many', 'made', 'since',
                           'including', 'later', 'well', 'became', 'called', 'three', 'named', 'second', 'several', 'early',
                           'often', 'however', 'best', 'use', 'although', 'within'])

    # Regexp for page names to ignore
    IGNORE_PAGES = re.compile('(\S[\:]\S)|([\(]disambiguation[\)])|(Wikipedia)')

    NEW_PAGE_SUBSTR     , NEW_PAGE_RE      = '<title>'    , re.compile('<title>(.*)</title>')
    CATEGORY_SUBSTR     , CATEGORY_RE      = '[[Category:', re.compile('[\[]+Category[\:]([^\]\|]*)([\|][^\]\|]*)*[\]]+') # eg, "[[Category:Social theories]]"
    CATEGORY_SUBSTR2    , CATEGORY_RE2     = '{{'         , re.compile('[\{][\{]([^\}]+)[\}][\}]\s*<[\/]text>') # eg, "{{music-stub}}</text>" (in SimpleWiki)
    PAGE_REDIRECT_SUBSTR, PAGE_REDIRECT_RE = '<redirect'  , re.compile('\s*<redirect') # eg, " <redirect title="Light pollution" />" (in SimpleWiki)
    NEW_SECTION_SUBSTR  , NEW_SECTION_RE   = '=='         , re.compile('\s*=[=]+([^=]*)=[=]+\s*')

    CATEGORY_PAGE_NAME = re.compile('Category[\:](.*)')

    # Text replacements
    RE_REMOVALS0 = [(sr,re.compile(rr)) for sr,rr in [('[', '[\[]+[^\]\:]+[\:][^\]]+[\]]+'),
                                                      ('{{', '[\{][\{][^\}]+[\}][\}]')]]
    BLANKS_CONVERSIONS = ['&quot;', '&amp;nbsp;', '&amp;', '&nbsp;', '|']
    OTHER_CONVERSIONS = [('&lt;', '<'), ('&gt;', '>'),
                         (chr(195)+chr(164), 'a'), (chr(195)+chr(167), 'c'), (chr(195)+chr(169), 'e'), (chr(195)+chr(184), 'o'), (chr(197)+chr(143), 'o'),
                         (chr(194)+chr(188), '1/4'),
                         (chr(194)+chr(183), '*'),
                         (chr(226)+chr(128)+chr(152), "'"  ), (chr(226)+chr(128)+chr(153), "'"  ),
                         (chr(226)+chr(128)+chr(156), '"'  ), (chr(226)+chr(128)+chr(157), '"'  ),
                         (chr(226)+chr(128)+chr(147), ' - '), (chr(226)+chr(128)+chr(148), ' - '), (chr(226)+chr(136)+chr(146), '-')]
    RE_REMOVALS = [(sr,re.compile(rr)) for sr,rr in [('<ref', '<ref[^\>]*>[^\<\>]+<[\/]ref>'),
                                                     ('<', '<[^\>]*>'),
                                                     ("''", "[']+"),
                                                     ('wikt:', 'wikt:\S+')]]
    STR_REMOVALS = ['[[',']]', '#*:','#*']

    def __init__(self, wiki_type, debug_flag=False):
        assert wiki_type in ['wiki', 'simplewiki', 'wiktionary', 'wikibooks', 'wikiversity']
        self.wiki_type = wiki_type
        if self.wiki_type == 'wiki':
            min_chars_per_line, min_words_per_section = 50, 50
        elif self.wiki_type == 'simplewiki':
            min_chars_per_line, min_words_per_section = 1, 1
        elif self.wiki_type == 'wiktionary':
            min_chars_per_line, min_words_per_section = 1, 3
        elif self.wiki_type == 'wikibooks':
            min_chars_per_line, min_words_per_section = 1, 10
        elif self.wiki_type == 'wikiversity':
            min_chars_per_line, min_words_per_section = 1, 3
        CorpusReader.__init__(self, min_chars_per_line=min_chars_per_line, min_words_per_section=min_words_per_section, debug_flag=debug_flag)
        if self.wiki_type == 'wiktionary':
            self.set_sections_to_use(['Noun'])

    def search_categories(self, category, target_categories, max_depth=5):
        checked_categories = set()
        categories_to_check = set([(category, 0)])
        while len(categories_to_check) > 0:
            cat, depth = categories_to_check.pop()
            if cat in target_categories:
                return depth
            checked_categories.add(cat)
            if self.parent_categories.has_key(cat) and depth+1 <= max_depth:
                categories_to_check.update([(c,depth+1) for c in self.parent_categories[cat].difference(checked_categories)])
        return -1

    def read_sub_categories(self, wikifile, max_read_lines=None):
        print '=> Reading sub categories'
        self.all_categories, self.parent_categories = set(), {}
        category = None
        num_lines = 0
        with open(wikifile,'rt') as infile:
            # text is a single line
            for text in infile:
                if len(text)==0:
                    print 'Reached End of File'
                    break
                num_lines += 1
                if (max_read_lines is not None) and (num_lines > max_read_lines):
                    break
                if num_lines % 1000000 == 0:
                    print 'Read %d lines, total of %d categories...' % (num_lines, len(self.all_categories))
                    gc.collect()
                # NEW_PAGE_SUBSTR is '<title>', this method will only consider lines with such <title> tag and skip those content lines
                # and for those lines with <title>(.*)</title>, we only stop for <title>Category:xxx</title> and add xxx to our category
                if WikiReader.NEW_PAGE_SUBSTR in text:
                    # new_page should be a length-1 list, since there is only 1 title between <title>(.*)</title>
                    new_page = re.findall(WikiReader.NEW_PAGE_RE, text)
                    if len(new_page)>0:
                        assert len(new_page)==1
                        page_name = new_page[0]
                        # some line is <title>April</title>, cmatch will be None, which is what we want
                        # but some line is <title>Category:Computer science</title> in which case cmatch will be not None
                        cmatch = re.match(WikiReader.CATEGORY_PAGE_NAME, page_name)
                        if cmatch is not None:
                            category = cmatch.groups(0)[0].strip()
                            self.all_categories.add(category)
                            assert (not self.parent_categories.has_key(category)) #or (len(self.parent_categories[category])==0)
                            self.parent_categories[category] = set()
                        else:
                            category = None
                        continue
                # when we find a line <title>Category:xxx</title>, we set category flag not None and
                # scan its following lines to search for parent category
                # e.g last block we find a category 'Computer science', we add it to the self.all_categories
                # in the following block, we find the parent category for 'Computer science' is 'Computing',
                # so we also add 'Computing' in self.all_categories
                # and set self.parent_categories['Computer science].add('Computing')
                if category is not None:
                    p_category = None
                    # sometimes parent category title is written like [[Category:Computing]],
                    # sometimes like {{music-stub}}</text> , we need to handle all 2 possibilities
                    if WikiReader.CATEGORY_SUBSTR in text:
                        p_category = re.match(WikiReader.CATEGORY_RE, text)
                    elif WikiReader.CATEGORY_SUBSTR2 in text:
                        p_category = re.match(WikiReader.CATEGORY_RE2, text)
                    if p_category is not None:
                        assert len(p_category.groups())>=1
                        parent_name = p_category.groups(0)[0].strip()
                        self.all_categories.add(parent_name)
                        self.parent_categories[category].add(parent_name)

        print 'Read %d lines, %d categories' % (num_lines, len(self.all_categories))

    # for simplewiki read, use_categories is self.use_categories, which is all_categories created by above method
    def read_pages_in_categories(self, wikifile, use_categories=None, max_read_lines=None):
        print '=> Reading pages in %s categories' %  ('ALL' if use_categories is None else len(use_categories))
        pages_in_categories = set()
        page_name, page_categories = None, set()
        num_lines = 0
        with open(wikifile,'rt') as infile:
            for text in infile:
                if len(text)==0:
                    print 'Reached End Of File'
                    break # EOF
                num_lines += 1
                if (max_read_lines is not None) and (num_lines > max_read_lines):
                    break
                if num_lines % 1000000 == 0:
                    print 'Read %d lines, %d pages so far...' % (num_lines, len(pages_in_categories))
                    gc.collect()
                # NEW_PAGE_SUBSTR is '<title>'
                if WikiReader.NEW_PAGE_SUBSTR in text:
                    new_page = re.findall(WikiReader.NEW_PAGE_RE, text)
                    if len(new_page)>0:
                        assert len(new_page)==1
                        # Check previous page
                        if len(page_categories) > 0:
                            assert page_name is not None
                            pages_in_categories.add(page_name)
                        page_name = new_page[0]
                        page_categories = set()
                        continue
                category = None
                if WikiReader.CATEGORY_SUBSTR in text:
                    category = re.match(WikiReader.CATEGORY_RE, text)
                elif WikiReader.CATEGORY_SUBSTR2 in text:
                    category = re.match(WikiReader.CATEGORY_RE2, text)
                if category is not None:
                    assert len(category.groups())>=1
                    assert page_name is not None
                    cat_name = category.groups(0)[0].strip()
                    if (re.search(WikiReader.IGNORE_PAGES, page_name) is None) and ((use_categories is None) or (cat_name in use_categories)):
                        page_categories.add(cat_name)

        # Check last page
        # page_categories does not have much to do in this method, it is only checked that len(page_categories) > 0
        # this method returns pages_in_categories, which add all page name in format <title>page_name</title>
        if len(page_categories) > 0:
            assert page_name is not None
            pages_in_categories.add(page_name)

        print 'Read %d lines, %d pages in %d categories' % (num_lines, len(pages_in_categories), len(use_categories))
        return pages_in_categories

    @staticmethod
    def text_replcaments(text):
        for sr,rr in WikiReader.RE_REMOVALS0:
            if sr in text:
                text = re.sub(rr, ' ', text)
        for bc in WikiReader.BLANKS_CONVERSIONS:
            text = text.replace(bc, ' ')
#         print '----------->'
#         print '%s' % text
        for oc,cc in WikiReader.OTHER_CONVERSIONS:
            text = text.replace(oc, cc)
        # DEAL WITH HIGH ASCIIs...
        if True:
            text = ''.join([(c if ord(c)<128 else ' ') for c in text])
        else:
            if len([c for c in text if ord(c)>=128])>0:
                print '%s' % ''.join([(c if ord(c)<128 else ' chr(%d) '%ord(c)) for c in text])
                assert False, 'chr > 127'
#         print '++++++'
#         print 'before re_removals: %s' % text
#         print '++++++'
        for sr,rr in WikiReader.RE_REMOVALS:
            if sr in text:
                text = re.sub(rr, ' ', text)
        for sr in WikiReader.STR_REMOVALS:
            text = text.replace(sr, '')
        return text

    def read(self, wikifile, outfile, use_pages=None, max_read_lines=None,
             only_first_section_per_page=False, max_sections_per_page=99999999,
             parser = SimpleWordParser(tolower=True, ascii_conversion=True, ignore_special_words=False),
             stop_words=set(), pos_words=set(),
             page_name_word_sets=None, corpus_words=None,
             min_pos_words_in_page_name=1, min_pos_words_in_section=5,
             use_all_pages_match_pos_word=False, use_all_pages_match_sets=False, always_use_first_section=False,
             action='write'):
        print '=> Reading Wiki corpus (%s)' % self.wiki_type
        if use_pages is not None:
            print 'Using set of %d pages' % len(use_pages)

        self._reset(outfile=outfile, stop_words=stop_words, pos_words=pos_words, page_name_word_sets=page_name_word_sets, corpus_words=corpus_words,
                    min_pos_words_in_page_name=min_pos_words_in_page_name, min_pos_words_in_section=min_pos_words_in_section,
                    use_all_pages_match_pos_word=use_all_pages_match_pos_word, use_all_pages_match_sets=use_all_pages_match_sets,
                    always_use_first_section=always_use_first_section,
                    action=action)

        skip_lines_first_1char  = set(['\n']) #,'=','{','}','|','!',';','#']) #,' ','*'])
        skip_lines_first_6chars = set(['<media', '[[File', '[[Imag', '[[Cate'])
        content_lines_re = re.compile('^([^<])|([<]text)')
        if self.wiki_type == 'wiki':
            skip_lines_first_1char.update(['=','{','}','|','!',';','#',' ','*'])

        # if the action is 'locdic', this will initiate a location dictionary
        # add_section in later part will add words to locdic
        self._start_action()
        page_name, section_name, section_name_words, section_in_page = None, None, [], 0
        page_name_words, section_words, section_text = [], [], ''
        num_sections_added_in_page = 0
        skip_page = True
        start_time = time.time()
        num_lines = 0
        with open(wikifile,'rt') as infile:
            for text in infile:
                if len(text)==0:
                    print 'Reached End Of File'
                    break # EOF
                num_lines += 1
                if (max_read_lines is not None) and (num_lines > max_read_lines):
                    break
                if num_lines % 1000000 == 0:
                    print 'Read %d lines, %d pages, %d sections so far, %d sections actioned...' % (num_lines, self.num_pages, self.num_sections, self.num_section_action)
                    gc.collect()
                if WikiReader.NEW_PAGE_SUBSTR in text:
                    new_page = re.findall(WikiReader.NEW_PAGE_RE, text)
                    if len(new_page)>0:
                        assert len(new_page)==1
                        # Add last section from previous page
                        if (not skip_page) and ((not only_first_section_per_page) or (section_in_page == 1)) and (num_sections_added_in_page < max_sections_per_page):
                            section_text = WikiReader.text_replcaments(section_text)
                            section_words = parser.parse(section_text)
                            num_sections_added_in_page += self._add_section(page_name, page_name_words, section_name, section_name_words, section_in_page, section_words)
                        page_name = new_page[0]
                        page_name_words = parser.parse(page_name)
                        section_in_page, section_name, section_name_words, section_words, section_text = 1, '', [], [], ''
                        num_sections_added_in_page = 0
                        skip_page = (re.search(WikiReader.IGNORE_PAGES, page_name) is not None) or ((use_pages is not None) and (page_name not in use_pages))
                        skip_page = skip_page or (not self._check_page_name(page_name, page_name_words))
                        if not skip_page:
                            self._add_page(page_name, page_name_words)
                        continue
                if skip_page: continue
                if (section_in_page == 1) and (len(section_words) == 0) and (WikiReader.PAGE_REDIRECT_SUBSTR in text):
                    if re.match(WikiReader.PAGE_REDIRECT_RE, text):
                        skip_page = True
                if skip_page: continue
                if WikiReader.NEW_SECTION_SUBSTR in text:
                    new_section = re.match(WikiReader.NEW_SECTION_RE, text)
                    if new_section is not None:
                        assert len(new_section.groups())==1
                        # Add previous section
                        if ((not only_first_section_per_page) or (section_in_page == 1)) and (num_sections_added_in_page < max_sections_per_page):
                            section_text = WikiReader.text_replcaments(section_text)
                            section_words = parser.parse(section_text)
                            num_sections_added_in_page += self._add_section(page_name, page_name_words, section_name, section_name_words, section_in_page, section_words)
                        section_in_page += 1
                        section_name, section_words, section_text = new_section.groups(0)[0], [], ''
                        section_name = WikiReader.text_replcaments(section_name)
                        section_name_words = parser.parse(section_name)
                        continue
                if text[ 0] in skip_lines_first_1char: continue
                if text[:6] in skip_lines_first_6chars: continue
                if len(text) < self.min_chars_per_line: continue
                text = text.strip()
                if re.match(content_lines_re, text) is None: continue
                section_text += ' ' + text.strip() + ' '

        # Add last section
        if (not skip_page) and ((not only_first_section_per_page) or (section_in_page == 1)) and (num_sections_added_in_page < max_sections_per_page):
            section_text = WikiReader.text_replcaments(section_text)
            section_words = parser.parse(section_text)
            num_sections_added_in_page += self._add_section(page_name, page_name_words, section_name, section_name_words, section_in_page, section_words)

        end_time = time.time()
        print 'read_wiki total time = %.1f secs.' % (end_time-start_time)
        print 'Read %d lines, %d pages, %d sections; applied action on %d sections' % (num_lines, self.num_pages, self.num_sections, self.num_section_action)
        self._end_action()

        return self._locdic
