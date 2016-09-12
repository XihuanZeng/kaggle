import numpy as np
import time
import re
import os
from bs4 import BeautifulSoup
from CorpusReader import CorpusReader
from Parser import SimpleWordParser


class HtmlReader(CorpusReader):
    '''
    HtmlReader - read an HTML corpus
    '''
    # this re matches all the images tags
    # e.g <img src="ck12_1_files/20151013042023840136.jpeg" id="x-ck12-Qy1NUy1MUy0wMS0wMS0wMS1DaGltcGFuemVlLW1vbS1iYWJ5" title="An adult and infant chimapanzee" alt="An adult and infant chimapanzee" longdesc="An%20adult%20and%20infant%20chimpanzee%20%28%3Cem%3EPan%20troglodytes%3C/em%3E%29." />
    # this will match line like this and also the word in alt attribute, which is the alternative text describing the image
    RE_SUBSTITUTE = [re.compile(rk) for rk in ['<img [^>]*alt="([^\"]*)"[^>]*>']]

    # 'a' attribute defines hyperlink, so this will remove all the hyperlinks, remove all the <whatever> symbols and left with plain text
    # note that the order of remove make sense, we first search to remove hyperlinks, by that time, the html tags <> should remain
    RE_REMOVE     = [re.compile(rr) for rr in ['<a [^>]+>[^<]*<[\/]a>', '<a [^>]+>\s*<strong>[^<]*<[\/]strong>\s*<[\/]a>',
                                               '<[^>]*>', 'http[\:][^\"\>]+[\"\> ]']]

    # some of the symbols in the html file is denoted by dashes, such as &#8212 denote '-'
    # [('&#%3d;'%x,'A') for x in range(192,199)] -> [('&#192;', 'A'), ('&#193;', 'A'), ('&#194;', 'A'), ('&#195;', 'A'), ('&#196;', 'A'), ('&#197;', 'A'), ('&#198;', 'A')]
    # these are different alphabet that all equivalent to A in English
    RE_REPLACE = [(re.compile(rr),rs) for rr,rs in [
                    ('&#8216;', "'"), ('&#8217;', "'"), ('&#8220;', '"'), ('&#8221;', '"'),
                    ('&quot;', '"'), ('&lsquo;', "'"), ('&rsquo;', "'"), ('&ldquo;', '"'), ('&rdquo;', '"'),
                    ('&#8206;', '.'), ('&#8195', '.'), ('&#8230;', ' ... '),
                    ('&#8212;', ' - '), ('&#8211;', ' - '), ('&#8722;', ' - '),
                    ('&#8594;', ' - '), ('&#8592;', ' - '), ('&#8596;', ' - '), ('#8595;', ' - '), ('&#8593;', ' - '), # various arrows
                    ('&#8804;', ' '), # <=
                    ('&#8801;', ' '), # = with 3 lines
                    ('&#730;', ' degrees '),
                    ('&nbsp;', ' '), ('&deg;', ' degrees '), ('&#8203;', ''), ('&#9786;', ''),
                    ('&#38;', '&'), ('&#8226;', ' '), ('&#9702;', ' '), ('&#8729;', ' '), ('&#8227;', ' '), ('&#8259;', ' '), ('&#176;', ' degrees '), ('&#8734;', ' infinity '),
                    ('&#36;', '$'), ('&#8364;', ' euro '), ('&#163;', ' pound '), ('&#165;', ' yen '), ('&#162;', ' cent '),
                    ('&#169;', ' '), ('&#174;', ' '), ('&#8471;', ' '), ('&#8482;', ' '), ('&#8480;', ' '),
                    ('&#945;', 'alpha'), ('&#946;', 'beta'), ('&#947;', 'gamma'), ('&#948;', 'delta'), ('&#949;', 'epsilon'), ('&#950;', 'zeta'),
                    ('&#951;', 'eta'), ('&#952;', 'theta'), ('&#953;', 'iota'), ('&#954;', 'kappa'), ('&#955;', 'lambda'), ('&#956;', 'mu'), ('&#957;', 'nu'),
                    ('&#958;', 'xi'), ('&#959;', 'omicron'), ('&#960;', 'pi'), ('&#961;', 'rho'), ('&#963;', 'sigma'), ('&#964;', 'tau'), ('&#965;', 'upsilon'),
                    ('&#966;', 'phi'), ('&#967;', 'chi'), ('&#968;', 'psi'), ('&#969;', 'omega'), ('&#913;', 'Alpha'), ('&#914;', 'Beta'), ('&#915;', 'Gamma'),
                    ('&#916;', 'Delta'), ('&#917;', 'Epsilon'), ('&#918;', 'Zeta'), ('&#919;', 'Eta'), ('&#920;', 'Theta'), ('&#921;', 'Iota'), ('&#922;', 'Kappa'),
                    ('&#923;', 'Lambda'), ('&#924;', 'Mu'), ('&#925;', 'Nu'), ('&#926;', 'Xi'), ('&#927;', 'Omicron'), ('&#928;', 'Pi'), ('&#929;', 'Rho'),
                    ('&#931;', 'Sigma'), ('&#932;', 'Tau'), ('&#933;', 'Upsilon'), ('&#934;', 'Phi'), ('&#935;', 'Chi'), ('&#936;', 'Psi'), ('&#937;', 'Omega'),
                    ('&#186;', ' '), ('&#180;', '`'),
                    ('&#189;', ' 1/2'), ('&#188;', '1/4'), ('&#190;', '3/4'),
                    ('&#181;', 'm'),  ('&#299;', 'i'), ('&#333;', 'o'), ('&#257;', 'a'),
                    ('&#215;', '*'), ('&#8226;', '*'), ('&#183;', '*'), ('&#247;', '/')] +
                   [('&#%3d;'%x,'A') for x in range(192,199)] + [('&#%3d;'%x,'C') for x in range(199,200)] + [('&#%3d;'%x,'E') for x in range(200,204)] +
                   [('&#%3d;'%x,'I') for x in range(204,208)] + [('&#%3d;'%x,'D') for x in range(208,209)] + [('&#%3d;'%x,'N') for x in range(209,210)] +
                   [('&#%3d;'%x,'O') for x in range(210,215)+[216]] + [('&#%3d;'%x,'U') for x in range(217,221)] + [('&#%3d;'%x,'Y') for x in range(221,222)] +
                   [('&#%3d;'%x,'S') for x in range(223,224)] + [('&#%3d;'%x,'a') for x in range(224,231)] + [('&#%3d;'%x,'c') for x in range(231,232)] +
                   [('&#%3d;'%x,'e') for x in range(232,236)] + [('&#%3d;'%x,'i') for x in range(236,240)] + [('&#%3d;'%x,'n') for x in range(241,242)] +
                   [('&#%3d;'%x,'o') for x in range(242,247)+[248]] + [('&#%3d;'%x,'u') for x in range(249,253)] + [('&#%3d;'%x,'y') for x in [253,255]]
                   ]

    @staticmethod
    def parse_text(text):
        for rk in HtmlReader.RE_SUBSTITUTE:
            found = True
            while found:
                rkm = re.search(rk, text)
                if rkm is None:
                    found = False
                else:
                    text = text[:rkm.start()] + rkm.groups(0)[0] + text[rkm.end():]
        for rr in HtmlReader.RE_REMOVE:
            text = re.sub(rr, ' ', text)
        for rr,rs in HtmlReader.RE_REPLACE:
            text = re.sub(rr, rs, text)
        return text

    def read(self, htmldir, outfile, stop_words=set(), pos_words=set(), page_name_word_sets=None, corpus_words=None,
             page_title_ignore_suffixes=['-1', '-2', '- Advanced'],
             ignore_sections=set(),
             min_pos_words_in_page_name=0, min_pos_words_in_section=0,
             use_all_pages_match_pos_word=False, use_all_pages_match_sets=False, always_use_first_section=False,
             action='write'):

        # reset the class variables every time since these class variables are static variables that belongs to the Class, not a particular class object
        self._reset(outfile=outfile, stop_words=stop_words, pos_words=pos_words, page_name_word_sets=page_name_word_sets, corpus_words=corpus_words,
                    min_pos_words_in_page_name=min_pos_words_in_page_name, min_pos_words_in_section=min_pos_words_in_section,
                    use_all_pages_match_pos_word=use_all_pages_match_pos_word, use_all_pages_match_sets=use_all_pages_match_sets,
                    always_use_first_section=always_use_first_section,
                    action=action)

        parser = SimpleWordParser(tolower=True, ascii_conversion=True, ignore_special_words=False)
        # the action variable is 'write', so _start_action will open the output file and write to it
        self._start_action()
        page_name, section_name, section_in_page = None, None, 0
        page_name_words, section_words = [], []
        start_time = time.time()
        # we only include x.html while x is a scalar, meaning we ignore the table html
        filenames = ['%s/%s'%(htmldir,fname) for fname in os.listdir(htmldir) if re.match(r'(\d+).html', fname) != None]
        assert len(filenames)>0
        for ifname,fname in enumerate(filenames):
            print 'Reading %s' % fname
            with open(fname, 'rb') as myfile:
                # this is a very long string
                text = myfile.read()
            soup = BeautifulSoup(text, 'lxml')
            if soup.h1 is None:
                print 'Could not find page title in file %s - skipping' % fname
                continue
            # note that the html file could have many h1 tags, while only the first one is the title
            page_name = soup.h1.text.strip()
            # e.g some of the page name has Momentum-1, where the suffix '-1' should be eliminated
            for ptis in page_title_ignore_suffixes:
                if page_name.endswith(ptis):
                    page_name = page_name[:-len(ptis)]
                    break
            page_name_words = parser.parse(page_name)
            # page name = surface processes and landforms __0
            # this is write fo file with the page name
            page_name = CorpusReader.part_name_from_words(page_name_words, ifname)
            print 'page name = %s' % page_name
            self._add_page(page_name, page_name_words)
            # using the section_re to split the text(without title)
            parts = re.split('(<h[1-4])', text)
            # start from 3 because the first 3 parts belong to the title <h1> tag, which should be skipped
            for ipart in range(3,len(parts),2):
                # odd number of parts are splitter tags
                # even number of parts are the contents of the tag
                soup = BeautifulSoup(parts[ipart] + parts[ipart+1], 'lxml')
                section_name = soup.find(parts[ipart][1:]).text.strip().lower()
                # some section that has name that matches set(['review', 'practice', 'references', 'explore more.*'])
                # we know this is a review section that does not contains information about science knowledge
                if np.any([(re.match(isr, section_name) is not None) for isr in ignore_sections]):
                    continue
                section_name_words = parser.parse(section_name)
                section_in_page = (ipart - 1) / 2
                # only select text from all the <p> tags within each section
                text = ''
                for p in soup.find_all('p'):
                    text += p.next.strip()
                # this will replace some of the symbols to Eng, e.g '&#916;' -> 'Delta'
                text = HtmlReader.parse_text(text)
                # word tokenizing
                words = parser.parse(text)
                section_words = words
                # for each filename, add those sections, which is write to files
                # note that section_name is not written to file.
                self._add_section(page_name, page_name_words, section_name, section_name_words, section_in_page, section_words)

        end_time = time.time()
        print 'read_html total time = %.1f secs.' % (end_time-start_time)
        print 'Read %d pages, %d sections; applied action on %d sections' % (self.num_pages, self.num_sections, self.num_section_action)
        self._end_action()

        return self._locdic

