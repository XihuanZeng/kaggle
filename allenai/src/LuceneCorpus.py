import numpy as np
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

from CorpusReader import CorpusReader
from NLPUtils import AsciiConvertor


class LuceneCorpus(object):
    # to init a LuceneCorpus, we need the outputdir, which is passed as index_dir
    # we need filenames that contains one for more corpus we just created
    # we need a parser, this parser should implement function 'parse' which knows how to split, how to stem
    def __init__(self, index_dir, filenames, parser, similarity=None):
        """
        :param index_dir: where to store the Lucene index
        :param filenames: the corpus created previously. Note that the format of corpus that has been created is consistent
        :param parser: SimpleWordParser in Parser.py, where we can apply functions such as stemming
        :param similarity: We can put None here(then default Vector Space Model with TF-IDF is used) or we can use BM25 similarity to index
        :return:
        """
        self._index_dir = index_dir
        self._filenames = filenames
        self._parser = parser
        self._similarity = similarity
        lucene.initVM()
        # the WhitespaceAnalyzer split the text based on whitespace
        self._analyzer = WhitespaceAnalyzer(Version.LUCENE_CURRENT)
        self._store = SimpleFSDirectory(File(self._index_dir))
        self._searcher = None

    def prp_index(self):
        '''
        Prepare the index given our "corpus" file(s)
        '''
        print '=> Preparing Lucene index %s' % self._index_dir
        writer = self._get_writer(create=True)
        print '   Currently %d docs (dir %s)' % (writer.numDocs(), self._index_dir)
        num_pages, num_sections = 0, 0
        page_name, section_name = None, None
        num_lines = 0
        for ifname,fname in enumerate(self._filenames):
            print '   Adding lines to index from file #%d: %s' % (ifname, fname)
            with open(fname,'rt') as infile:
                for text in infile:
                    if len(text)==0:
                        print 'Reached EOF'
                        break # EOF
                    # CorpusReader.PAGE_NAME_PREFIX is <Page>
                    # all our corpus we manipulated them to have this tag as the start of a page
                    if text.startswith(CorpusReader.PAGE_NAME_PREFIX):
                        page_name = text[len(CorpusReader.PAGE_NAME_PREFIX):].strip()
                        section_name = None
                        num_pages += 1
                    elif text.startswith(CorpusReader.SECTION_NAME_PREFIX):
                        section_name = text[len(CorpusReader.SECTION_NAME_PREFIX):].strip()
                        num_sections += 1
                    else:
                        assert (page_name is not None) and (section_name is not None)
                        if self._parser is None:
                            luc_text = text
                        else:
                            # note in our case the we always have SimpleWordParser
                            section_words = self._parser.parse(text, calc_weights=False) #True)
                            luc_text = ' '.join(section_words)
                        # for each section, we add the whole section to Lucene index, we store the text and makes it searchable
                        # seems like page is not necessary here since we do not add document page by page but section by section
                        doc = Document()
                        # there is only one field for each document, which is the text field
                        # section_name is not used as a field
                        doc.add(Field("text", luc_text, Field.Store.YES, Field.Index.ANALYZED))
                        writer.addDocument(doc)
                    num_lines += 1
                    if num_lines % 100000 == 0:
                        print '    read %d lines so far: %d pages, %d sections' % (num_lines, num_pages, num_sections)

        print '   Finished - %d docs (dir %s)' % (writer.numDocs(), self._index_dir)
        writer.close()

    def search(self, words, max_docs, weight_func=lambda n: np.ones(n), score_func=lambda s: s):
        '''
        Search the index for the given words, return total score
        '''
        searcher = self._get_searcher()
        if type(words)==str:
            search_text = words
            search_text = AsciiConvertor.convert(search_text)
            for c in '/+-&|!(){}[]^"~*?:':
                search_text = search_text.replace('%s'%c, '\%s'%c)
        else:
            search_text = ' '.join(words)
        print 'search_text: %s' % search_text
        # note that whatever parser that we put as our argument, eventually when searching with query, we will use Lucene parser to split query words
        query = QueryParser(Version.LUCENE_CURRENT, "text", self._analyzer).parse(search_text)
        hits = searcher.search(query, max_docs)

        score_sum = 0.0
        weights = weight_func(len(hits.scoreDocs))
        for hit,weight in zip(hits.scoreDocs, weights):
            score_sum += weight * score_func(hit.score)
        return score_sum

    def _get_writer(self, analyzer=None, create=False):
        config = IndexWriterConfig(Version.LUCENE_CURRENT, self._analyzer)
        if create:
            config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
        if self._similarity is not None:
            config.setSimilarity(self._similarity)
        writer = IndexWriter(self._store, config)
        return writer

    def _get_searcher(self):
        if self._searcher is None:
            self._searcher = IndexSearcher(DirectoryReader.open(self._store))
            if self._similarity is not None:
                self._searcher.setSimilarity(self._similarity)
        return self._searcher

