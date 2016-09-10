# We need word parsers(tokenizer) to tokenize the sentences that contains ASCII characters(such as celsius)
# in fact, only 3 questions has ASCII char that need to take care of.
# e.g: 'The human body has an average, normal temperature of about 98.6\xc2\xb0F'
import numpy as np
import re
from NLPUtils import AsciiConvertor, SpecialWords

class WordParser(object):
    '''
    WordParser - base class for parsers
    '''
    def __init__(self, min_word_length=2, max_word_length=25, ascii_conversion=True):
        """
        :param min_word_length: a word is treat as a word if it is at least 2 char long
        :param max_word_length: a word is treat as a word if it is at most 25 char long
        :param ascii_conversion: whether we first transfer all char to utf-8
        :return:
        """
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        self.ascii_conversion = ascii_conversion
    def filter_words_by_length(self, words):
        return [word for word in words if len(word) >= self.min_word_length and len(word) <= self.max_word_length]
    def convert_ascii(self, text):
        if self.ascii_conversion:
            return AsciiConvertor.convert(text)
        return text
    def parse(self, text):
        # this is the key function to be implemented by subclass
        return text


class SimpleWordParser(WordParser):
    '''
    SimpleWordParser - supports tuples
    e.g parser = SimpleWordParser(tuples = [1,2])
    parser.parse('today is a beautiful sunny day') ->
    ['today',
     'beautiful',
     'sunny',
     'day',
     'today beautiful',
     'beautiful sunny',
     'sunny day']
    '''
    def __init__(self, stop_regexp = '[\-\+\*_\.\:\,\;\?\!\'\"\`\\\/\)\]\}]+ | [\*\:\;\'\"\`\(\[\{]+|[ \t\r\n\?]',
                 min_word_length = 2, word_func = None, tolower = True, ascii_conversion = True,
                 ignore_special_words = True, split_words_regexp = None, tuples = [1]):
        """
        :param stop_regexp: word tokenizer
        :param min_word_length: a word is treat as a word if it is at least 2 char long
        :param word_func:
        :param tolower: whether we first lower all the words, this is always set True
        :param ascii_conversion: whether we convert all char to utf-8, always set True, refer to NLPUtils.Ascii.Converter
        :param ignore_special_words: whether we skip all special words, refer to NLPUtils.SpecialWords
        :param split_words_regexp: within each word, if you still wants to split, mostly set to None
        :param tuples:  if [1], we get single words
                        if [1,2], we get single words + 2-gram
                        ...
        :return:
        """
        self.stop_regexp = re.compile(stop_regexp)
        self.word_func = word_func
        self.tolower = tolower
        self.ignore_special_words = ignore_special_words
        self.split_words_regexp = None if split_words_regexp is None else re.compile(split_words_regexp)
        self.tuples = tuples
        assert set([1,2,3,4]).issuperset(self.tuples)
        WordParser.__init__(self, min_word_length=min_word_length, ascii_conversion=ascii_conversion)

    def parse(self, text, calc_weights=False):
        if self.tolower:
            text = text.lower()
        text = ' ' + text.strip() + ' ' # add ' ' at the beginning and at the end so that, eg, a '.' at the end of the text will be removed, and "'''" at the beginning will be removed
        text = self.convert_ascii(text)
        words = re.split(self.stop_regexp, text)
        if self.split_words_regexp is not None:
            swords = []
            for word in words:
                w_words = re.split(self.split_words_regexp, word)
                if len(w_words) == 1:
                    swords.append(w_words[0])
                else:
                    if np.all([len(w)>=self.min_word_length for w in w_words]):
                        swords += w_words
                    else:
                        swords.append(word) # don't split - some parts are too short
            words = swords
        if self.ignore_special_words:
            words = SpecialWords.filter(words)
        if self.word_func is not None:
            fwords = []
            for word in words:
                try:
                    fword = str(self.word_func(word))
                except UnicodeDecodeError:
                    fword = word
                fwords.append(fword)
            words = fwords
        words = self.filter_words_by_length(words)
        ret_words = []
        if 1 in self.tuples:
            ret_words += words
        if 2 in self.tuples:
            ret_words += ['%s %s'%(words[i],words[i+1]) for i in range(len(words)-1)]
        if 3 in self.tuples:
            ret_words += ['%s %s %s'%(words[i],words[i+1],words[i+2]) for i in range(len(words)-2)]
            if 2 in self.tuples:
                ret_words += ['%s %s'%(words[i],words[i+2]) for i in range(len(words)-2)]
        if 4 in self.tuples:
            ret_words += ['%s %s %s %s'%(words[i],words[i+1],words[i+2],words[i+3]) for i in range(len(words)-3)]
            if 3 in self.tuples:
                ret_words += ['%s %s %s'%(words[i],words[i+2],words[i+3]) for i in range(len(words)-3)]
                ret_words += ['%s %s %s'%(words[i],words[i+1],words[i+3]) for i in range(len(words)-3)]
            if 2 in self.tuples:
                ret_words += ['%s %s'%(words[i],words[i+3]) for i in range(len(words)-3)]
                if 3 not in self.tuples:
                    ret_words += ['%s %s'%(words[i],words[i+2]) for i in range(len(words)-2)]
        if calc_weights:
            return ret_words, {}
        else:
            return ret_words


