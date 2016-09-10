from nltk.corpus import stopwords

class AsciiConvertor(object):
    """
    note that extended ASCII contains some latin alphabet
    this converts some non-english alphabet to english alphabet
    to use it, AsciiConvertor.convert(text_with_Latin) = 'word_with_English'

    a = word_with_Latin
    b = a.decode('utf-8')

    for i in b:
        print AsciiConvertor.ascii_mapping[ord(i)]
    """
    ascii_orig = ['0','1','2','3','4','5','6','7','8','9',
                  'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
                  'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
                  '+','-','=','*','/','\\','_','~','>','<','%','$','#','@','&',
                  '.',',',';',':','!','?',
                  '\'']
    ascii_conv = {138: 's', 140: 'o', 142: 'z',
                  150: '-', 151: '-', 152: '~', 154: 's', 156: 'o', 158: 'z', 159: 'y',
                  192: 'a', 193: 'a', 194: 'a', 195: 'a', 196: 'a', 197: 'a', 198: 'a', 199: 'c', 200: 'e', 201: 'e', 202: 'e', 203: 'e', 204: 'i', 205: 'i',
                  206: 'i', 207: 'i', 209: 'n', 210: 'o', 211: 'o', 212: 'o', 213: 'o', 214: 'o', 215: '*', 216: 'o', 217: 'u', 218: 'u', 219: 'u', 220: 'u',
                  221: 'y', 223: 's', 224: 'a', 225: 'a', 226: 'a', 227: 'a', 228: 'a', 229: 'a', 230: 'a', 231: 'c', 232: 'e', 233: 'e', 234: 'e', 235: 'e',
                  236: 'i', 237: 'i', 238: 'i', 239: 'i', 241: 'n', 242: 'o', 243: 'o', 244: 'o', 245: 'o', 246: 'o', 248: 'o', 249: 'u', 250: 'u',
                  250: 'u', 251: 'u', 252: 'u', 253: 'y', 255: 'y'
                  }
    ascii_mapping = None

    @staticmethod
    def convert(text):
        if AsciiConvertor.ascii_mapping is None:
            print 'Building ascii dict'
            AsciiConvertor.ascii_mapping = [' ']*256
            for c in AsciiConvertor.ascii_orig:
                AsciiConvertor.ascii_mapping[ord(c)] = c
            for oc,c in AsciiConvertor.ascii_conv.iteritems():
                AsciiConvertor.ascii_mapping[oc] = c
        return ''.join(map(lambda c: AsciiConvertor.ascii_mapping[ord(c)], text))


class SpecialWords(object):
    '''
    Stop words
    this class can filter out stop words from a list of strings.
    Stopwords are from 2 sources, one is given by the author, the other is from nltk stopwords

    usage:
    SpecialWords.filter(['I','hate','it','very','much'])   --->   ['I', 'hate', 'much']
    '''

    ignore_words = set(stopwords.words('english'))
    # some of the stop words in nltk should not be ignored
    ignore_words.difference_update(['above','after','again','against','all','any','before','below','between','both','down','during',
                                    'each','few','further','into','just','more','most','no','not','now','off','once','only','out','over','own',
                                    'same','through','under','until','up'])

    @staticmethod
    def filter(words):
        """
        :param words: a list of str
        :return:
        """
        fwords = [word for word in words if word not in SpecialWords.ignore_words]
        if len(fwords) > 0:
            return fwords
        else:
            return words

