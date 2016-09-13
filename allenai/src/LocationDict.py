import numpy as np
import scipy
import scipy.optimize
import time
import re
import os

from Parser import SimpleWordParser

class LocationDictionary(object):
    def __init__(self, save_locations=False, doc_name_weight=0, base_locdic=None):
        self.save_locations = save_locations
        self.doc_name_weight = doc_name_weight
        self.reset()
        self.set_search_word_filter()
        if base_locdic is not None:
            self.copy(base_locdic)

    def reset(self):
        self.doc_ids, self.doc_names, self.doc_name_words = {}, [], []
        self.word_ids = {}
        self.word_counts = []
        self.word_locations, self.word_doc_names = [], []
        self.doc_lengths, self.doc_name_lengths = [], []
        self.doc_unique_words, self.doc_name_unique_words = [], []
        self.total_count = 0.0
        self._cache = {}
        self._cache_keys = None
        self._search_cache = None
        self._tf_idf_cache = None

    def copy(self, base_locdic):
        self.doc_ids = base_locdic.doc_ids.copy()
        self.doc_names = list(base_locdic.doc_names)
        self.doc_name_words = list(base_locdic.doc_name_words)
        self.word_ids = base_locdic.word_ids.copy()
        self.word_counts = list(base_locdic.word_counts)
        self.word_locations = [wl.copy() for wl in base_locdic.word_locations]
        self.word_doc_names = [wd.copy() for wd in base_locdic.word_doc_names]
        self.doc_lengths = list(base_locdic.doc_lengths)
        self.doc_name_lengths = list(base_locdic.doc_name_lengths)
        self.doc_unique_words = list(base_locdic.doc_unique_words)
        self.doc_name_unique_words = list(base_locdic.doc_name_unique_words)
        self.total_count = base_locdic.total_count

    def add_words(self, doc_name, doc_name_words, words):
#         print 'Adding words: %s' % ' ; '.join(words)
        if self.doc_ids.has_key(doc_name):
            assert self.doc_name_weight == 0
            assert 'Should not be here?!...'
        else:
            self.doc_ids[doc_name] = len(self.doc_ids)
            self.doc_names.append(doc_name)
            words = doc_name_words + words # use the doc's name as part of the doc's content (important also for search)
            self.doc_lengths.append(0)
            self.doc_unique_words.append(0)
            if self.doc_name_weight != 0:
                self.doc_name_words.append(doc_name_words)
                self.doc_name_lengths.append(0)
                self.doc_name_unique_words.append(0)
        doc_id = self.doc_ids[doc_name]
        self.doc_lengths[doc_id] += len(words)
        if self.doc_name_weight != 0:
            self.doc_name_lengths[doc_id] += len(doc_name_words)
        for iw,word in enumerate(words):
            if not self.word_ids.has_key(word):
                self.word_ids[word] = len(self.word_ids)
                self.word_counts.append(0)
                self.word_locations.append({})
                if self.doc_name_weight != 0:
                    self.word_doc_names.append({})
            word_id = self.word_ids[word]
            self.word_counts[word_id] += 1
            self.total_count += 1.0
            if self.save_locations:
                if not self.word_locations[word_id].has_key(doc_id):
                    self.word_locations[word_id][doc_id] = []
                    self.doc_unique_words[doc_id] += 1
                self.word_locations[word_id][doc_id].append(iw)
            else:
                if not self.word_locations[word_id].has_key(doc_id):
                    self.word_locations[word_id][doc_id] = 0 # we save the number of occurrences, not the list of locations...
                    self.doc_unique_words[doc_id] += 1
                self.word_locations[word_id][doc_id] += 1
        if self.doc_name_weight != 0:
            for iw,word in enumerate(doc_name_words):
                word_id = self.word_ids[word] # should already be here, since doc name words are part of the doc's words
                if not self.word_doc_names[word_id].has_key(doc_id):
                    self.word_doc_names[word_id][doc_id] = 0
                    self.doc_name_unique_words[doc_id] += 1
                self.word_doc_names[word_id][doc_id] += 1

    def get_word(self, word_id):
        return [word for word,wid in self.word_ids.iteritems() if wid==word_id][0]

    def get_num_docs(self):
        return len(self.doc_ids)

    def get_word_num_docs(self, word_id):
        return len(self.word_locations[word_id])

    def get_word_num_doc_names(self, word_id):
        return len(self.word_doc_names[word_id])

    def get_word_tf_idf(self, word_id, doc_id, tf_log_flag=False, doc_name_flag=False):
        if doc_name_flag:
            doc_len = self.doc_name_lengths[doc_id]
            word_count_in_doc = self.word_doc_names[word_id].get(doc_id, 0)
            num_doc_unique_words = self.doc_name_unique_words[doc_id]
            word_num_docs = self.get_word_num_doc_names(word_id)
            if doc_len==0 or word_count_in_doc==0:
                return 0.0
        else:
            doc_len = self.doc_lengths[doc_id]
            word_count_in_doc = len(self.word_locations[word_id][doc_id]) if self.save_locations else self.word_locations[word_id][doc_id]
            num_doc_unique_words = self.doc_unique_words[doc_id]
            word_num_docs = self.get_word_num_docs(word_id)
        assert word_count_in_doc>=1 and doc_len>=1
        if tf_log_flag:
            tf = np.log(1.0 + word_count_in_doc + 0.5/num_doc_unique_words) / np.log(1.0 + doc_len + 0.5)
        else:
            tf = (word_count_in_doc + 0.5/num_doc_unique_words) / (doc_len + 0.5)
        idf = np.log((self.get_num_docs() + 0.5) / (word_num_docs + 0.5))
        return tf*idf

    def sort_words_by_num_docs(self):
        '''
        Sort the number of docs each word appears in (sorted in descending order)
        '''
        return sorted([(self.get_word_num_docs(wid),word) for word,wid in self.word_ids.iteritems()], reverse=True)

    def sort_words_by_count(self):
        '''
        Sort the word by their count (in descending order)
        '''
        words = sorted(self.word_ids.keys(), key=lambda w: self.word_ids[w])
        return np.take(words, np.argsort(self.word_counts)[::-1])

    def set_search_word_filter(self, min_word_docs_frac=0, max_word_docs_frac=0.1, min_word_count_frac=0, max_word_count_frac=0.01):
        '''
        Set the min/max fraction of documents that each search word may appear in
        '''
        assert min_word_docs_frac <= max_word_docs_frac and min_word_count_frac <= max_word_count_frac
        self.min_word_docs_frac = min_word_docs_frac
        self.max_word_docs_frac = max_word_docs_frac
        self.min_word_count_frac = min_word_count_frac
        self.max_word_count_frac = max_word_count_frac

    def _check_cache(self):
        if self._cache_keys != (self.total_count, self.min_word_docs_frac, self.max_word_docs_frac, self.min_word_count_frac, self.max_word_count_frac):
            # (Re-)compute cache
            print '=> Computing cache'
            self._cache_keys = (self.total_count, self.min_word_docs_frac, self.max_word_docs_frac, self.min_word_count_frac, self.max_word_count_frac)
            self._cache = {}
            self._cache['is_word_filtered'] = [self._filter_word_for_search(wid) for wid in range(len(self.word_ids))]
            print '   Total %d filtered words: %s' % (np.sum(self._cache['is_word_filtered']),
                                                      [word for word,wid in self.word_ids.iteritems() if self._cache['is_word_filtered'][wid]])

    def _filter_word_for_search(self, word_id):
        n_docs = self.get_num_docs() + 0.0
        n_words = self.total_count + 0.0
        return ((self.get_word_num_docs(word_id)/n_docs) < self.min_word_docs_frac or (self.get_word_num_docs(word_id)/n_docs) > self.max_word_docs_frac or
                (self.word_counts[word_id]/n_words) < self.min_word_count_frac or (self.word_counts[word_id]/n_words) > self.max_word_count_frac)

    def _is_word_filtered(self, word_id):
        return self._cache['is_word_filtered'][word_id]

    def _filter_words_for_search(self, word_ids):
        self._check_cache()
        return [wid for wid in word_ids if not self._is_word_filtered(wid)]

    def _get_random_words(self, num_words):
        '''
        Return num_words word ids, randomly chosen from the set of un-filtered words (with repetitions)
        '''
        assert False, 'Not used'
        self._check_cache()
        rwords = []
        for i in range(num_words):
            rwords.append(np.searchsorted(self._cache['word_probs'], np.random.rand(), side='left'))
        return rwords

    def search_docs(self, words, word_ids=None, words_weights=None, min_words_per_doc=None, max_words_distance=None,
                    prob_type='tf-idf', tf_log_flag=True):
        assert max_words_distance is None, 'max_words_distance not supported yet'
        assert prob_type in ['word-probs', 'tf-idf']
        if words_weights is None:
            words_weights = {} # default weight is 1
        word_ids_weights = dict([(self.word_ids[word], wgt) for word,wgt in words_weights.iteritems() if self.word_ids.has_key(word)])

        if words is not None:
            assert word_ids is None
            words = np.unique(words)
            # Get the word ids to search for
            word_ids = [self.word_ids[word] for word in words if self.word_ids.has_key(word)]
        else:
            word_ids = np.unique(word_ids)
        word_ids = self._filter_words_for_search(word_ids)
        # Ignore words with weight=0
        word_ids = [wid for wid in word_ids if word_ids_weights.get(wid,1)!=0]
        # Normalize weights
        wmean = np.mean([word_ids_weights.get(wid,1.0) for wid in word_ids])
        for wid in word_ids:
            word_ids_weights[wid] = word_ids_weights.get(wid, 1.0) / wmean
        if len(word_ids)==0:
            return dict([((0 if mwpd is None else mwpd),([],[])) for mwpd in min_words_per_doc])
        words_per_doc, words_per_doc_name = {}, {}
        for wid in word_ids:
            for did in self.word_locations[wid].keys():
                if not words_per_doc.has_key(did):
                    words_per_doc[did] = []
                words_per_doc[did].append(wid)
                if self.doc_name_weight != 0:
                    if not words_per_doc_name.has_key(did):
                        words_per_doc_name[did] = []
                    words_per_doc_name[did].append(wid)
        min_words_per_doc2 = []
        for mwpd in min_words_per_doc:
            if mwpd is None:
                mwpd = len(word_ids)
            elif mwpd < 0:
                mwpd = np.max([1, len(word_ids) + mwpd])
            elif mwpd > len(word_ids):
                continue
            min_words_per_doc2.append(mwpd)
        if len(min_words_per_doc2) == 0:
            return {}
        min_words_per_doc2 = np.unique(min_words_per_doc2)[::-1] # sort in reverse order and remove duplicates
        docs_probs = {}
        prev_doc_probs = {}
        min_mwpd = min_words_per_doc2[-1]
        for mwpd in min_words_per_doc2:
            if mwpd == 0:
                docs_probs[mwpd] = ([], []) # dummy empty sets for searching with 0 min words
                continue
            assert (mwpd > 0) and (mwpd <= len(word_ids))
            docs = [doc for doc,ws in words_per_doc.iteritems() if len(ws)>=mwpd]
            n_docs = self.get_num_docs() + 0.0
            probs = []
            for doc in docs:
                if prev_doc_probs.has_key(doc):
                    prob = prev_doc_probs[doc]
                else:
                    if prob_type == 'word-probs':
                        prob = -np.sum([np.log(word_ids_weights[wid]*(self.get_word_num_docs(wid)+0.5)/(n_docs+0.5))/len(word_ids) for wid in words_per_doc[doc]])
                        if self.doc_name_weight != 0:
                            prob += self.doc_name_weight * -np.sum([np.log(word_ids_weights[wid]*(self.get_word_num_doc_names(wid)+0.5)/(n_docs+0.5))/len(word_ids) for wid in words_per_doc_names[doc]])
                    elif prob_type == 'tf-idf':
                        prob = np.sum([word_ids_weights[wid]*self.get_word_tf_idf(wid, doc, tf_log_flag=tf_log_flag, doc_name_flag=False)/len(word_ids) for wid in words_per_doc[doc]])
                        if self.doc_name_weight != 0:
                            prob += self.doc_name_weight * np.sum([word_ids_weights[wid]*self.get_word_tf_idf(wid, doc, tf_log_flag=tf_log_flag, doc_name_flag=True)/len(word_ids) for wid in words_per_doc[doc]])
                    else:
                        raise ValueError('Unknown prob_type')
                    if mwpd > min_mwpd: # no pointing in caching values we won't be using...
                        prev_doc_probs[doc] = prob
                assert prob>=0
                probs.append(prob)
            docs_probs[mwpd] = (docs, probs)
        return docs_probs

    def double_search(self, words1, words2, num_words1=[None], num_words2=[None], words1_weights={}, words2_weights={},
                      score='hg', score_params=None, prob_type='tf-idf', tf_log_flag=True):
        assert score in ['counts','hg','weights']
        if (self._search_cache is not None) and (self._search_cache['words'] == words1) and \
           (self._search_cache['num_words'] == num_words1) and (self._search_cache['params'] == (score,prob_type,tf_log_flag)):
            docs1 = self._search_cache['docs']
        else:
            docs1 = self.search_docs(words1, words_weights=words1_weights,
                                     min_words_per_doc=num_words1, prob_type=prob_type, tf_log_flag=tf_log_flag)
            self._search_cache = {'words': words1, 'num_words': num_words1, 'params': (score,prob_type,tf_log_flag), 'docs': docs1}
        docs2 = self.search_docs(words2, words_weights=words2_weights,
                                 min_words_per_doc=num_words2, prob_type=prob_type, tf_log_flag=tf_log_flag)
        best_score_over, best_score_under = 0, 0
        for mw1,(d1,p1) in docs1.iteritems():
            if len(d1)==0 and mw1>0: continue
            for mw2,(d2,p2) in docs2.iteritems():
                if len(d2)==0: continue
                assert mw2 > 0
                if mw1 == 0:
                    d12 = d2
                else:
                    d12 = set(d1).intersection(d2)
                if score == 'hg':
                    assert len(d1) > 0
                    # Compute p-value for over/under-representation of intersection
                    score_over  = -np.log10(np.max([1E-320, hg_test(M=int(self.get_num_docs()), n=len(d1), N=len(d2), k=len(d12), dir= 1)['prob']]))
                    score_under = -np.log10(np.max([1E-320, hg_test(M=int(self.get_num_docs()), n=len(d1), N=len(d2), k=len(d12), dir=-1)['prob']]))
                elif score == 'counts':
                    score_over  = len(d12)
                    score_under = len(d1)+len(d2)-len(d12)
                elif score == 'weights':
                    if (score_params is not None) and score_params.has_key('norm'):
                        p1, p2 = score_params['norm'](np.asarray(p1)), score_params['norm'](np.asarray(p2))
                    if mw1 == 0:
                        w1 = dict(zip(d2,np.ones(len(d2)))) # use score=1 for each document in d2 (since we didn't actually search for any words from words1)
                    else:
                        w1 = dict(zip(d1,p1))
                    w2 = dict(zip(d2,p2))
                    if False:
                        print 'intersection weights:'
                        for did in d12:
                            print ' doc id %-7s: %7.4f * %7.4f = %7.4f  (%s)' % (did, w1[did], w2[did], w1[did]*w2[did], self.doc_names[did])
                    weights_intersection = sorted([w1[did]*w2[did] for did in d12], reverse=True)
                    if len(weights_intersection)==0:
                        score_over = 0.0
                    else:
                        if (score_params is not None) and score_params.has_key('coeffs'):
                            coeffs = score_params['coeffs'](len(weights_intersection))
                        else:
                            coeffs = np.ones(len(weights_intersection))
                    d2not1 = set(d2).difference(d1)
                    weights_c_intersection = sorted([w2[did] for did in d2not1], reverse=True)
                    if len(weights_c_intersection)==0:
                        score_under = 0.0
                    else:
                        if (score_params is not None) and score_params.has_key('coeffs'):
                            coeffs = score_params['coeffs'](len(weights_c_intersection))
                        else:
                            coeffs = np.ones(len(weights_c_intersection))
                        score_under = np.dot(weights_c_intersection, coeffs)
                    if (score_params is not None) and score_params.has_key('calc_over_vs_under') and (score_params['calc_over_vs_under']==True):
                        w1p, w2p = np.sum(w1.values()) / self.get_num_docs(), np.sum(w2.values()) / self.get_num_docs()
                        oexp = w1p * w2p * self.get_num_docs()
                        oscore = (np.sum(weights_intersection) - oexp) / np.sum([np.sqrt(oexp*(1.0-w1p)), np.sqrt(oexp*(1.0-w2p))])
                        score_under = scipy.stats.norm.cdf(oscore)
                assert score_over>=0 and score_under>=0
                if (score_params is not None) and score_params.has_key('minword1_coeffs'):
                    mwc1 = score_params['minword1_coeffs'](mw1, np.max(docs1.keys()))
                    mwc2 = score_params['minword2_coeffs'](mw2, np.max(docs2.keys()))
                    best_score_over  += score_over  * mwc1 * mwc2
                    best_score_under += score_under * mwc1 * mwc2
                else:
                    best_score_over  = np.max([best_score_over , score_over])
                    best_score_under = np.max([best_score_under, score_under])
        return best_score_over, best_score_under




def build_training_location_dictionary(train, parser=SimpleWordParser(),
                                       use_questions=True, use_answers=True, min_words_per_qa=1, base_locdic=None,
                                       min_word_docs_frac=0, max_word_docs_frac=0.2, min_word_count_frac=0, max_word_count_frac=0.01,
                                       ascii_conversion=True):
    parser.ascii_conversion = ascii_conversion
    locdic = LocationDictionary(doc_name_weight=0, base_locdic=base_locdic)
    locdic.set_search_word_filter(min_word_docs_frac=min_word_docs_frac, max_word_docs_frac=max_word_docs_frac,
                                  min_word_count_frac=min_word_count_frac, max_word_count_frac=max_word_count_frac)
    if use_answers:
        for i,(qid,qst,ans) in enumerate(np.array(train[['ID','question','answer']])):
            words = parser.parse(qst) if use_questions else []
            words += parser.parse(ans)
            if len(words) >= min_words_per_qa:
                locdic.add_words('train_%s_%d'%(qid,i), [], words)
    else:
        assert use_questions
        for qst,ids in train.groupby('question').groups.iteritems():
            words = parser.parse(qst)
            if len(words) >= min_words_per_qa:
                locdic.add_words('train_%s'%(train.irow(ids[0])['ID']), [], words)
    return locdic

def build_files_location_dictionary(filenames, dirname, file_suffix, part_sep='\r\n\r\n', min_words_in_part=10,
                                    parser=SimpleWordParser(),
                                    min_word_docs_frac=0, max_word_docs_frac=0.2, min_word_count_frac=0, max_word_count_frac=0.01,
                                    ascii_conversion=True):
    parser.ascii_conversion = ascii_conversion
    if filenames is None:
        filenames = ['%s/%s'%(dirname,fname) for fname in os.listdir(dirname) if fname.endswith(file_suffix)]
        assert len(filenames)>0
    locdic = LocationDictionary(doc_name_weight=0)
    locdic.set_search_word_filter(min_word_docs_frac=min_word_docs_frac, max_word_docs_frac=max_word_docs_frac,
                                  min_word_count_frac=min_word_count_frac, max_word_count_frac=max_word_count_frac)
    total_parts = 0
    for fname in filenames:
        with open (fname, 'rb') as myfile:
            text = myfile.read()#.replace('\x00', ' ')
        parts = re.split(part_sep, text)
        print 'Found %d parts' % len(parts)
        for pai,part in enumerate(parts):
            if len(part)>0:
                words = parser.parse(part)
                if len(words) >= min_words_in_part:
                    locdic.add_words('%s_p%d'%(fname,pai), None, words)
                    total_parts += 1
    print 'Read total of %d parts from %d files' % (total_parts, len(filenames))
    return locdic