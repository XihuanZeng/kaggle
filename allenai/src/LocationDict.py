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
#             if self._tf_idf_cache is None:
#                 self._tf_idf_cache_log_flag = tf_log_flag
#                 self._tf_idf_cache = {}
#             else:
#                 assert self._tf_idf_cache_log_flag == tf_log_flag
#             cache_key = (doc_id, word_id)
#             if self._tf_idf_cache.has_key(cache_key):
#                 return self._tf_idf_cache[cache_key]
            doc_len = self.doc_lengths[doc_id]
            word_count_in_doc = len(self.word_locations[word_id][doc_id]) if self.save_locations else self.word_locations[word_id][doc_id]
            num_doc_unique_words = self.doc_unique_words[doc_id]
            word_num_docs = self.get_word_num_docs(word_id)
        assert word_count_in_doc>=1 and doc_len>=1
        if tf_log_flag:
            tf = np.log(1.0 + word_count_in_doc + 0.5/num_doc_unique_words) / np.log(1.0 + doc_len + 0.5)
        else:
            tf = (word_count_in_doc + 0.5/num_doc_unique_words) / (doc_len + 0.5)
#         print '  word %s (%s) , doc %s (%s)' % (word_id, self.get_word(word_id), doc_id, self.doc_names[doc_id])
#         print '    TF  = (%d + 0.5/%d) / (%d + 0.5) = %.5f' % (word_count_in_doc, num_doc_unique_words, doc_len, tf)
        idf = np.log((self.get_num_docs() + 0.5) / (word_num_docs + 0.5))
#         print '    IDF = log((%d +0.5) / (%d + 0.5)) = %.5f' % (self.get_num_docs(), word_num_docs, idf)
#         print '  -> TF*IDF = %.6f' % (tf*idf)
#         if not doc_name_flag:
#             self._tf_idf_cache[cache_key] = tf*idf
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
#             self._cache['word_probs'] = []
#             n_docs = self.get_num_docs() + 0.0
#             for wid in range(len(self.word_ids)):
#                 if self._cache['is_word_filtered'][wid]:
#                     self._cache['word_probs'].append(0) # we don't want to select a filtered word
#                 else:
#                     self._cache['word_probs'].append(self.word_counts[wid])
#             sum_probs = np.sum(self._cache['word_probs'])
#             self._cache['word_probs'] = np.array(np.cumsum(self._cache['word_probs']), dtype=np.float32) / (sum_probs + 0.0)
#             print '   Total for probs = %d (%d filtered words: %s)' % (sum_probs, np.sum(self._cache['is_word_filtered']),
#                                                                        [word for word,wid in self.word_ids.iteritems() if self._cache['is_word_filtered'][wid]])
#             print '   Total count = %d' % np.sum(self.word_counts)
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
#             print ' random word: %4d - %s' % (rwords[-1], [word for word,wid in self.word_ids.iteritems() if wid==rwords[-1]])
        return rwords

    def search_docs(self, words, word_ids=None, words_weights=None, min_words_per_doc=None, max_words_distance=None,
                    prob_type='tf-idf', tf_log_flag=True):
        assert max_words_distance is None, 'max_words_distance not supported yet'
        assert prob_type in ['word-probs', 'tf-idf']
#         start_time = time.time()
#         print '-> Searching for words: %s' % str(words)
        if words_weights is None:
            words_weights = {} # default weight is 1
        word_ids_weights = dict([(self.word_ids[word], wgt) for word,wgt in words_weights.iteritems() if self.word_ids.has_key(word)])
#         print 'word_ids_weights: %s' % word_ids_weights

        if words is not None:
            assert word_ids is None
            words = np.unique(words)
            # Get the word ids to search for
            word_ids = [self.word_ids[word] for word in words if self.word_ids.has_key(word)]
#             print 'IDs: %s' % word_ids
        else:
            word_ids = np.unique(word_ids)
        word_ids = self._filter_words_for_search(word_ids)
        # Ignore words with weight=0
        word_ids = [wid for wid in word_ids if word_ids_weights.get(wid,1)!=0]
#         print 'words: %s' % words
#         print 'word_ids_weights: %s' % word_ids_weights
        # Normalize weights
        wmean = np.mean([word_ids_weights.get(wid,1.0) for wid in word_ids])
        for wid in word_ids:
            word_ids_weights[wid] = word_ids_weights.get(wid, 1.0) / wmean
#         print '-> word_ids weights: %s' % word_ids_weights

#         print '    init time = %.3f secs.' % (time.time()-start_time)

#         print 'Searching for filtered words: %s' % ', '.join(['%s (%s)'%(wid,self.get_word(wid)) for wid in word_ids])
        if len(word_ids)==0:
            return dict([((0 if mwpd is None else mwpd),([],[])) for mwpd in min_words_per_doc])
#         if min_words_per_doc is None:
#             min_words_per_doc = len(word_ids)
#         assert min_words_per_doc > 0
#         if min_words_per_doc > len(word_ids):
#             return dict([(mwpd,([],[])) for mwpd in min_words_per_doc])

        # Find all relevant docs
#         start_time = time.time()
        words_per_doc, words_per_doc_name = {}, {}
        for wid in word_ids:
            for did in self.word_locations[wid].keys():
#                 words_per_doc[did] = words_per_doc.get(did,set()).union([wid])
                if not words_per_doc.has_key(did):
                    words_per_doc[did] = []
                words_per_doc[did].append(wid)
                if self.doc_name_weight != 0:
#                     words_per_doc_name[did] = words_per_doc_name.get(did,set()).union([wid])
                    if not words_per_doc_name.has_key(did):
                        words_per_doc_name[did] = []
                    words_per_doc_name[did].append(wid)
#         print 'words_per_doc: %s' % words_per_doc
#         print 'words_per_doc_name: %s' % words_per_doc_name
#         print '    find time = %.3f secs.' % (time.time()-start_time)

#         start_time = time.time()

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
#         print 'min_words_per_doc  = %s' % min_words_per_doc
#         print 'min_words_per_doc2 = %s' % min_words_per_doc2

        docs_probs = {}
        prev_doc_probs = {}
        min_mwpd = min_words_per_doc2[-1]
        for mwpd in min_words_per_doc2:
#             print '-> mwpd = %d' % mwpd
            if mwpd == 0:
                docs_probs[mwpd] = ([], []) # dummy empty sets for searching with 0 min words
                continue
            assert (mwpd > 0) and (mwpd <= len(word_ids))
            docs = [doc for doc,ws in words_per_doc.iteritems() if len(ws)>=mwpd]
#             assert set(docs).issuperset(prev_doc_probs.keys())
#             print 'mwpd=%d, docs: %s' % (mwpd, docs)
            # Compute probability/score of each document
            n_docs = self.get_num_docs() + 0.0
            probs = []
            for doc in docs:
                if prev_doc_probs.has_key(doc):
                    prob = prev_doc_probs[doc]
#                     print ' using prob %.3f for doc %s' % (prob, doc)
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
    #                     print '  doc %s (%s): prob = %7.4f' % (doc, self.doc_names[doc], prob)
#                     print ' computed prob %.3f for doc %s' % (prob, doc)
                    if mwpd > min_mwpd: # no pointing in caching values we won't be using...
                        prev_doc_probs[doc] = prob
                assert prob>=0
                probs.append(prob)
            docs_probs[mwpd] = (docs, probs)

#         print '    probs time = %.3f secs.' % (time.time()-start_time)
        return docs_probs

    def double_search(self, words1, words2, num_words1=[None], num_words2=[None], words1_weights={}, words2_weights={},
                      score='hg', score_params=None, prob_type='tf-idf', tf_log_flag=True):
        assert score in ['counts','hg','weights']
#         print '--> double_search: [%s] , [%s]' % (words1, words2)
#         docs1 = [(min_words_per_doc1,self.search_docs(words1, min_words_per_doc=[min_words_per_doc1])[0][0]) for min_words_per_doc1 in num_words1]
#         docs2 = [(min_words_per_doc2,self.search_docs(words2, min_words_per_doc=[min_words_per_doc2])[0][0]) for min_words_per_doc2 in num_words2]

#         words1_weights = {}
#         for iw,word in enumerate(words1):
#             words1_weights[word] = words1_weights.get(word,0.0) + ((10.0+iw)**4)/(10.0**4)    # (10.0+iw)**4/(10.0**4): 0.4092
# #         wsum = np.sum(words1_weights.values())
# #         for word in words1_weights.keys():
# #              words1_weights[word] = words1_weights[word] / wsum
# #         print 'words1_weights: %s' % words1_weights

#         start_time = time.time()

        if (self._search_cache is not None) and (self._search_cache['words'] == words1) and \
           (self._search_cache['num_words'] == num_words1) and (self._search_cache['params'] == (score,prob_type,tf_log_flag)):
            docs1 = self._search_cache['docs']
        else:
            docs1 = self.search_docs(words1, words_weights=words1_weights,
                                     min_words_per_doc=num_words1, prob_type=prob_type, tf_log_flag=tf_log_flag)
            self._search_cache = {'words': words1, 'num_words': num_words1, 'params': (score,prob_type,tf_log_flag), 'docs': docs1}
        docs2 = self.search_docs(words2, words_weights=words2_weights,
                                 min_words_per_doc=num_words2, prob_type=prob_type, tf_log_flag=tf_log_flag)

#         end_time = time.time()
#         print 'search_docs time = %.3f secs.' % (end_time-start_time)

#         print 'docs1: %s' % docs1
#         print 'docs2: %s' % docs2
#         num_random = 1
#         docs_rand = []
#         for ri in range(num_random):
#             rwords = self._get_random_words(len(words2))
#             docs_rand.append([(min_words_per_doc2,self.search_docs(words=None, word_ids=rwords, min_words_per_doc=min_words_per_doc2)[0]) for min_words_per_doc2 in num_words2])
# #             print 'docs_rand: %s' % str(docs_rand[-1])
#         wi_time = 0
#         start_time = time.time()

        best_score_over, best_score_under = 0, 0
        for mw1,(d1,p1) in docs1.iteritems():
            if len(d1)==0 and mw1>0: continue
#             print 'mw1=%s , d1: %s' % (str(mw1), str(d1))
            for mw2,(d2,p2) in docs2.iteritems():
#                 if words2==['water']:
#                 print 'mw2=%s , d2: %s' % (str(mw2), str(d2))
                if len(d2)==0: continue
                assert mw2 > 0
                if mw1 == 0:
                    d12 = d2
                else:
                    d12 = set(d1).intersection(d2)
#                 if words2==['water']:
#                     print 'min words %s,%s -> d1 %d , d2 %d -> d12 %d' % (mw1, mw2, len(d1), len(d2), len(d12))
                # Compute intersections for random words
#                 rand_intersections = []
#                 for ri,docsr in enumerate(docs_rand):
#                     for mwr,dr in docsr:
# #                         print 'docsr: %s , %s' % (str(mwr), dr)
#                         if mwr==mw2:
#                             d1r = set(d1).intersection(dr)
# #                             print '  rand %d: min words %s,%s -> d1 %d , dr %d -> d1r %d' % (ri, mw1, mw2, len(d1), len(dr), len(d1r))
#                             rand_intersections.append(len(d1r))
#                             break
                if score == 'hg':
                    assert len(d1) > 0
                    # Compute p-value for over/under-representation of intersection
                    score_over  = -np.log10(np.max([1E-320, hg_test(M=int(self.get_num_docs()), n=len(d1), N=len(d2), k=len(d12), dir= 1)['prob']]))
                    score_under = -np.log10(np.max([1E-320, hg_test(M=int(self.get_num_docs()), n=len(d1), N=len(d2), k=len(d12), dir=-1)['prob']]))
    #                 rand_rank_over  = 1.0 - scipy.stats.percentileofscore(rand_intersections, len(d12), kind='strict')/100.0
    #                 rand_rank_under = scipy.stats.percentileofscore(rand_intersections, len(d12), kind='weak')/100.0
    #                 print ' -> pvals: %.2f , %.2f ; rank = %.2f , %.2f' % (pval_over, pval_under, rand_rank_over, rand_rank_under)
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
#                     t1 = time.time()
                    weights_intersection = sorted([w1[did]*w2[did] for did in d12], reverse=True)
#                     wi_time += (time.time() - t1)

                    if len(weights_intersection)==0:
                        score_over = 0.0
                    else:
                        if (score_params is not None) and score_params.has_key('coeffs'):
                            coeffs = score_params['coeffs'](len(weights_intersection))
                        else:
                            coeffs = np.ones(len(weights_intersection))
                        ##score_over  = np.average(weights_intersection, weights=coeffs) #np.mean(weights_intersection[-10:])
#                         print 'weights intersection: %s' % ', '.join(['%7.4f'%x for x in weights_intersection])
#                         print 'coeffs              : %s' % ', '.join(['%7.4f'%x for x in coeffs])
                        score_over  = np.dot(weights_intersection, coeffs)
#                         print '-> score_over = %.4f' % score_over

                    d2not1 = set(d2).difference(d1)
                    weights_c_intersection = sorted([w2[did] for did in d2not1], reverse=True)
                    if len(weights_c_intersection)==0:
                        score_under = 0.0
                    else:
                        if (score_params is not None) and score_params.has_key('coeffs'):
                            coeffs = score_params['coeffs'](len(weights_c_intersection))
                        else:
                            coeffs = np.ones(len(weights_c_intersection))
##                        score_under = np.average(weights_c_intersection, weights=coeffs) #np.mean(weights_intersection[:10])
                        score_under = np.dot(weights_c_intersection, coeffs)
#                     print '   score over = %7.2f , under = %7.2f' % (score_over, score_under)

                    if (score_params is not None) and score_params.has_key('calc_over_vs_under') and (score_params['calc_over_vs_under']==True):
                        w1p, w2p = np.sum(w1.values()) / self.get_num_docs(), np.sum(w2.values()) / self.get_num_docs()
                        oexp = w1p * w2p * self.get_num_docs()
                        oscore = (np.sum(weights_intersection) - oexp) / np.sum([np.sqrt(oexp*(1.0-w1p)), np.sqrt(oexp*(1.0-w2p))])
                        #score_under = np.clip(oscore/100.0 + 0.5, 0.0, 1.0) # change from std's to scale 0...1
                        score_under = scipy.stats.norm.cdf(oscore)
#                         print 'w1p=%.3f, w2p=%.3f -> oexp=%.3f ; wi=%.3f -> score under = %.3f' % (w1p, w2p, oexp, np.sum(weights_intersection), score_under)
#                         avg_d1 = np.mean(w1.values())
#                         print 'mw1=%d, mw2=%d: over = %.3f , under = %.3f (avg d1 = %.3f) -> over/under = %.3f' % (mw1, mw2, score_over, score_under, avg_d1, score_over / (score_over + avg_d1 * score_under))
#                         score_under = score_over / np.sqrt(score_over + avg_d1 * score_under) # w/o sqrt: 0.4152

                assert score_over>=0 and score_under>=0
#                 print ' mw1=%2d , mw2=%2d -> score over = %.3f' % (mw1, mw2, score_over)
                if (score_params is not None) and score_params.has_key('minword1_coeffs'):
                    mwc1 = score_params['minword1_coeffs'](mw1, np.max(docs1.keys()))
                    mwc2 = score_params['minword2_coeffs'](mw2, np.max(docs2.keys()))
#                     print '   adding %.3f * %.3f = %.3f' % (score_over, mwc1*mwc2, score_over*mwc1*mwc2)
                    best_score_over  += score_over  * mwc1 * mwc2
                    best_score_under += score_under * mwc1 * mwc2
                else:
                    best_score_over  = np.max([best_score_over , score_over])
                    best_score_under = np.max([best_score_under, score_under])
#         print '---> best scores: %.3f , %.3f' % (best_score_over, best_score_under)

#         end_time = time.time()
#         print 'compute score time = %.3f secs. (wi time = %.3f)' % (end_time-start_time, wi_time)

        return best_score_over, best_score_under




def build_training_location_dictionary(train, parser=SimpleWordParser(),
                                       use_questions=True, use_answers=True, min_words_per_qa=1, base_locdic=None,
                                       min_word_docs_frac=0, max_word_docs_frac=0.2, min_word_count_frac=0, max_word_count_frac=0.01,
                                       ascii_conversion=True):
#     print '=> Building LocationDictionary for %d training samples' % len(train)
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
#         print '%s' % text
#         print 'Found %d parts: [0] %s ; [1] %s ; [2] %s' % (len(parts), parts[0][:50], parts[1][:50], parts[2][:50])
        print 'Found %d parts' % len(parts)
        for pai,part in enumerate(parts):
            if len(part)>0:
#                 print '-------------------------------------------------------------- 1'
#                 print 'part: <<<%s>>>' % part
                words = parser.parse(part)
#                 print '-------------------------------------------------------------- 3'
#                 print 'words: %s' % ' ; '.join(words)
#                 for word in words:
#                     for c in word:
#                         if (ord(c)<ord('a') or ord(c)>ord('z')) and (ord(c)<ord('0') or ord(c)>ord('9')) and ord(c)!=ord('.'):
#                             print 'word=%s [%s] ord %d' % (word, c, ord(c))
#                             jshdjshd()
                if len(words) >= min_words_in_part:
#                     print '+++++ Adding part with %d words:\n%s' % (len(words), part)
#                     print '===== Words:\n%s' % ' ; '.join(words)
                    locdic.add_words('%s_p%d'%(fname,pai), None, words)
                    total_parts += 1
#                 else:
#                     print '----- Skipping part with %d words:\n%s' % (len(words), part)
#             if pai>3:
#                 skdjskdj()
    print 'Read total of %d parts from %d files' % (total_parts, len(filenames))
    return locdic