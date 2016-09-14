import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
import scipy
import scipy.optimize
import time
import re
import gc

from Parser import SimpleWordParser
from LocationDict import LocationDictionary


class AnswersFunc(object):
    def __call__(self, question, answers):
        pass

class AnswersLengthFunc(AnswersFunc):
    def __init__(self, log_flag=True):
        self.log_flag = log_flag
    def __call__(self, question, answers):
        lens = np.array(map(len, answers), dtype=np.float32)
        assert np.sum(lens)>0
        if self.log_flag:
            return np.abs(np.log((1.0+lens) / (1.0+np.mean(lens))))
        else:
            print '%s' % ((1.0+lens) / (1.0+np.mean(lens)))
            return (1.0+lens) / (1.0+np.mean(lens))

class AnswersNumWordsFunc(AnswersFunc):
    def __init__(self, parser=SimpleWordParser()):
        self.parser = parser
    def __call__(self, question, answers):
        num_words = []
        for ans in answers:
            num_words.append(len(self.parser.parse(ans)))
        return np.array(num_words)

class AnswersIsBCFunc(AnswersFunc):
    def __init__(self):
        pass
    def __call__(self, question, answers):
        assert len(answers)==4
        return np.array([0,1,1,0])

class AnswersBCDAFunc(AnswersFunc):
    # Counts in training set: A: 584, B: 672, C: 640, D: 604
    def __init__(self):
        pass
    def __call__(self, question, answers):
        assert len(answers)==4
        return np.array([0,3,2,1])

class AnswersIsNumericalFunc(AnswersFunc):
    NUMBER_STR = {'zero':0, 'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'twenty':20}
    def __init__(self):
        pass
    def __call__(self, question, answers):
        nums = [self._is_numerical(ans) for ans in answers]
        if np.all([(num is not None) for num in nums]):
            if len(set(nums))==len(nums):
                order = np.argsort(nums)
                return np.array([(2 if (x>0 and x<len(answers)-1) else 0) for x in order])
            else: # not unique numbers
                return np.ones(len(answers))
        else:
            return np.ones(len(answers))

    def _is_numerical(self, answer):
        if len(answer)==0:
            return None
        # Find first word that is a number
        words = answer.split(' ')
        for word in words:
            word = word.replace(',','') # turn "2,000" to "2000"
            if re.match('^[0-9]+([\.][0-9]+)?$', word) is not None:
                return float(word)
            else:
                if AnswersIsNumericalFunc.NUMBER_STR.has_key(word):
                    return float(AnswersIsNumericalFunc.NUMBER_STR[word])
        return None # did not find a word that looks like a number

class AnswersInQuestionFunc(AnswersFunc):
    def __init__(self, parser=SimpleWordParser()):
        self.parser = parser
    def __call__(self, question, answers):
        q_words = set(self.parser.parse(question))
        in_fracs = []
        for ans in answers:
            a_words = set(self.parser.parse(ans))
            if len(a_words) == 0:
                in_fracs.append(0.0)
            else:
                in_fracs.append(len(q_words.intersection(a_words)) / (len(a_words) + 0.0))
        return np.array(in_fracs)

class AnswersInAnswersFunc(AnswersFunc):
    def __init__(self, parser=SimpleWordParser()):
        self.parser = parser
    def __call__(self, question, answers):
        a_words = [set(self.parser.parse(ans)) for ans in answers]
        in_fracs = []
        for i,ans in enumerate(answers):
            w1 = a_words[i]
            if len(w1) == 0:
                in_fracs.append(0.0)
            else:
                frac = 0.0
                for j in range(len(answers)):
                    if j==i: continue
                    frac += (len(w1.intersection(a_words[j])) / (len(w1) + 0.0))
                in_fracs.append(frac / (len(answers) - 1.0))
        in_fracs = np.array(in_fracs)
        if np.all(in_fracs == 0):
            return np.ones(len(answers)) * 0.25
        else:
            return in_fracs / np.sum(in_fracs)

class AnswerCountFunc(AnswersFunc):
    def __init__(self, train, check_same_question=True, count_type='count', parser=None, single_words=False, use_questions=False, norm_scores=True):
        self.train = train
        self.check_same_question = check_same_question
        self.count_type = count_type
        self.count_type in ['count','correct','pval','zscore','ans_vs_qst']
        self.parser = parser
        self.single_words = single_words
        if single_words:
            assert self.parser is not None
        self.use_questions = use_questions
        self.norm_scores = norm_scores
        #self.ans_trans_f = self._get_answer_trans_func()
        self.ans_count, self.ans_correct = None, None
        self.qst_count, self.qst_correct = None, None
        self._parsed_words = None

    def _get_answer_trans_func(self):
        if self.parser is None:
            return lambda ans: [ans]
        else:
            if self.single_words:
                return lambda ans: self.parser.parse(ans)
            else:
                return lambda ans: [' '.join(self.parser.parse(ans))]

    def _parse_train(self, train):
        self._parsed_words = {}
        for txt in np.concatenate([np.unique(train['answer']), np.unique(train['question'])]):
            self._parsed_words[txt] = self._get_answer_trans_func()(txt)

    def _calc_stats(self, train):
        self.ans_count, self.ans_correct = {}, {}
        self.qst_count, self.qst_correct = {}, {}
        for ans,corr in zip(train['answer'],train['correct']):
            words = self._parsed_words[ans] #self.ans_trans_f(ans)
            for word in words:
                if not self.ans_count.has_key(word):
                    self.ans_count[word] = 1
                    self.ans_correct[word] = corr
                else:
                    self.ans_count[word] += 1
                    self.ans_correct[word] += corr
        if self.use_questions or (self.count_type == 'ans_vs_qst'):
            for qst in train['question']:
                words = self._parsed_words[qst] #self.ans_trans_f(qst)
                for word in words:
                    if not self.qst_count.has_key(word):
                        self.qst_count[word] = 1
                        self.qst_correct[word] = 0.25
                    else:
                        self.qst_count[word] += 1
                        self.qst_correct[word] += 0.25

    def __call__(self, question, answers):
        if self._parsed_words is None:
            self._parse_train(self.train) # parse only upon the 1st call

        if self.check_same_question:
            train = self.train[self.train['question']!=question]
        else:
            train = self.train
        if self.check_same_question or (self.ans_count is None):
            self._calc_stats(train)

        assert len(self.ans_count) > 0 and len(self.ans_correct) > 0
        if self.use_questions:
            assert len(self.qst_count) > 0 and len(self.qst_correct) > 0

        t_answers = [self._get_answer_trans_func()(ans) for ans in answers]
        print 'answers: %s' % answers
        print ' t_answers: %s' % t_answers
        if self.count_type == 'count':
            counts = []
            for t_ans in t_answers:
                if len(t_ans) == 0:
                    counts.append(0)
                else:
                    if self.use_questions:
                        counts.append(np.mean([self.ans_count.get(ta,0) for ta in t_ans] + [self.qst_count.get(ta,0) for ta in t_ans]))
                    else:
                        counts.append(np.mean([self.ans_count.get(ta,0) for ta in t_ans]))
        elif self.count_type == 'correct':
            counts = []
            for t_ans in t_answers:
                corrs = []
                for ta in t_ans:
                    corr, count = 0.25, 1.0
                    if self.ans_count.has_key(ta):
                        count += self.ans_count[ta]
                        corr  += self.ans_correct[ta]
                    if self.use_questions and self.qst_count.has_key(ta):
                        count += self.qst_count[ta]
                        corr  += self.qst_correct[ta]
                    corrs.append(-np.log(corr / count))
                if len(corrs) == 0:
                    counts.append(0.25)
                else:
                    counts.append(np.exp(-np.mean(corrs)))
            assert not np.any([np.isnan(x) for x in counts])
        elif self.count_type == 'ans_vs_qst':
            # Count the number of times the answer (or its words) appears as a correct answer vs. the number of times it appears in questions
            assert not self.use_questions, 'use_questions not supported for pval or zscore'
            counts = []
            for t_ans in t_answers:
                ratios = []
                for ta in t_ans:
                    ans_corr, qst_count = 0.25, 1.0 # Laplace
                    if self.ans_count.has_key(ta):
                        ans_corr  += self.ans_correct[ta]
                    if self.qst_count.has_key(ta):
                        qst_count += self.qst_count[ta]
                    ratios.append(-np.log(ans_corr / qst_count))
                    print '  ans count , corr = %d , %d' % (self.ans_count.get(ta,0), self.ans_correct.get(ta,0))
                    print '  qst count , corr = %d , %d' % (self.qst_count.get(ta,0), self.qst_correct.get(ta,0))
                    print ' -> ratio = %.3f' % ratios[-1]
                if len(ratios) == 0:
                    counts.append(0.0)
                else:
                    counts.append(np.exp(-np.mean(ratios)))
            assert not np.any([np.isnan(x) for x in counts])
        else: # pval or zscore
            assert not self.use_questions, 'use_questions not supported for pval or zscore'
            counts = []
            for t_ans in t_answers:
                t_probs = []
                for ta in t_ans:
                    if self.ans_count.has_key(ta):
                        mult_test_corr = True if (self.count_type == 'pval') else False
                        hg = hg_test(len(train), np.sum(train['correct']), self.ans_count[ta], self.ans_correct[ta], dir=None, mult_test_correct=mult_test_corr)
                        prob, dir = hg['prob'], hg['dir']
                        if self.count_type == 'pval':
                            if dir == -1:
                                pr = 0.25*(np.exp(prob)-np.exp(0))/(np.exp(1)-np.exp(0))
                            else:
                                pr = 1.0 + 0.75*(np.exp(prob)-np.exp(0))/(np.exp(0)-np.exp(1))
                        else: # 'zscore'
                            ##assert prob <= 0.55, 'Prob is too large? (%.5f)' % prob
                            prob = np.clip(prob, 1E-20, 0.5)
                            pr = (-dir) * scipy.stats.norm.isf(prob) # Z-score
                    else:
                        if self.count_type == 'pval':
                            pr = 0.25
                        else: # 'zscore'
                            pr = 0.0
                    if self.count_type == 'pval':
                        t_probs.append(-np.log(pr))
                    else: # 'zscore'
                        t_probs.append(pr)
                if len(t_probs) == 0:
                    counts.append(0.25)
                else:
                    if self.count_type == 'pval':
                        counts.append(np.exp(-np.mean(t_probs)))
                    else: # 'zscore'
                        counts.append(scipy.stats.norm.sf(np.sum(t_probs)/np.sqrt(len(t_probs))) ** 2) # take square so that Z=0 will become prob=0.25 (and not 0.5)
        counts = np.array(counts)
        if self.norm_scores:
            if np.all(counts == 0):
                return np.ones(len(answers)) * 0.25
            return counts / (np.sum(counts) + 0.0)
        else:
            return counts


class AnswerPairCountFunc(AnswersFunc):
    def __init__(self, train, check_same_question=True):
        self.train = train
        self.check_same_question = check_same_question
    def __call__(self, question, answers):
        if self.check_same_question:
            train = self.train[self.train['question']!=question]
        else:
            train = self.train
        ans_count = {}
        for idx,inds in train.groupby('ID').groups.iteritems():
            print 'idx %s' % idx
            assert len(inds)==4
            for i in range(3):
                row_i = train.irow(inds[i])
                for j in range(i+1,4):
                    row_j = train.irow(inds[j])
                    pair = tuple(sorted([row_i['answer'], row_j['answer']]))
                    print ' pair: %s' % str(pair)


class AnswersSameQuestionFunc(AnswersFunc):
    def __init__(self, train, use_stemmer=False):
        self.train = train
        if use_stemmer:
            self.stemmer = PorterStemmer()
            self.parser = SimpleWordParser(word_func=self.stemmer.stem)
        else:
            self.parser = SimpleWordParser(word_func=None)
    def __call__(self, question, answers):
        question = question.lower()
        same_q = self.train[self.train['question']==question]
        if len(same_q) == 0:
            return np.array([0.25,0.25,0.25,0.25])

        print '==> Found same question: %s' % question
#         print '%s' % same_q
        print '-------------------------------------------------'
        scores = []
        for ia,answer in enumerate(answers):
            answer = answer.lower()
            same_qa = same_q[same_q['answer']==answer]
            if len(same_qa) > 0:
                assert len(same_qa)==1
#                 print 'same_qa: %s' % same_qa.irow(0)['correct']
                scores.append(same_qa.irow(0)['correct']) # 0 or 1
            else:
                print '-> Answer (%s) not found...' % (answer)
                scores.append(0.01)
#             a_words = self.parser.parse(answer)
#             print '  -> %s' % ' ; '.join(a_words)
#             probs = locdic.double_search(words1=q_words, words2=a_words, num_words1=[None,1,2,3,4,5,6,7], num_words2=[None,1,2,3,4,5]) # (3,5)(2) -> 0.3012,0.2444; (2,3,4,5)(1,2) -> 0.3216,0.2756; (1-7),(1-5) -> 0.3232,0.2816
#             print '  answer #%d: %.2e , %.2e' % (ia, probs[0], probs[1])
        print 'scores: %s' % scores
#         ref_score0, ref_score1 = np.max([s[0] for s in scores]), np.max([s[1] for s in scores])
#         scores = [(s[0]/ref_score0, s[1]/ref_score1) for s in scores]
#         print '-> scores: %s ; %s' % (', '.join(['%.2e'%s[0] for s in scores]), ', '.join(['%.2e'%s[1] for s in scores]))
        return np.array(scores)


class AnswersWordZscoreFunc(AnswersFunc):
    def __init__(self, locdic=None, parser=SimpleWordParser(), score='zscore', norm_scores=True):
        self.locdic = locdic
        self.parser = parser
        self.score = score
        assert self.score == 'zscore'
        self.norm_scores = norm_scores

    def __call__(self, question, answers):
        if not isinstance(self.locdic, LocationDictionary):
            self.locdic = self.locdic() # a generator
            assert isinstance(self.locdic, LocationDictionary)
        num_total_docs = self.locdic.get_num_docs() + 0.0
        STD_FACTOR = 300.0
        EPSILON = 1E-320
        print 'question = %s' % question
        q_words = np.unique(self.parser.parse(question, calc_weights=False))
        print '  -> words: %s' % ' ; '.join(q_words)
        q_word_ids = [self.locdic.word_ids[word] for word in q_words if self.locdic.word_ids.has_key(word)]
        print '  -> ids: %s' % str(q_word_ids)
        q_docs = []
        for wid in q_word_ids:
            q_docs.append(set(self.locdic.word_locations[wid].keys()))
        print '  -> # docs: %s' % ' ; '.join(['%d'%len(d) for d in q_docs])

        scores = []
        for ia,answer in enumerate(answers):
            a_words = np.unique(self.parser.parse(answer, calc_weights=False))
            print '  -> words: %s' % ' ; '.join(a_words)
            a_word_ids = [self.locdic.word_ids[word] for word in a_words if self.locdic.word_ids.has_key(word)]
            print '  -> ids: %s' % str(a_word_ids)
            zscores = []
            for a_word in a_word_ids:
                a_docs = self.locdic.word_locations[a_word].keys()
                a_pr = len(a_docs) / num_total_docs
                a_pr = np.clip(a_pr, 1E-5, 1.0-(1E-5))
#                 print '  a_word (%.5f) = %s' % (a_pr, a_word)
                for q_word,qd in zip(q_word_ids,q_docs):
                    if len(qd)==0: continue # no documents containing the question word
                    if q_word==a_word: continue # same word - skip
                    num_intersect = len(qd.intersection(a_docs))
#                     print '   qdocs %5d , adocs %5d -> intersection %4d' % (len(qd), len(a_docs), num_intersect)
                    n_exp, n_std = len(qd) * a_pr, np.sqrt(len(qd) * a_pr * (1.0-a_pr))
                    z = (num_intersect - n_exp) / n_std
                    zscores.append(-z)
#                     print '     -> z = %.3f' % z

            print '  answer #%d: zscores = %s => sum = %.2f , num = %d' % (ia, zscores, np.sum(zscores), len(zscores))
            if len(zscores)==0:
                scores.append(0.0)
            else:
                scores.append(scipy.stats.norm.sf(np.sum(zscores)/(STD_FACTOR*np.sqrt(len(zscores)))) ** 2) # take square so that Z=0 will become prob=0.25 (and not 0.5)
#             scores.append(-np.sum(zscores)/np.sqrt(len(zscores))) # take square so that Z=0 will become prob=0.25 (and not 0.5)
            print '---> score = %.2e' % scores[-1]
        if self.norm_scores:
            ref_score = np.sum(scores) + len(scores)*EPSILON
            scores = [(s+EPSILON)/ref_score for s in scores]
        print ' -> scores: %s' % (', '.join(['%.2e'%s for s in scores]))
        assert not np.any([np.isnan(s) for s in scores])
#         dkfjdkfj()
        return np.array(scores)


class AnswersDoubleSearchFunc(AnswersFunc):
    def __init__(self, locdic=None, parser=SimpleWordParser(), num_words_qst=[None,1,2,3,4,5], num_words_ans=[None,1,2,3],
                 score='hg', score_params=None, prob_type='tf-idf', tf_log_flag=True, norm_scores=True):
        self.locdic = locdic
#         if use_stemmer:
#             self.stemmer = PorterStemmer()
#             self.parser = SimpleWordParser(word_func=self.stemmer.stem)
#         else:
#             self.parser = SimpleWordParser(word_func=None)
        self.parser = parser
        self.num_words_qst = num_words_qst
        self.num_words_ans = num_words_ans
        self.score = score
        self.score_params = score_params
        self.prob_type= prob_type
        self.tf_log_flag = tf_log_flag
        self.norm_scores = norm_scores
        self.total_time = 0

    def __call__(self, question, answers):
        if not isinstance(self.locdic, LocationDictionary):
            self.locdic = self.locdic() # a generator
            assert isinstance(self.locdic, LocationDictionary)
        EPSILON = 1E-320
        print 'question = %s' % question
        q_words, q_weights = self.parser.parse(question, calc_weights=True)
        if (q_weights is None) or (len(q_weights) == 0):
            print '  -> %s' % ' ; '.join(q_words)
        else:
            print '  -> %s' % ' ; '.join(['%s (%.1f)'%(w,q_weights.get(w,-1)) for w in q_words])
#         print '-> %s' % ' ; '.join(q_words)
#         if len(q_weights) > 0:
#             print '     weights: %s' % q_weights
        scores = []
        for ia,answer in enumerate(answers):
            a_words, a_weights = self.parser.parse(answer, calc_weights=True)
            if (a_weights is None) or (len(a_weights) == 0):
                print '  -> %s' % ' ; '.join(a_words)
            else:
                print '  -> %s' % ' ; '.join(['%s (%.1f)'%(w,a_weights.get(w,-1)) for w in a_words])
#             if len(a_weights) > 0:
#                 print '     weights: %s' % a_weights
            t1 = time.time()
            probs = self.locdic.double_search(words1=q_words, words2=a_words, words1_weights=q_weights, words2_weights=a_weights,
                                              num_words1=self.num_words_qst, num_words2=self.num_words_ans,
                                              score=self.score, score_params=self.score_params, prob_type=self.prob_type, tf_log_flag=self.tf_log_flag)
            self.total_time += (time.time() - t1)
            print '  answer #%d: %.2e , %.2e' % (ia, probs[0], probs[1])
            scores.append(probs)
#         ref_score0, ref_score1 = np.max([s[0] for s in scores]), np.max([s[1] for s in scores])
        if self.norm_scores:
            ref_score0, ref_score1 = np.sum([s[0] for s in scores]) + len(scores)*EPSILON, np.sum([s[1] for s in scores]) + len(scores)*EPSILON
            scores = [((s[0]+EPSILON)/ref_score0, (s[1]+EPSILON)/ref_score1) for s in scores]
        print ' -> scores: %s ; %s' % (', '.join(['%.2e'%s[0] for s in scores]), ', '.join(['%.2e'%s[1] for s in scores]))
        print '    total time so far: %.2f' % self.total_time
        return np.array(scores)

class AnswersTrainDoubleSearchFunc(AnswersDoubleSearchFunc):
    def __init__(self, train, base_locdic=None, check_same_question=True, use_questions=True, use_answers=True, min_words_per_qa=1,
                 parser=SimpleWordParser(),
                 num_words_qst=[None,1,2,3,4,5], num_words_ans=[None,1,2,3],
                 score='hg', score_params=None, prob_type='tf-idf', tf_log_flag=True, norm_scores=True,
                 min_word_docs_frac=0, max_word_docs_frac=0.2, min_word_count_frac=0, max_word_count_frac=0.01):
        AnswersDoubleSearchFunc.__init__(self, locdic=None, parser=parser, num_words_qst=num_words_qst, num_words_ans=num_words_ans,
                                         score=score, score_params=score_params, prob_type=prob_type, tf_log_flag=tf_log_flag, norm_scores=norm_scores)
        self.train = train
        self.base_locdic = base_locdic
        self.use_questions = use_questions
        self.use_answers   = use_answers
        self.min_words_per_qa = min_words_per_qa
        self.min_word_docs_frac = min_word_docs_frac
        self.max_word_docs_frac = max_word_docs_frac
        self.min_word_count_frac = min_word_count_frac
        self.max_word_count_frac = max_word_count_frac
        self.check_same_question = check_same_question
        self.locdic = None

    def __call__(self, question, answers):
        if self.locdic is None:
            if self.check_same_question:
                ld_train = self.train[self.train['question']!=question]
            else:
                ld_train = self.train
            if (self.base_locdic is not None) and (not isinstance(self.base_locdic, LocationDictionary)):
                self.base_locdic = self.base_locdic() # generator
            self.locdic = build_training_location_dictionary(ld_train, parser=self.parser,
                                                             use_questions=self.use_questions, use_answers=self.use_answers, min_words_per_qa=self.min_words_per_qa,
                                                             base_locdic=self.base_locdic,
                                                             min_word_docs_frac=self.min_word_docs_frac, max_word_docs_frac=self.max_word_docs_frac,
                                                             min_word_count_frac=self.min_word_count_frac, max_word_count_frac=self.max_word_count_frac)
#             print 'base locdic %d docs -> +train locdic %d docs' % (len(self.base_locdic.doc_ids), len(self.locdic.doc_ids))

        scores = AnswersDoubleSearchFunc.__call__(self, question, answers)
        if self.check_same_question:
            self.locdic = None # delete locdic so that we'll create a new one for the next question
            gc.collect()
        return scores


class AnswersPairsDoubleSearchFunc(AnswersFunc):
    '''
    Check pairs of answers
    '''
    def __init__(self, locdic=None, parser=SimpleWordParser(), num_words_qst=[None,1,2,3,4,5], num_words_ans=[None,1,2,3],
                 score='hg', score_params=None, prob_type='tf-idf', tf_log_flag=True, norm_scores=True,
                 sim_scores_comb_weights=([1,0,0], [0,0,1]), search_type='q_vs_a1a2'):
        assert search_type in ['q_vs_a1a2', 'a1_vs_a2']
        self.locdic = locdic
        self.parser = parser
        self.num_words_qst = num_words_qst
        self.num_words_ans = num_words_ans
        self.score = score
        self.score_params = score_params
        self.prob_type= prob_type
        self.tf_log_flag = tf_log_flag
        self.norm_scores = norm_scores
        self.sim_scores_comb_weights = sim_scores_comb_weights
        self.search_type = search_type

    def __call__(self, question, answers):
        if not isinstance(self.locdic, LocationDictionary):
            self.locdic = self.locdic() # a generator
            assert isinstance(self.locdic, LocationDictionary)
        EPSILON = 1E-320
        print 'question = %s' % question
        q_words, q_weights = self.parser.parse(question, calc_weights=True)
        print '-> %s' % ' ; '.join(q_words)
        sim_scores = {}
        for ia1 in range(len(answers)-1):
            answer1 = answers[ia1]
            a_words1, a_weights1 = self.parser.parse(answer1, calc_weights=True)
            print '  -> (%d) %s' % (ia1, ' ; '.join(a_words1))
            for ia2 in range(ia1+1,len(answers)):
                answer2 = answers[ia2]
                a_words2, a_weights2 = self.parser.parse(answer2, calc_weights=True)
                print '  -> (%d) %s' % (ia2, ' ; '.join(a_words2))
                if self.search_type == 'q_vs_a1a2':
                    a_weights12 = a_weights1.copy()
                    for w2,wgt2 in a_weights2.iteritems():
                        a_weights12[w2] = np.max([wgt2, a_weights1.get(w2, -np.inf)])
                    probs = self.locdic.double_search(words1=q_words, words2=a_words1+a_words2, words1_weights=q_weights, words2_weights=a_weights12,
                                                      num_words1=self.num_words_qst, num_words2=self.num_words_ans,
                                                      score=self.score, score_params=self.score_params, prob_type=self.prob_type, tf_log_flag=self.tf_log_flag)
                elif self.search_type == 'a1_vs_a2':
                    probs = self.locdic.double_search(words1=a_words1, words2=a_words2, words1_weights=a_weights1, words2_weights=a_weights2,
                                                      num_words1=self.num_words_ans, num_words2=self.num_words_ans,
                                                      score=self.score, score_params=self.score_params, prob_type=self.prob_type, tf_log_flag=self.tf_log_flag)
                print '  answers #%d, #%d: %.2e , %.2e' % (ia1, ia2, probs[0], probs[1])
                sim_scores[(ia1,ia2)] = probs[0]
        # The score of each answer is its best similarity to another answer
        scores = []
        for ia in range(len(answers)):
            sscrs = sorted([scr for (ia1,ia2),scr in sim_scores.iteritems() if (ia1==ia) or (ia2==ia)], reverse=True)
            scores.append([np.dot(sscrs, self.sim_scores_comb_weights[0]), np.dot(sscrs, self.sim_scores_comb_weights[1])])
#         print 'scores: %s' % scores
        if self.norm_scores:
            ref_score0, ref_score1 = np.sum([s[0] for s in scores]) + len(scores)*EPSILON, np.sum([s[1] for s in scores]) + len(scores)*EPSILON
            scores = [((s[0]+EPSILON)/ref_score0, (s[1]+EPSILON)/ref_score1) for s in scores]
        print ' -> scores: %s ; %s' % (', '.join(['%.2e'%s[0] for s in scores]), ', '.join(['%.2e'%s[1] for s in scores]))
        return np.array(scores)

class AnswersLuceneSearchFunc(AnswersFunc):
    def __init__(self, lucene_corpus, parser, max_docs, weight_func=lambda n: np.ones(n), score_func=None, norm_scores=True):
        self.lucene_corpus = lucene_corpus
        self.parser = parser
        self.max_docs = max_docs
        self.weight_func = weight_func
        if score_func is None:
            self.score_func = lambda s: s
        else:
            self.score_func = score_func
        self.norm_scores = norm_scores

    def __call__(self, question, answers):
        EPSILON = 1E-30
        print 'question = %s' % question
        if self.parser is None:
            q_words = question
        else:
            q_words = self.parser.parse(question, calc_weights=False)
            print '  -> %s' % ' ; '.join(q_words)
        scores = []
        for ia,answer in enumerate(answers):
            if self.parser is None:
                a_words = answer
                if len(a_words.strip()) > 0:
                    search_words = '(%s) AND (%s)' % (q_words, a_words)
                else:
                    search_words = q_words
            else:
                a_words = self.parser.parse(answer, calc_weights=False)
                print '  -> %s' % ' ; '.join(a_words)
                search_words = q_words + a_words
            score = self.lucene_corpus.search(words=search_words, max_docs=self.max_docs, weight_func=self.weight_func, score_func=self.score_func)
            print '  answer #%d: %.2f' % (ia, score)
            scores.append(score)

        if self.norm_scores:
            ref_score0 = np.sum(scores) + len(scores)*EPSILON
            scores = np.array(scores)/ref_score0
        print ' -> scores: %s' % (', '.join(['%.2e'%s for s in scores]))
        return np.asarray(scores)

