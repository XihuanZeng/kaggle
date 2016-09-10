import pandas as pd
import numpy as np
import re
from nltk.stem import PorterStemmer
from Parser import


MARK_ANSWER_ALL  = ' <ALL>'
MARK_ANSWER_BOTH = ' <BOTH>'
MARK_ANSWER_NONE = ' <NONE>'


def sub_complex_answers(train):
    """
    :param train: pandas df
    :return:

    Substitute complex answers like "Both A and B" by the contents of answers A and B,
    "All of the above" by the contents of all answers, and "None of the above" by "".
    We also mark these substitutions for later use. This is done in place
    """
    print 'Substituting complex answers'
    all_re  = re.compile('\s*all of the above\s*')
    both_re = re.compile('\s*both ([a-d]) and ([a-d])[\.]?\s*')
    none_re = re.compile('\s*none of the above\s*')
    for ind,answers in zip(train.index, np.array(train[['answerA','answerB','answerC','answerD']])):
        for ansi,anst in zip(['A','B','C','D'], answers):
            new_ans = None
            all_m = re.match(all_re, anst.lower())
            if all_m is not None:
                new_ans = '%s and %s and %s%s' % (answers[0], answers[1], answers[2], MARK_ANSWER_ALL)
            else:
                both_m = re.match(both_re, anst.lower())
                if both_m is not None:
                    both1, both2 = both_m.groups()[0].upper(), both_m.groups()[1].upper()
                    new_ans = '%s and %s%s' % (answers[ord(both1)-ord('A')], answers[ord(both2)-ord('A')], MARK_ANSWER_BOTH)
                else:
                    if re.match(none_re, anst.lower()) is not None:
                        new_ans = '%s' % MARK_ANSWER_NONE
            if new_ans is not None:
                train.set_value(ind, 'answer%s'%ansi, new_ans)




def prp_binary_dataf(train):
    stemmer = PorterStemmer()
    parser = SimpleWordParser(word_func=stemmer.stem, min_word_length=1, tolower=True, ascii_conversion=True, ignore_special_words=False)
    indices, questions, answers, correct, ans_names, more_cols_vals = [], [], [], [], [], []
    is_all, is_both, is_none, keywords = [], [], [], []
    if 'correctAnswer' in train.columns:
        correct_answer = np.array(train['correctAnswer'])
    else:
        correct_answer = np.zeros(len(train))
    more_cols = [col for col in train.columns if col not in ['question', 'answerA', 'answerB', 'answerC', 'answerD', 'correctAnswer']]
    for idx,(qst,ansA,ansB,ansC,ansD),cor,mcols in zip(train.index, np.array(train[['question', 'answerA', 'answerB', 'answerC', 'answerD']]),
                                                       correct_answer, np.array(train[more_cols])):
        for ia,(ic,ans) in enumerate(zip(['A','B','C','D'],[ansA, ansB, ansC, ansD])):
            indices.append(idx)
            questions.append(qst)
            a_ans, a_all, a_both, a_none, a_keywords = ans, 0, 0, 0, 0
            if ans.endswith(MARK_ANSWER_ALL):
                a_ans = ans[:-len(MARK_ANSWER_ALL)]
                a_all = 1
            elif ans.endswith(MARK_ANSWER_BOTH):
                a_ans = ans[:-len(MARK_ANSWER_BOTH)]
                a_both = 1
            elif ans.endswith(MARK_ANSWER_NONE):
                a_ans = ans[:-len(MARK_ANSWER_NONE)]
                a_none = 1
            else:
                words = parser.parse(ans)
                if 'both' in words:
                    a_both = 0.5
                # note that this is not used
                if stemmer.stem('investigation') in words:
                    a_keywords = 1
            answers.append(a_ans)
            is_all.append(a_all)
            is_both.append(a_both)
            is_none.append(a_none)
            keywords.append(a_keywords)
            # this is for test set
            if cor==0:
                correct.append(0) # no 'correctAnswer' column -> set correct=0 for all answers
            else:
                correct.append(1 if ia==(ord(cor)-ord('A')) else 0)
            ans_names.append(ic)
            more_cols_vals.append(mcols)
    pdict = {'ID': indices, 'question': questions, 'answer': answers, 'correct': correct, 'ans_name': ans_names,
             'is_all': is_all, 'is_both': is_both, 'is_none': is_none} #, 'ans_keywords': keywords}
    for icol,mcol in enumerate(more_cols):
        pdict[mcol] = np.array([vals[icol] for vals in more_cols_vals])
    return pd.DataFrame(pdict)
