'''
Created on Sep 23, 2016

@author: riccardo
'''
import unittest
import numpy as np
from sklearn.linear_model.logistic import LogisticRegression, LogisticRegressionCV
from seiskitlearn.core.classification import fix_array
from seiskitlearn.core.evaluation import Eval, classindex, classlabel
import pytest
from sklearn.metrics.classification import brier_score_loss, log_loss

class Test(unittest.TestCase):


    def setUp(self):
#         size = 10
#         data0 = np.random.normal(1, 10, size=size)
#         data1 = np.random.normal(10, 1, size=size)
#         self.target = np.append(np.zeros(len(data0)), np.ones(len(data1)))
#         self.data = fix_array(np.append(data0, data1))
#         self.logreg = LogisticRegression()
#         p = self.logreg.get_params()
#         self.logreg.fit(self.data, self.target)
#         p = self.logreg.get_params()
#         j = 9
        pass

    def tearDown(self):
        pass


    def testCM_auc(self):
        s = Eval(np.array([1, 0, 0, 1, 1, 2]), np.array([[0, 1, 0],
                                                  [0, 1, 0],
                                                  [0, 0, 1],
                                                  [1, 0, 0],
                                                  [1, 0, 0],
                                                  [0, 0.5, .5]]), labels=[0,1],
                                                  classnames=['Ok','LowSNR'])

        print "This should have two labels 'Ok' and 'LowSNR':"
        print ""
        print s.report(format_='rst')
        print ""

        y_true = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        ## this has many good classified, but with low probas
        y_pred1 = [[.55, .45],
                  [.55, .45],
                  [.55, .45],
                  [.55, .45],
                  [0, 1],
                  #
                  [.45, .55],
                  [.45, .55],
                  [.45, .55],
                  [1, 0],
                  [.45, .55],
                  ]


     # this has fewer good classified, but with higher probas
        y_pred2 = [[.9, 0.1],
                   [.8, 0.2],
                   [1, 0],
                   [.45, .55],
                   [.4, .6],
                   #
                   [0, 1],
                   [.2, .8],
                   [0, 1],
                   [.55, .45],
                   [.65, .35],
                   ]
        labelz__ = [1, 2]
        c1 = Eval(y_true, y_pred1, labelz__)
        c2 = Eval(y_true, y_pred2, labelz__)

        assert (np.asarray(c1.au_roc()) < np.asarray(c2.au_roc())).all()

        print "Three different printouts (with probability distributions): None, rst, html:\n"
        print c1.report('html')
        print ""
        print c1.report('rst')
        print ""
        print c1.report('')
        print ""
        print c1.prob_dist()
        print "This evaluation should have lower fscore accuracy etcetera but better au_roc score"
        print " and lower (better) brier and log loss:"
        print ""
        print c2.report('')
        print ""
        print c2.prob_dist()



@pytest.mark.parametrize("test_input,expected", [
    ([0, 0.5, 0.5], 1),
    ([0, 1, 0], 1),
    ([[0, 0.5, 0.5]], [1]),
    ([[0, 0.5, 0.5], [.9, 0.1, 0]], [1, 0]),
    ])
def test_classindex(test_input, expected):
    assert np.array_equal(classindex(test_input), expected)


@pytest.mark.parametrize("test_input,labels,expected", [
    ([0, 0.5, 0.5], ['a', 'b', 'c'], 'b'),
    ([0, 1, 0], ['a', 'b', 'c'], 'b'),
    ([[0, 0.5, 0.5], [.9, 0.1, 0]], ['a', 'b', 'c'], ['b', 'a']),
    ([[0, 0.5, 0.5]], ['a', 'b', 'c'], ['b']),
    ])
def test_classlabel(test_input, labels, expected):
    assert np.array_equal(classlabel(test_input, labels), expected)


@pytest.mark.parametrize("labels,probas,expected", [  # probas is probas of pos_label=1
                                                    #  (cci below means correctly classified instances)
    ([1,1, 0, 0], [1, 1, 0, 0], 0),  # BEST CASE scenario score = 0
    ([1,1, 0, 0], [0, 0, 1, 1], 1),  # worse case scenario score = 1
    ([1,1, 0, 0], [.5, .5, .5, .5], .25), # THIS IS HIGHER THAN THE PREVIOUS (although 4 cci > 2)
    ([1,1, 0, 0], [.6, .6, .4, .4], .16), # THIS IS HIGHER THAN THE PREVIOUS (although 4 cci > 2)
    ([1,1, 0, 0], [1, .4, .6, 0], .18),  # THIS IS HIGHER THAN THE PREVIOUS (2 cci < 4)
    ])
def test_brier_score(labels, probas, expected):
    assert np.isclose(brier_score_loss(labels,probas), expected)


@pytest.mark.parametrize("labels,probas,expected", [  # probas is probas of pos_label=1
                                                    #  (cci below means correctly classified instances)
    ([1,1, 0, 0], [1, 1, 0, 0], 0),  # BEST CASE scenario score = 0
    ([1,1, 0, 0], [0, 0, 1, 1], 34.5391761936),  # worse case scenario score = 1
    ([1,1, 0, 0], [.5, .5, .5, .5], 0.69314718056), # THIS IS HIGHER THAN THE PREVIOUS (although 4 cci > 2)
    ([1,1, 0, 0], [.6, .6, .4, .4], 0.510825623766), # THIS IS HIGHER THAN THE PREVIOUS (although 4 cci > 2)
    ([1,1, 0, 0], [1, .4, .6, 0],  0.458145365937),  # THIS IS HIGHER THAN THE PREVIOUS (2 cci < 4)
    ])
def test_log_score(labels, probas, expected):
    assert np.isclose(log_loss(labels,probas), expected)
    

def test_brier_score_multinomial():
    y_true = [1, 2, 0]
    y_prob = [0, 1, 0]
    
    v0 = brier_score_loss(y_true, y_prob, pos_label=0)
    v1 = brier_score_loss(y_true, y_prob, pos_label=1)
    v2 = brier_score_loss(y_true, y_prob, pos_label=2)
    
    assert np.isclose(v0, 0.6666666)
    assert np.isclose(v1, 0.6666666)
    assert np.isclose(v2, 0)
    
    v0 = brier_score_loss(y_true, y_prob, pos_label=1)
    v1 = brier_score_loss(y_true, y_prob, pos_label=1)
    v2 = brier_score_loss(y_true, y_prob, pos_label=1)
    
    assert np.isclose(v0, 0.6666666)
    assert np.isclose(v1, 0.6666666)
    assert np.isclose(v2, 0.6666666)


@pytest.mark.parametrize("labels,probas,expected", [  # probas is probas of pos_label=1
                                                    #  (cci below means correctly classified instances)
    ([0, 1, 2], [[1, 0, 0],[0, 1, 0],[0, 0, 1]], 0),  # BEST CASE scenario score = 0
    ([0, 1, 2], [[0, 1, 0],[1, 0, 0],[1, 0, 0]], 34.5387763949),  # WORST CASE scenario score = 0
    ([0, 1, 2], [[0, 0, 1],[1, 0, 0],[1, 0, 0]], 34.5387763949),  # ANOTHER WORST CASE scenario score = 0
    ])
def test_log_score_multnomial(labels, probas, expected):
    v = log_loss(labels,probas)
    assert np.isclose(v, expected)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()