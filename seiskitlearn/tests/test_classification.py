'''
Created on Sep 25, 2016

@author: riccardo
'''
import unittest
from seiskitlearn.core.classification import fix_array, fix_data # , score, predict, predict_proba
import pytest
import numpy as np
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics.classification import precision_recall_fscore_support
from numpy import array_equal

class Test(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass

#     def test_simple_logreg(self):
#         # mu, sigma, size the arguments below
#         data1 = np.random.normal(1, 10, 500)
#         data2 = np.random.normal(-100, 100, 500)
#         data, target = fix_data(data1, data2)
#         # s = score(LogisticRegression(), data, target)
#         p = predict(LogisticRegression(), data, target)
#         p2 = predict_proba(LogisticRegression(), data, target)
#         assert (np.argmax(p2, axis=1) == p).all()
        
    def test_pre_rec_fscore(self):
        gt1 = np.array([0, 0, 1, 1, 2, 2])
        pred1 = np.array([0, 0, 1, 2, 0, 1])
        
        res1 = precision_recall_fscore_support(gt1, pred1)
        
        gt2 = np.array([2, 2, 0, 0, 1, 1])
        pred2 = np.array([0, 1, 0, 0, 1, 2])
        
        res2 = precision_recall_fscore_support(gt2, pred2)
       
        assert np.array_equal(res1, res2)
        
        gt3 = np.array(['c', 'c', 'a', 'a', 'b', 'b'])
        pred3 = np.array(['a', 'b', 'a', 'a', 'b', 'c'])
        
        res3 = precision_recall_fscore_support(gt2, pred2)
        assert np.array_equal(res1, res3)
        
        # this reorders the labels
        res4 = precision_recall_fscore_support(gt2, pred2, labels=[1, 0, 2])
        
        # this gives all zeros, cause the labels are not in gt2
        res5 = precision_recall_fscore_support(gt2, pred2, labels=['b', 'c', 'a'])
    # from the docs: unittest.TestCase methods cannot directly receive fixture function arguments
    # as implementing
    # that is likely to inflict on the ability to run general unittest.TestCase test suites.

@pytest.mark.parametrize("test_input,expected, use_normal_comparison", [
    (True, True, True),
    ([1], np.array([[1]]), False),
    ((1), np.array([[1]]), False),
    (np.array([1]), np.array([[1]]), False),
    ([1, -0.2], np.array([[1], [-0.2]]), False),
    ((1, -0.2), np.array([[1], [-0.2]]), False),
    (np.array([1, -0.2]), np.array([[1], [-0.2]]), False),
    ([[1], [-0.2]], np.array([[1], [-0.2]]), False),
    (([1], [-0.2]), np.array([[1], [-0.2]]), False),
    (np.array([[1], [-0.2]]), np.array([[1], [-0.2]]), False),
#     ([[1, 2.2], [-0.2]], np.array([[1, 2.2], [-0.2]]), False),
#     (([1, 2.2], [-0.2]), np.array([[1, 2.2], [-0.2]]), False),
#     (np.array([[1, 2.2], [-0.2]]), np.array([[1, 2.2], [-0.2]]), False),
    ([[1, 2.2], [-0.2, 1]], np.array([[1, 2.2], [-0.2, 1]]), False),
    (([1, 2.2], [-0.2, 1]), np.array([[1, 2.2], [-0.2, 1]]), False),
    (np.array([[1, 2.2], [-0.2, 1]]), np.array([[1, 2.2], [-0.2, 1]]), False),
    ])
def testfixarray(test_input, expected, use_normal_comparison):
    assert fix_array(test_input) == fix_array(expected) if use_normal_comparison else \
            (fix_array(test_input) == fix_array(expected)).all()


@pytest.mark.parametrize("test_input,data_expected, target_expected", [
    ([[1], [2.2]], np.array([[1], [2.2]]), np.array([0, 1])),
    ([[1, 3, 5.5], [2.2, 1, 0], [4,4,5]], np.array([[1], [3], [5.5], [2.2], [1], [0], [4], [4], [5]]),
     np.array([0, 0,0, 1,1,1,2,2,2]))
    ])
def testfixdata(test_input, data_expected, target_expected):
    data, target = fix_data(*test_input)
    assert np.array_equal(data, data_expected)
    assert np.array_equal(target, target_expected)
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()