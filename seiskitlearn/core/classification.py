# -*- coding: utf-8 -*-
'''
Created on Sep 22, 2016

@author: riccardo
'''

from __future__ import print_function
# from sklearn import cross_validation as c_v  # , datasets
from sklearn.model_selection import cross_val_score, cross_val_predict
import numpy as np
from sklearn.utils import indexable
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils.validation import _num_samples
import scipy.sparse as sp
from sklearn.base import clone, is_classifier
# from itertools import product, izip
from core.evaluation import Eval
from sklearn.linear_model.logistic import LogisticRegression
# from sklearn.utils.multiclass import unique_labels
# from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm.classes import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import classification_report
# from sklearn.svm import SVC

# from sklearn.linear_model.logistic import LogisticRegression, LogisticRegressionCV
# from sklearn.metrics.classification import precision_recall_fscore_support


# def score(estimator, X, y=None, scoring='f1', cv=10, n_jobs=1, verbose=1,
#           fit_params=None, pre_dispatch='2*n_jobs'):
#     """Same as sklearn.cross_validation.score"""
#     return c_v.cross_val_score(estimator, X, y, scoring, cv, n_jobs, verbose, fit_params,
#                                pre_dispatch)
# 
# 
# def predict(estimator, X, y=None, cv=10, n_jobs=1, verbose=1,
#             fit_params=None, pre_dispatch='2*n_jobs'):
#     """Same as sklearn.cross_validation.predict"""
#     return c_v.cross_val_predict(estimator, X, y, cv, n_jobs, verbose, fit_params, pre_dispatch)
# 
# 
# def predict_proba(estimator, X, y=None, cv=10, n_jobs=1, verbose=1, fit_params=None,
#                   pre_dispatch='2*n_jobs'):
#     """Generate cross-validated estimates for each input data point. This method is the SAME
#     as sklearn.cross_validation.predict *but* it returns an array where each elements has
#     the class probabilities (array of N elements where N=number of classes) instead of the
#     predicted class index (scalar)
# 
#     Read more in the :ref:`User Guide <cross_validation>`.
# 
#     Parameters
#     ----------
#     estimator : estimator object implementing 'fit' and 'predict'
#         The object to use to fit the data.
# 
#     X : array-like
#         The data to fit. Can be, for example a list, or an array at least 2d.
# 
#     y : array-like, optional, default: None
#         The target variable to try to predict in the case of
#         supervised learning.
# 
#     cv : int, cross-validation generator or an iterable, optional
#         Determines the cross-validation splitting strategy.
#         Possible inputs for cv are:
# 
#         - None, to use the default 3-fold cross-validation,
#         - integer, to specify the number of folds.
#         - An object to be used as a cross-validation generator.
#         - An iterable yielding train/test splits.
# 
#         For integer/None inputs, if ``y`` is binary or multiclass,
#         :class:`StratifiedKFold` used. If the estimator is a classifier
#         or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
# 
#         Refer :ref:`User Guide <cross_validation>` for the various
#         cross-validation strategies that can be used here.
# 
#     n_jobs : integer, optional
#         The number of CPUs to use to do the computation. -1 means
#         'all CPUs'.
# 
#     verbose : integer, optional
#         The verbosity level.
# 
#     fit_params : dict, optional
#         Parameters to pass to the fit method of the estimator.
# 
#     pre_dispatch : int, or string, optional
#         Controls the number of jobs that get dispatched during parallel
#         execution. Reducing this number can be useful to avoid an
#         explosion of memory consumption when more jobs get dispatched
#         than CPUs can process. This parameter can be:
# 
#             - None, in which case all the jobs are immediately
#               created and spawned. Use this for lightweight and
#               fast-running jobs, to avoid delays due to on-demand
#               spawning of the jobs
# 
#             - An int, giving the exact number of total jobs that are
#               spawned
# 
#             - A string, giving an expression as a function of n_jobs,
#               as in '2*n_jobs'
# 
#     Returns
#     -------
#     preds : ndarray of arrays
#         This is the result of calling 'predict_proba'
#     """
#     # copied from cross_validation.predict
#     X, y = indexable(X, y)  # pylint: disable=W0632
# 
#     cv = c_v.check_cv(cv, X, y, classifier=is_classifier(estimator))
#     # We clone the estimator to make sure that all the folds are
#     # independent, and that it is pickle-able.
#     parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
#     preds_blocks = parallel(delayed(_fit_and_predict_proba)(clone(estimator), X, y,
#                                                             train, test, verbose,
#                                                             fit_params)
#                             for train, test in cv)
# 
#     preds = [p for p, _ in preds_blocks]
#     locs = np.concatenate([loc for _, loc in preds_blocks])
#     if not c_v._check_is_partition(locs, _num_samples(X)):
#         raise ValueError('cross_val_predict only works for partitions')
#     inv_locs = np.empty(len(locs), dtype=int)
#     inv_locs[locs] = np.arange(len(locs))
# 
#     # Check for sparse predictions
#     if sp.issparse(preds[0]):
#         preds = sp.vstack(preds, format=preds[0].format)
#     else:
#         preds = np.concatenate(preds)
#     return preds[inv_locs]
# 
# 
# def _fit_and_predict_proba(estimator, X, y, train, test, verbose, fit_params):
#     """Fit estimator and predict values for a given dataset split.
#     Read more in the :ref:`User Guide <cross_validation>`.
#     Parameters
#     ----------
#     estimator : estimator object implementing 'fit' and 'predict'
#         The object to use to fit the data.
#     X : array-like of shape at least 2D
#         The data to fit.
#     y : array-like, optional, default: None
#         The target variable to try to predict in the case of
#         supervised learning.
#     train : array-like, shape (n_train_samples,)
#         Indices of training samples.
#     test : array-like, shape (n_test_samples,)
#         Indices of test samples.
#     verbose : integer
#         The verbosity level.
#     fit_params : dict or None
#         Parameters that will be passed to ``estimator.fit``.
#     Returns
#     -------
#     preds : sequence
#         Result of calling 'estimator.predict'
#     test : array-like
#         This is the value of the test parameter
#     """
#     # Adjust length of sample weights
#     fit_params = fit_params if fit_params is not None else {}
#     fit_params = dict([(k, c_v._index_param_value(X, v, train))
#                       for k, v in fit_params.items()])
# 
#     X_train, y_train = c_v._safe_split(estimator, X, y, train)
#     X_test, _ = c_v._safe_split(estimator, X, y, test, train)
# 
#     if y_train is None:
#         estimator.fit(X_train, **fit_params)
#     else:
#         estimator.fit(X_train, y_train, **fit_params)
#     preds = estimator.predict_proba(X_test)
#     return preds, test


def fix_array(obj):
    """
        Converts obj to a scikit matrix of shape (M,N), where M is assumed to be the number of
        instances and N the number of features.
        Basically, checks if obj if a python "standard array" (having the '__len__' attribute but
        not the 'shape' one) or a 1-dimension numpy array (having the 'shape' attribute whose
        length is 1). In any of these two cases, returns numpy.reshape(obj, (M, N)) where
        M=len(obj) and N is the length of obj[0] (or 1 if obj[0] has no length, e.g. scalar)

        Example:
        fix_array([1,2,3]) = numpy.array([[1], [2], [3]])
        fix_array((1,2,3)) = numpy.array([[1], [2], [3]])
        fix_array(numpy.array([1,2,3])) = numpy.array([[1], [2], [3]])
        fix_array(anything_else) = anything_else
    """
    if (hasattr(obj, "__len__") and not hasattr(obj, 'shape')) \
            or (hasattr(obj, 'shape') and len(obj.shape) == 1):
        try:
            cols = len(obj[0])
        except TypeError:
            cols = 1

        return np.reshape(obj, (len(obj), cols))
    return obj


def fix_data(*class_features):
    """
        Converts the arrays class_features into a single matrix, basically calling numpy.vstack,
        but returns also a vector of classes from 0 to len(class_features)-1

        Example:
        data1 = [1,2,3], data2 = [3,4,5]
        fix_data(data1, data2) returns
        np.array([1,2,3,3,4,5]), np.array([0,0,0,1,1,1])

        The same is obtained by passing data1 = [[1], [2], [3]] and/or data2 =  [[3], [4], [5]]
        as fix_array is called prior to numpy.vstack
    """
    data = []
    target = []
    for idx, obj in enumerate(class_features):
        fix_obj = fix_array(obj)
        target = np.append(target, idx*np.ones(len(fix_obj)))
        data.append(fix_obj)
    data = np.vstack(data)
    return data, target


def gridsearch(estimator, X, y, param_grid, cv=10,
               scores=['precision_macro', 'recall_macro'],
               test_method='predict_proba',
               test_size=0.25):
    """
    Performs a grid search with cross validation and prints the classification
    report for the best parameters found on a subset of X,y (test set).
    Function copied and modified from
    http://scikit-learn.org/stable/auto_examples/model_selection/grid_search_digits.html#sphx-glr-auto-examples-model-selection-grid-search-digits-py

    :param estimator: estimator object. This is assumed to implement the scikit-learn
    estimator interface.

    :param X: array-like, shape = [n_samples, n_features]. Training vector, where
    n_samples is the number of samples  and n_features is the number of features.

    :param y: array-like, shape = [n_samples] or [n_samples, n_output].
    Target (class labels in the supervised case) relative to X for classification.

    :param param_grid: dict or list of dictionaries passed as argument to `GridSearchCV`
    Dictionary with parameters names (string) as keys and lists of
    parameter settings to try as values, or a list of such
    dictionaries, in which case the grids spanned by each dictionary
    in the list are explored. This enables searching over any sequence
    of parameter settings.

    :param cv : int, cross-validation generator or an iterable  passed as
    argument to `GridSearchCV` (defaults to 10 when missing)
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross validation,
          - integer, to specify the number of folds in a `(Stratified)KFold`,
          - An object to be used as a cross-validation generator.
          - An iterable yielding train, test splits.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

    :param scores: an array of object  passed as argument to `GridSearchCV`, where each object
    can be: string, callable or None, default=None
    A string (see model evaluation documentation) or
    a scorer callable object / function with signature
    ``scorer(estimator, X, y)``.
    If ``None``, the ``score`` method of the estimator is used.

    :param test_method: the method to be called on the classifier to evaluate the
    results on the test set. Can be eother 'predict_proba' (the default) or 'predict'

    :param  test_size: float, int, or None (default is 0.2)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset (X and y) to include in the test split. If
        int, represents the absolute number of test samples. If None,
        test size is set to 0.25.

    :Example:
    ```
    # given X and y arrays (e.g., X=MxN matrix of features, y=M-length array of class labels):

    # perform Grid Search on logistic regression C parameter:
    tuned_parameters = {'C': [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e25]}
    gridsearch(LogisticRegression(), X, y, tuned_parameters, test_size=0.1,
                scores=['f1_macro', 'neg_log_loss'])

    # perform Grid Search on two support vector machine parameter sets:
    tuned_parameters = tuned_parameters = [{'kernel': ['rbf'],'gamma': [1e-3, 1e-4],
                                            'C': [1, 10, 100, 1000]},
                                           {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    gridsearch(SVC(C=1), X, y, tuned_parameters, test_size=0.1,
                scores=['f1_macro', 'neg_log_loss'])
    ```
    """
    frmt = 'html'  # None # 'rst'

    def print_(string, decoration_char, overline=False):
        """print function for rst sections"""
        if overline:
            print(decoration_char*len(string))
        print(string)
        print(decoration_char*len(string))
        print()

    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=0)

    print_(estimator.__class__.__name__ + " GridSearchCV", "=", True)
    for score in scores:
        print_("Tuning hyper-parameters for %s" % str(score), "=")

        clf = GridSearchCV(estimator, param_grid=param_grid, cv=cv,
                           scoring=score)
        clf.fit(X_train, y_train)

        print_("Best parameters set found on development set:", "-")
        print(clf.best_params_)
        print()

        print_("Grid scores on development set:", "-")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print()

        print_("Detailed classification report:", "-")
        print("The model is trained on the full development set (%d instances)." % len(y_train))
        print("The scores are computed on the full evaluation set (%d instances)." % len(y_test))
        print()

        y_true, y_pred = y_test, getattr(clf, test_method)(X_test)
        evl = Eval(y_true, y_pred)
        print(evl.report(frmt))
        # print()
        print_("Probabilities distribution (if classifier allows it)", "`")
        print(evl.prob_dist().report(frmt))
        print()


def generate_normal_random_data(n_samples=100):
    """this is our test set, it's just a straight line with some
    Gaussian noise. Returns the tuple X, y where X are a (Nx1) features vector
    (i.e., single feature) and y is a N array with are the class labels in [0,1]"""
    # xmin, xmax = -5, 5
    np.random.seed(0)
    X = np.random.normal(size=n_samples)
    y = (X > 0).astype(np.float)
    X[X > 0] *= 4
    X += .3 * np.random.normal(size=n_samples)
    X = X[:, np.newaxis]
    return X, y

if __name__ == '__main__':
    X_, y_ = generate_normal_random_data(10000)
    scorez = ['neg_log_loss']  # ['f1_macro', 'neg_log_loss']
    tuned_parameters = {'C': [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e25]}
    gridsearch(LogisticRegression(), X_, y_, tuned_parameters, test_size=0.1,
               scores=scorez)

    gridsearch(GaussianNB(), X_, y_, {}, test_size=0.1,
               scores=scorez)

    gridsearch(SVC(C=1, probability=True), X_, y_, [
                                  #  {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                                  {'kernel': ['linear'], 'C': [1]}
                                  ], test_size=0.1,
               scores=scorez)

    # ret = gridsearch([SVC()], X, y, test_size=0.2)
    # for r in ret:
    #    print(r.report())
    #pass
#     X, y = generate_normal_random_data(10000)
#     estimators = get_estimators(LogisticRegression, C=[1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e25])
#     ret = gridsearch(estimators, X, y, test_size=0.2)
#     # ret = gridsearch([SVC()], X, y, test_size=0.2)
#     for r in ret:
#         print(r.report())