# -*- coding: utf-8 -*-
'''
Created on Sep 22, 2016

@author: riccardo
'''
# from __future__ import unicode_literals
import sys
import numpy as np
from sklearn.metrics.classification import precision_recall_fscore_support, confusion_matrix,\
    classification_report, accuracy_score, brier_score_loss, log_loss
from sklearn.metrics import roc_curve, auc
from itertools import izip, cycle, count
from sklearn.utils.multiclass import unique_labels
from seiskitlearn.core import utils


class Eval(object):

    RST_COLSEP = "  "
    FLOAT_FORMAT = '{:.2f}'  # '{:.1%}'

    def __init__(self, ground_truth, predicted, labels=None, classnames=None, instances_names=None):
        """
            :param ground_truth: an array of classes (usually integers from 0 on), denoting the
            ground truth
            :param predicted: an array of classes (usually integers from 0 on), denoting the
            predicted class.
            It can be also an array of arrays of probabilities, in such case each array element P
            is in turn an array of N elements, where N = number of classes. The i-th element of P
            denotes the probability of the i-th class.
            :param labels: List of labels to index the confusion matrix. This may be used to
            reorder or select a subset of labels.
            If none is given, those that appear at least once in y_true or y_pred are used in
            sorted order.
            If predicted is an array of arrays of probabilities, the list is used also to retrieve
            the predicted classes from the `predicted` array (Using i=argmax and then
            assigning the corresponding label[i]). You can pass `classes_` attribute of the
            estimator. If None, labels in `predicted` are assumed to be integers from 0,1,2,..
            etcetera, and must match the labels in `ground_truth`
            :param classnames: additional names for each class label to be printed in the confusion
            matrix
            :param instance_names: an array the same length of ground_truth with additional names
            for each instance
            :Example:

            Eval([0,1,0], [1,1,0])
            Eval([0,1,2], [1,2,0])
            Eval([0,1,2], [1,2,0], labels=[1, 2])  # do not print 0
            Eval([0,1,0], [[.1,.9],[1, 0, 0],[]], labels=['No', 'Yes'])
        """
        self._cm = self._as = self._classnames = None
        # init with empty tuples cause otherwise pydev complains with (false positives) errors:
        self._prfs = {'micro': (), 'macro': (), 'samples': (), 'weighted': (), None: ()}
        self._bsl = self._auroc = self._rocdata = None
        # convert to numpy if not numpy:
        classnames = np.asarray(classnames) if classnames is not None else None
        labels = np.asarray(labels) if labels is not None else None
        if ((labels is not None) and (classnames is not None)) and (len(labels) != len(classnames)):
            raise ValueError(("Mismatch between labels and classnames (both None or arrays with "
                              "same length)"))

        self._y_true = np.asarray(ground_truth)
        predicted = np.asarray(predicted)
        self._y_pred_probas = np.array([])
        self._uncertain = {'_y_true': np.array([]), '_y_pred_probas': np.array([])}

        input_has_probas = len(predicted.shape) == 2
        if input_has_probas:  # probabilities case
            # handle multi maxima, e,g, probability of an instance is [0.4 0.4 0.3]
            # we decide NOT to display it in the confusion matrix. FIXME: what does scikit learn do?

            self._y_pred_probas = predicted
            self._y_pred = classindex(predicted) if labels is None else classlabel(predicted,
                                                                                   labels)
            indices_with_uncertain_max = \
                np.apply_along_axis(lambda probas: len(np.where(probas == probas.max())[0]) != 1,
                                    axis=1, arr=predicted)
            if indices_with_uncertain_max.any():
                self._uncertain['_y_true'] = self._y_true[indices_with_uncertain_max]
                self._uncertain['_y_pred_probas'] = predicted[indices_with_uncertain_max]
                indices_with_single_max = ~indices_with_uncertain_max
                self._y_true = self._y_true[indices_with_single_max]
                self._y_pred = self._y_pred[indices_with_single_max]
                self._y_pred_probas = predicted[indices_with_single_max]
        else:
            self._y_pred = np.asarray(predicted)

        unique_sorted_labels = unique_labels(self._y_pred, self._y_true)
        self._all_labels = unique_sorted_labels
        if labels is None:
            self._labels = unique_sorted_labels
        else:
            self._labels = labels

        self._classnames = classnames

    @property
    def classes(self):
        """Same as self.labels"""
        return self.labels

    @property
    def labels(self):
        """Returns the labels (class labels). Usually, class labels are integers from 0 on"""
        return self._labels

    @property
    def classnames(self):
        """Returns the user supplied class names (if any), or None"""
        return self._classnames

    def dict(self, score, *args, **kwargs):
        """Returns a dict with keys `self.labels` mapped to the values returned by the `score`
        function. Useful to have e.g. self.precision(), self.recall() self.brier_score_loss() ...
        as dicts, when needed.
        :param score: A callable (e.g. a function, or amethod of this object) or a string denoting
        a class method, which will be called with given `*args` and `**kwargs`. The returned value
        of the callable **MUST** obviously be an array of values/object in the same order as
        returned by `self.labels`
        :Example:
        Given a
            - c = Eval(..., labels=[2,1,0])
                  (note that labels order matters), then
            - c.precision()[0] = precision of its 1st element, i.e. class `2` (see above)
            - c.dict('precision')[0] = c.dict(c.precision)[0] = precision of key `0`, i.e. class `0`
        """
        try:
            return dict(izip(self.labels, getattr(self, score)(*args, **kwargs))) \
                if hasattr(self, score) else dict(izip(self.labels, score(*args, **kwargs)))
        except (AttributeError, ValueError, TypeError):
            raise ValueError("Bad method or string in `dict` method: %s " % str(score))

    def precision(self, average=None):
        """Returns the precision of this evaluation.
       :param average: None or string, ['micro', 'macro', 'samples', 'weighted']
        If None, the scores for each class are returned, in the same order as `self.labels` (thus
        given a cm instance of Eval, a dictionary-like object can be obtained with:
        `dict(zip(cm.labels, cm.precision())` or with the `dict` method of this class)
        If average is not None, this determines the type of averaging performed on the data:
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).
        """
        return self._summarystat(average)[0]  # 0 = precision, regardless if it's scalar or array

    def recall(self, average=None):
        """Returns the recall of this evaluation.
       :param average: None or string, ['micro', 'macro', 'samples', 'weighted']
        If None, the scores for each class are returned, in the same order as `self.labels` (thus
        given a cm instance of Eval, a dictionary-like object can be obtained with:
        `dict(zip(cm.labels, cm.precision())` or with the `dict` method of this class)
        If average is not None, this determines the type of averaging performed on the data:
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).
        """
        return self._summarystat(average)[1]  # 1 = recall, regardless if it's scalar or array

    def f1score(self, average=None):
        """Returns the precision of this evaluation.
       :param average: None or string, ['micro', 'macro', 'samples', 'weighted']
        If None, the scores for each class are returned, in the same order as `self.labels` (thus
        given a cm instance of Eval, a dictionary-like object can be obtained with:
        `dict(zip(cm.labels, cm.precision())` or with the `dict` method of this class)
        If average is not None, this determines the type of averaging performed on the data:
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).
        """
        return self._summarystat(average)[2]  # 2 = fscore, regardless if it's scalar or array

    def support(self):
        """Returns the support (number of instances) of a given (or all) label(s)
        :param label: any label returned by `self.labels`. If None (default if missing), all labels
        values  (in the same order as `self.labels`) are returned"""
        return self._summarystat(None)[3]

    def num_instances(self):
        """Returns the total number of instances"""
        return self.support().sum()

    def total_support(self):
        """Same as num_instances"""
        return self.num_instances()

    def accuracy_score(self, normalize=True):
        """Returns the accuracy score"""
        if self._as is None:
            self._as = accuracy_score(self._y_true, self._y_pred, normalize=normalize)
        return self._as

    def log_loss(self, eps=1e-2, normalize=False):
        if not self.has_probas():
            return np.nan
        logloss = log_loss(self._y_true, self._y_pred_probas, eps=eps)
        if normalize:
            # get the minimum:
            logloss /= -np.log(eps)
        return logloss

    def _summarystat(self, average):
        if self._prfs[average] in (None, ()):
            self._prfs[average] = precision_recall_fscore_support(self._y_true, self._y_pred,
                                                                  labels=self.labels,
                                                                  average=average, pos_label=None)
        return self._prfs[average]

    def confusion_matrix(self):
        """Returns the confusion matrix (numpy matrix). The class labels are in the same order they
        are defined in this object (see `self.labels)`"""
        if self._cm is None:
            self._cm = confusion_matrix(self._y_true, self._y_pred, labels=self.labels)
        return self._cm

    def has_probas(self):
        """Returns True if the input result of the classifier has been given as probabilities"""
        return not (self._y_pred_probas is None or not self._y_pred_probas.size)

    def _probs_array_of(self, label):
        y_pred_probas = self._y_pred_probas  # pylint: disable=W0621
        label_index = np.argwhere(self.labels == label).ravel()
        if len(label_index) != 1:
            raise ValueError("Unable to find label '%s'" % str(label))
        label_index = label_index[0]
        y_score = np.apply_along_axis(lambda row: row[label_index], axis=1, arr=y_pred_probas) \
            if self.has_probas() else np.full((len(self._y_pred),), np.nan)
        return y_score

    def brier_score_loss(self, average=None):
        """Returns the brier score loss. Average can be only 'macro' or 'weighted' as this
        is a custom method"""
        if self._bsl is None:
            self._bsl = np.full((len(self.labels),), np.nan) if not self.has_probas() else \
                np.array([brier_score_loss(self._y_true, self._probs_array_of(label),
                                           pos_label=label) for label in self.labels])
        if not average:
            return self._bsl
        weights = self.support() if average == 'weighted' else None
        return np.average(self._bsl, weights=weights)

    def au_roc(self, average=None):
        """The mean area under roc curve.
            average can be either macro (no class weights) or weighted"""
        aucs = np.full((len(self.labels),), np.nan) if not self.has_probas() else \
            np.array([auc(r[0], r[1]) for r in self._roc_data()])
        if not average:
            return aucs
        weights = self.support() if average == 'weighted' else None
        return np.average(aucs, weights=weights)

    def _roc_data(self):
        """returns an array of [(fpr tpr, thresholds)] where each tuple refers to
        the i-th label (in the same order as returned from self.labels. This method
        is used to plot roc curves of probability thresholds AND areas under curve.
        **IT ASSUMES THE INPUT DATA HAS BEEN GIVEN AS PROBABILITIES!**
        """
        if self._rocdata is None:
            self._rocdata = []
            for label in self.labels:
                y_prob = self._probs_array_of(label)
                self._rocdata.append(roc_curve(self._y_true, y_prob, pos_label=label))
        return self._rocdata

    def plot_roc(self, fig=None, axs=None, plot_thresholds=False):  # FIXME: has to be tested!
        """Plots the roc curve. DEPRECATED, has to be fixed and re-tested!"""
        import matplotlib.pyplot as plt
        show_now = False
        if fig is None and axs is None:
            fig = plt.figure()
            show_now = True
        if axs is None:
            axs = fig.gca()  # creates one if non existing

        lwidth = 2
        base_colors = ['navy', 'darkorange', 'red', 'aqua', 'green', 'deeppink', 'cornflowerblue']
        colors = cycle(base_colors)
        linestyles = cycle(['-']*len(base_colors) + ['--']*len(base_colors))
        binarycase = len(self.labels) == 2
        # in binary case roc curves are identical, print different label
        labelz = [self.labels[0]] if binarycase else self.labels
        for lbl, clr, lstyle in izip(labelz, colors, linestyles):
            fpr, tpr, thres, auc_ = self.roc_curve(lbl, return_auc=True)
            if binarycase:
                plotlabel = 'Roc curve (area = %0.2f)' % auc_
            else:
                plotlabel = 'Class label "%s" (area = %0.2f)' % (str(lbl), auc_)
            axs.plot(fpr, tpr, color=clr, linestyle=lstyle, lw=lwidth, label=plotlabel, marker='o')
            if not plot_thresholds:
                continue
            for xpos, ypos, thr in izip(fpr, tpr, reversed(thres)):
                deltax = deltay = 0.01
                hal = 'left'
                if xpos > 0.9:
                    deltax -= 0.02
                    hal = 'right'
                axs.text(xpos+deltax, ypos+deltay, r'$\theta=%.1f$' % thr, ha=hal)

        axs.plot([0, 1], [0, 1], color='black', lw=lwidth, linestyle=':')
        axs.set_xlim([0.0, 1.0])
        axs.set_ylim([0.0, 1.05])
        axs.set_xlabel('False Positive Rate')
        axs.set_ylabel('True Positive Rate')
        axs.set_title('Receiver operating characteristic')
        axs.legend(loc="lower right")
        if show_now:
            plt.show(block=True)

    def __str__(self):
        return self.report(None)

    def report(self, format_=None, float_format=None,
               stats=('precision', 'recall', 'f1score', 'support',
                      'brier_score_loss', 'au_roc', 'accuracy_score', 'log_loss')):
        """
            Returns a string representation of this object. Similar to scikit learn report but
            prints also the confusion matrix AND/OR formats in html, if needed
            :param format_: either 'rst' or 'html' or None. In the latter case, a rst-simplified
            form (for even more readability) is returned
            :param float_format: the format used for numeric floating point data (precision,
            recall, f-measure and their averages). Defaults to Eval.FLOAT_FORMAT = '{:.2f}', i.e.
            two-digits float format. To use percentages (with, e.g., 1 decimal digit), type '{:.1%}'
        """

        # this is a typical outcome (when format_= None):

        # Label        Classified as             Precision  Recall  F1-score  Support
        #                  0       1
        # ==========  ======  ======  =========  =========  ======  ========  =======
        # 0 = LowSNR       0       1                  0.0%    0.0%      0.0%        1
        # 1 = Ok           0       3                 60.0%  100.0%     75.0%        3
        #
        #                               Mean ->      30.0%   50.0%     37.5%
        #                             W.Mean ->      45.0%   75.0%     56.2%
        # ==========  ======  ======  =========  =========  ======  ========  =======

        labels = self.labels
        numlabels = len(labels)

        # if labels are given as integers, convert them:
        if np.array_equal(labels, labels.astype(int)):
            labels = labels.astype(int)
        # convert to strings:
        labels = labels.astype(str)

        formatter = utils.Formatter()
        _repl = {'brier_score_loss': 'brier_loss', 'accuracy_score': 'accuracy'}
        # then create headings:
        heading_captions = ['label', "classified as", ''] + \
            [_repl[x] if x in _repl else x for x in stats]
        formatter.addrow(heading_captions, colspans=[(1, numlabels)])
        formatter.addrow([''] + labels.tolist() + [''] * (len(heading_captions) - 2))
        formatter.expand(numrows=numlabels+3)

        formatter.set(self.confusion_matrix(), 2, 1)
        # set labels on first col:
        bold = ('**', '**') if format_ == 'rst' else ("<strong>", "</strong>") \
            if format_ == 'html' else ("", "")
        classnames = cycle([None]) if self._classnames is None else self._classnames
        for i, lbl, name in izip(count(start=2), labels, classnames):
            name = (" = %s" % name) if name is not None else ""
            formatter.set("%s%s%s%s" % (bold[0], lbl, bold[1], name), i, 0)

        # define function for displaying stats (given a method as string representing the stat
        # function)
        def _calc(method, *args, **kwargs):
            try:
                return getattr(self, method)(*args, **kwargs)
            except (TypeError, AttributeError):
                return None

        def numeric_format(method):
            """function returning a numeric format according to the score method"""
            return int if method == 'support' else (float_format or Eval.FLOAT_FORMAT)

        avg_total_row = 3 + numlabels
        for i, method in enumerate(stats, numlabels + 2):
            score = _calc(method)
            if score is None:
                continue
            nfm = numeric_format(method)
            if np.isscalar(score):  # global score, set it to 'avg/total' cell:
                formatter.set(score, 3 + numlabels, i, frmt=nfm)
            else:
                formatter.set(score.reshape(numlabels, 1), 2, i, frmt=nfm)
                if method == 'support':
                    formatter.set(self.num_instances(), avg_total_row, i, frmt=nfm)
                else:
                    formatter.set(_calc(method, average='macro'), avg_total_row, i, frmt=nfm)
                    formatter.set(_calc(method, average='weighted'), 1 + avg_total_row, i, frmt=nfm)
        formatter.set('avg/total', avg_total_row, numlabels+1)
        formatter.set('w.avg', avg_total_row+1, numlabels+1)

        # convert to string:
        return formatter.report(format_, numheaders=2)

    def prob_dist(self, steps=None, cumulative=False):
        return ProbDist(self, steps, cumulative)


class ProbDist(object):
    """An object taking an Eval object as input and printing the probability distributions per
    class"""

    def __init__(self, eval_obj, steps=None, cumulative=False):
        self._eval = eval_obj
        self._steps = None
        self._cumulative = None
        self._data = None
        self.calculate(steps, cumulative)

    def calculate(self, steps=None, cumulative=False):
        if steps is None:
            steps = np.array(xrange(1, 11)) / 10.0
        steps.sort()
        self._steps = steps
        self._cumulative = cumulative
        ret = []

        if not self._eval.has_probas():
            return np.array(3 * [len(steps) * [np.nan]])

        for label in self._eval.labels:
            data = self._eval._probs_array_of(label)[self._eval._y_true == label]
            dist = np.zeros(len(steps))
            min_ = -np.inf
            for i, thr in enumerate(steps):
                dist[i] = len(data[(data > min_) & (data <= thr)])
                min_ = thr
            if cumulative:
                dist = np.cumsum(dist)
            ret.append(dist)

        self._data = np.asarray(ret).astype(int)

    def __str__(self):
        return self.report(None)

    def report(self, format_=None, float_format="{:.2f}"):
        arrstr = self._data.astype(str)
        formatter = utils.Formatter()
        formatter.addrow(arrstr, frmt=float_format)
        header = [(("&le;" if format_ == 'html' else "≤") + float_format).format(v)
                  for v in self._steps]
        formatter.insertrow(header, 0)
        labels = [['label']] + [["%s" % str(k)] for k in self._eval.labels]
        formatter.insertcol(labels, 0)
        aligns = ['left'] + ['right'] * len(self._steps)
        return formatter.report(format_, numheaders=1, aligns=aligns)


def classindex(class_proba):
    """
        Returns the class index (or indices) which is the maximum of class_proba.
        :param class_proba: a (N,M) numpy array of N instances each with M classes (in that case the
        returned array has length N, or a numpy array of length M (in that case the returned
        value is a scalar)
    """
    try:
        return np.argmax(class_proba, axis=1)
    except ValueError:  # we might have passed a flattened array, i.e. [0.2, 0.6 0.4]
        return np.argmax(class_proba)


def classlabel(class_proba, labels=None):
    """
        Returns the class label(s) which is the maximum of class_proba.
        :param class_proba: a (N,M) numpy array of N instances each with M classes (in that case the
        returned array has length N, or a numpy array of length M (in that case the returned
        value is a scalar)
        :param labels: if None, returns the index of class_proba, assuming that labels are
        denoted by increasing integers starting from 0: [0, 1, ..., N]. Otherwise the label in
        labels corresponding to the index of max(class_proba)
    """
    cidx = classindex(class_proba)
    if labels is not None:
        if np.isscalar(cidx):
            return labels[cidx]
        elif not np.array_equal(labels, xrange(len(labels))):  # otherwise just return cidx
            ret = np.empty(shape=cidx.shape, dtype=np.array(labels).dtype)
            for idx, lbl in enumerate(labels):
                ret[np.where(cidx == idx)[0]] = lbl
            return ret
    return cidx
