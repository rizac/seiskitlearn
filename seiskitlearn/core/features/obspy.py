'''
Created on Oct 21, 2016

@author: riccardo
'''
import numpy as np
from seiskitlearn.core.features import sigmoid_fit, lintrans, sigmoid, Extractor
from itertools import cycle, izip, count


def _xnorm(trace):
    return np.linspace(0, 1, num=len(trace), endpoint=True, dtype=float)


def _cumsum(trace):
    return lintrans(np.cumsum(np.square(trace.data), axis=None, dtype=None, out=None), 0, 1)


def sigmoid_steepness_and_midpoint(x, y, y_is_cumulative=False):
    # stime = trace.stats.starttime
    # etime = trace.stats.delta
    popt, pcov = sigmoid_fit(x, y if y_is_cumulative else _cumsum(y))
    return popt[2], popt[1]


def max_distance(x, y, sigmoid_steepness, sigmoid_midpoint):
    ret = np.abs(y - sigmoid(x, 1, sigmoid_midpoint, sigmoid_steepness))
    amax = np.argmax(ret)
    return x[amax], ret[amax]


class SigmoidExtractor(Extractor):

    featurenames = ['cum_sigmoid_steepness', 'cum_sigmoid_midpoint',
                    'x_of_cum_vs_sigmoid_max_vdist', 'x_of_cum_vs_sigmoid_max_vdist']

    @classmethod
    def features_from_instance(cls, trace):
        x = _xnorm(trace.data)
        cum_data = _cumsum(trace)
        k, x0 = sigmoid_steepness_and_midpoint(x, cum_data, True)
        x_of_max_ydist, max_ydist = max_distance(x, cum_data, k, x0)
        return [k, x0, x_of_max_ydist, max_ydist]
