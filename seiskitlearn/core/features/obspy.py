'''
Created on Oct 21, 2016

@author: riccardo
'''
import numpy as np
from core.features import sigmoid_fit, lintrans, sigmoid
from itertools import cycle, izip, count


def _xnorm(trace):
    return np.linspace(0, 1, num=len(trace), endpoint=True, dtype=float)


def _cumsum(trace):
    return lintrans(np.cumsum(np.square(trace.data), axis=None, dtype=None, out=None), 0, 1)


def sigmoid_steepness_and_midpoint(x, y, y_is_cumulative=False):
    data = y if y_is_cumulative else _cumsum(y)
    # stime = trace.stats.starttime
    # etime = trace.stats.delta
    popt, pcov = sigmoid_fit(x, data)
    return popt[2], popt[1]


def max_distance(x, y, sigmoid_steepness, sigmoid_midpoint):
    ret = np.abs(y - sigmoid(x, 1, sigmoid_midpoint, sigmoid_steepness))
    amax = np.argmax(ret)
    return x[amax], ret[amax]


class Extractor(object):

    PARAMS = {}

    @classmethod
    def extract(cls, ydata, xdata=None):
        ydata = np.asarray(ydata)
        if xdata is not None:
            xdata = np.asarray(xdata)
            if xdata.shape != ydata.shape:
                raise ValueError("xdata and ydata do not have the same shape")

        if len(ydata.shape) == 1:
            ydata = ydata.reshape((1, ydata.shape[0]))
            if xdata is not None:
                xdata = xdata.reshape((1, ydata.shape[0]))

        features = np.full_like(ydata, np.nan)
        for i, x, y in izip(count(), [None] if xdata is None else xdata, ydata):
            features[i] = cls.features(y, x)

        return cls.featurnaes(), features,

    @classmethod
    def featurenames(cls):
        raise NotImplementedError("You need to subclass Extractor providing a dict like object "
                                  "of strings mapped to features ")

    @classmethod
    def features(cls, *data, **kwdata):
        raise NotImplementedError("You need to subclass Extractor providing a numpy array "
                                  "of features")


class SigmoidExtractor(Extractor):

    def extract(self, trace):
        x = _xnorm(trace.data)
        cum_data = _cumsum(trace)
        k, x0 = sigmoid_steepness_and_midpoint(x, cum_data, True)
        x_of_max_ydist, max_ydist = max_distance(x, cum_data, k, x0)
        return dict(k=k, x0=x0, x_of_max_ydist=x_of_max_ydist, max_ydist=max_ydist)
