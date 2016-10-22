# good discussion here:  http://stackoverflow.com/questions/4308168/sigmoidal-regression-with-scipy-numpy-python-etc
# curve_fit() example from here: http://permalink.gmane.org/gmane.comp.python.scientific.user/26238
# other sigmoid functions here: http://en.wikipedia.org/wiki/Sigmoid_function

import numpy as np
import pylab
from scipy.optimize import curve_fit
from itertools import count, izip, cycle


def sigmoid(x, L, x0, k):
    """
    Returns the y values of a sigmoid function (aka logistic curve) defined by:

    .. math::

       f(x) = \frac{L} {1+{\mathrm  e}^{{-k(x-x_{0})}}}}

    where e is the natural logarithm base (also known as Euler's number),

    :param x0: the x-value of the sigmoid's midpoint,
    :param L: the curve's maximum value, and
    :param k: = the steepness of the curve.[1]
    """
    y = L / (1 + np.exp(-k*(x-x0)))
    return y


def lintrans(x, min_=0, max_=1):
    """normalizes x (np array) between min_ and max_. There is surely such a function in numpy/
    scipy but I didn't find it. nan values will be returned as they are 9not considered)"""
    x0 = np.nanmin(x)
    x1 = np.nanmax(x)
    return np.true_divide((x - x0), (x1 - x0)) * (max_ - min_) + min_


def sigmoid_fit(x, y, *args, **kwargs):
    return curve_fit(sigmoid, xdata, ydata, *args, **kwargs)


class Extractor(object):

    PARAMS = {}
    FEATURES = []

    @classmethod
    def features(cls, data_list, *args, **kwargs):
        features = np.full((len(data_list), len(cls.FEATURES)), np.nan)
        nominalfeats = [{} for _ in len(features)]
        for i, data_elm in izip(count(), data_list):
            feats = cls.features_from_instance(data_elm, *args, **kwargs)
            try:
                # assign to row. Note that a list of strings parsable to float is ok
                features[i] = feats
            except ValueError:
                # problem... did feats has non numeric values (non-parsable to float?)
                # proceed element wise, not fast but sure:
                for j, feat in enumerate(feats):
                    try:
                        features[i, j] = feat
                    except ValueError:
                        # feat is not a number nor a string convertible (e.g., '5.5')
                        # assign an incremental integer. Two equal feats (according to
                        # their hash) will return the same number. this means equality for strings
                        val = nominalfeats[j].get(feat, None)
                        if val is None:
                            val = nominalfeats[j][feat] = len(nominalfeats[j])
                        features[i, j] = val
        return features

    @classmethod
    def features_from_instance(cls, instance_data, *args, **kwargs):
        raise NotImplementedError("features_from_instance not implemented")


if __name__ == "__main__":
#     xdata = np.array([0.0,   1.0,  3.0, 4.3, 7.0,   8.0,   8.5, 10.0, 12.0])
#     ydata = np.array([0.01, 0.02, 0.04, 0.11, 0.43,  0.7, 0.89, 0.95, 0.99])
    
    xdata = [0,4,5,6,10]  #  np.arange(11).astype(float)  # ([0.0,   1.0,  3.0, 4.3, 7.0,   8.0,   8.5, 10.0, 12.0])
    ydata = [0, 3,4,5, 10]  # np.array([0, 0, 1, 2, 3,  5, 7, 8, 9, 9, 10]).astype(float)
    
    popt, pcov = sigmoid_fit(xdata, ydata, p0=(10, 5, 1))
    print popt
    
    x = np.linspace(0, 10, 1000, endpoint=True, dtype=float)  #  np.linspace(-1, 15, 50)
    y = sigmoid(x, *popt)
    
    pylab.plot(xdata, ydata, 'o', label='data')
    pylab.plot(x, y, label='fit')
    # pylab.ylim(0, 1.05)
    pylab.legend(loc='best')
    pylab.grid(True)
    pylab.show(block=True)