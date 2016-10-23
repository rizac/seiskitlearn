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
    return curve_fit(sigmoid, x, y, *args, **kwargs)


class Extractor(object):
    """Feature extractor. The user should call the method
    ```features(data_list,...) for getting a numpy MxN matric of features
    for training / testing / fitting any scikit learn classifier
    **after** overriding the class method
    ```features_from_instance```
    **and** the class member
    ```featurenames```

    :example:
    ```
        class MyExtractor(Extractor):
            featurenames = ['mean', 'sum']

            @classmethod
            def features_from_instance(cls, instance, *args, **kwargs):
                array = np.asarray(instance)
                return array.mean(), array.sum()

        # and then:

        data = [...]                            # Whatever ...
        features = MyExtractor.features(data)   # With optional positional or nominal args,
                                                # if needed inside `features` method.
                                                # In this case, no arguments are supplied
        feature_names = MyExtractor.featurenames
    ```
    """

    params = {}
    """Optional class parameters (as dict) accessible in `feature_from_instance`"""
    featurenames = []
    """The array (list tuple numpy array) of feature names. The length of the array/list
    returned by `feature_from_instance` must match the length of this array
    """

    @classmethod
    def features(cls, data_list, *args, **kwargs):
        """Extracts features from data_list. This method basically returns a numpy NxM array
        where N = len(data_list) and M is the length of the class method `featurenames` by calling
        N times the abstract method `feature_from_instance` (which MUST be overridden by subclasses)
        :param data_list a list/iterable/array of instances on which to extract their features by
        means of `feature_from_instance`.
        :param args: optional positional argument which will be passed to `feature_from_instance`
        :param kwargs: optional nominal argument which will be passed to `feature_from_instance`
        """
        return np.array([cls.features_from_instance(elm, *args, **kwargs) for elm in data_list])

    @classmethod
    def features_from_instance(cls, instance, *args, **kwargs):
        """This abstract method performs the actual computaion on a single instance.
        The returned value should match the length of the class attribute `featurenames`
        If some values are nominal, it is up to the user to convert them to numeric if possible
        (this allows homogeneus numpy arrays to be returned)
        :param instance The instance on which to extract their features
        :param args: optional positional argument passed from the class method `features`
        :param kwargs: optional nominal argument passed from the class method `features`
        """
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