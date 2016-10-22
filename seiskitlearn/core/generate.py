'''
Created on Sep 15, 2016

@author: riccardo
'''

import numpy as np

def generate(mu=0, sigma=1, size=1000):
    return np.random.normal(mu, sigma, size)


if __name__ == '__main__':
    d1 = generate(0, 1, size=10)
    d2 = generate(10, 10, size=10)
    print "d1 mu=%f sigma=%f" % (np.mean(d1), np.std(d1))
    print d1
    print ""
    print "d2 mu=%f sigma=%f" % (np.mean(d2), np.std(d2))
    print d2
