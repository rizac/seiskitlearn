# -*- coding: utf-8 -*-
'''
Created on Oct 18, 2016

@author: riccardo
'''
# from __future__ import unicode_literals
import unittest
import numpy as np
from seiskitlearn.core.utils import Formatter
import sys


class Test(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass

    def test_insert(self):
        a = np.array([[0, 0, 0], [0, 0, 0]])
        b = np.array([[1, 2, 3], [4, 5, 6]])
        c = np.array([[1, 2], [3, 4]])
        index = 1
        ret = []
        ret.append("a = %s" % str(a))
        ret.append("b = %s" % str(b))
        ret.append("np.insert(a, index, b, axis=0)")
        ret.append(str(np.insert(a, index, b, axis=0)))
        b = b[0, :]
        ret.append("b = %s" % str(b))
        ret.append("np.insert(a, index, b, axis=0)")
        ret.append(str(np.insert(a, index, b, axis=0)))
        b = c
        ret.append("b = %s" % str(b))
        ret.append("np.insert(a, index, b, axis=1)")
        ret.append(str(np.insert(a, index, b, axis=1)))
        b = b[:, 0]
        ret.append("b = %s" % str(b))
        ret.append("np.insert(a, index, b, axis=1)")
        ret.append(str(np.insert(a, index, b, axis=1)))
        k = "\n".join(ret)
        h = 9

    def testFormatterHtml(self):
        f = Formatter()
        f.addrow([[1, 1.1, 5], [0, 0, 5]], frmt="{:.2f}")
        print f
        f.addrow([['a', 'b', 7]])
        print f
        f.addrow(float('nan'))
        print f
        f.addcol(float('nan'))
        print f
        
        # now f is:
        # 1.00  1.10  5.00  n/a
        # 0.00  0.00  5.00  n/a
        # a     b     7     n/a
        # n/a   n/a   n/a   n/a
        
        f.set([[1, 1]], 1, 2)
        assert f.report() == """1.00  1.10  5.00  n/a
0.00  0.00  1     1  
a     b     7     n/a
n/a   n/a   n/a   n/a"""

        f.set([[8], [8]], 1, 2)
        assert f.report() == """1.00  1.10  5.00  n/a
0.00  0.00  8     1  
a     b     8     n/a
n/a   n/a   n/a   n/a"""

        f.set(9, 1, 2)
        assert f.report() == """1.00  1.10  5.00  n/a
0.00  0.00  9     1  
a     b     8     n/a
n/a   n/a   n/a   n/a"""

        f.set([[4, 5], [6, 7]], 1, 2)
        assert f.report() == """1.00  1.10  5.00  n/a
0.00  0.00  4     5  
a     b     6     7  
n/a   n/a   n/a   n/a"""

        
        f = Formatter()
        assert f.rows == f.cols == 0
        f.addrow(np.array([[1, 1.1, 5], [0, 0, 5]]))
        
        # note that adding two rows in chain changes the formatting (second argument
        # of add row is integer type)
        assert not np.array_equal(Formatter().addrow([[1, 1.1, 5], [0, 0, 5]])._data,
                              Formatter().addrow([1, 1.1, 5]).addrow([0, 0, 5])._data)
        
        #same for addcol
        assert not np.array_equal(Formatter().addcol([[1, 1.1, 5], [0, 0, 5]])._data,
                              Formatter().addcol([1, 1.1, 5]).addcol([0, 0, 5])._data)
        
        # now they should be equal (note 0.0 instead of 0):
        assert np.array_equal(Formatter().addrow([[1, 1.1, 5], [0, 0, 5]])._data,
                              Formatter().addrow([1, 1.1, 5]).addrow([0.0, 0, 5])._data)
        
        # same for addcol
        assert np.array_equal(Formatter().addcol([[1, 1.1, 5], [0, 0, 5]])._data,
                              Formatter().addcol([1, 1.1, 5]).addcol([0.0, 0, 5])._data)
        
        f.addrow(np.array([[1, 1.1], [0, 0]]), colspans=[(1,2)])
        assert f.report('html') == """<table>
<tr><td>1.0</td>  <td>1.1</td>  <td>5.0</td></tr>
<tr><td>0.0</td>  <td>0.0</td>  <td>5.0</td></tr>
<tr><td>1.0</td>  <td colspan="2">1.1</td></tr>
<tr><td>0.0</td>  <td colspan="2">0.0</td></tr>
</table>"""
        assert f.report('html', numheaders=1) == """<table>
<tr><th>1.0</th>  <th>1.1</th>  <th>5.0</th></tr>
<tr><td>0.0</td>  <td>0.0</td>  <td>5.0</td></tr>
<tr><td>1.0</td>  <td colspan="2">1.1</td></tr>
<tr><td>0.0</td>  <td colspan="2">0.0</td></tr>
</table>"""

        assert f.report('rst') == """===  ===  ===
1.0  1.1  5.0
0.0  0.0  5.0
1.0  1.1     
---  --------
0.0  0.0     
---  --------
===  ===  ==="""

        assert f.report('rst', aligns='right') == """===  ===  ===
1.0  1.1  5.0
0.0  0.0  5.0
1.0       1.1
---  --------
0.0       0.0
---  --------
===  ===  ==="""

        f.addcol([1, 4, 1.1, 1], frmt=int)
        assert f.report('rst', aligns='right') == """===  ===  ===  =
1.0  1.1  5.0  1
0.0  0.0  5.0  4
1.0       1.1  1
---  --------  -
0.0       0.0  1
---  --------  -
===  ===  ===  ="""

        f.addcol([[1, 4, 1.1, 1], [0, 0, float('nan'), 0]], frmt=int)
        assert f.report('rst', aligns='right', numheaders=2) == """===  ===  ===  =  =  ===
1.0  1.1  5.0  1  1    0
0.0  0.0  5.0  4  4    0
===  ===  ===  =  =  ===
1.0       1.1  1  1  n/a
---  --------  -  -  ---
0.0       0.0  1  1    0
---  --------  -  -  ---
===  ===  ===  =  =  ==="""

        f.insertrow("caption", 0)
        assert f.report('rst', aligns='right', numheaders=2) == """=======  =======  =======  =======  =======  =======
caption  caption  caption  caption  caption  caption
    1.0      1.1      5.0        1        1        0
=======  =======  =======  =======  =======  =======
    0.0      0.0      5.0        4        4        0
    1.0               1.1        1        1      n/a
-------  ----------------  -------  -------  -------
    0.0               0.0        1        1        0
-------  ----------------  -------  -------  -------
=======  =======  =======  =======  =======  ======="""

#THIS WOULD BE INTERESTING!!!:
#  f.insertrow(['a', '≥', 'g', 5.5, 7, None], 1)
#         assert f.report('rst', aligns='right', numheaders=2) == u"""=======  =======  =======  =======  =======  =======
# caption  caption  caption  caption  caption  caption
#       a        ≥        g      5.5        7     None
# =======  =======  =======  =======  =======  =======
#     1.0      1.1      5.0        1        1        0
#     0.0      0.0      5.0        4        4        0
#     1.0               1.1        1        1      n/a
# -------  ----------------  -------  -------  -------
#     0.0               0.0        1        1        0
# -------  ----------------  -------  -------  -------
# =======  =======  =======  =======  =======  ======="""

        f.insertrow(['a', '', 'g', 5.5, 7, None], 1)
        assert f.report('rst', aligns='right', numheaders=2) == """=======  =======  =======  =======  =======  =======
caption  caption  caption  caption  caption  caption
      a                 g      5.5        7     None
=======  =======  =======  =======  =======  =======
    1.0      1.1      5.0        1        1        0
    0.0      0.0      5.0        4        4        0
    1.0               1.1        1        1      n/a
-------  ----------------  -------  -------  -------
    0.0               0.0        1        1        0
-------  ----------------  -------  -------  -------
=======  =======  =======  =======  =======  ======="""

        f.insertcol(float('nan'), -1)
        assert f.report('rst', aligns='right', numheaders=2) == """=======  =======  =======  =======  =======  ===  =======
caption  caption  caption  caption  caption  n/a  caption
      a                 g      5.5        7  n/a     None
=======  =======  =======  =======  =======  ===  =======
    1.0      1.1      5.0        1        1  n/a        0
    0.0      0.0      5.0        4        4  n/a        0
    1.0               1.1        1        1  n/a      n/a
-------  ----------------  -------  -------  ---  -------
    0.0               0.0        1        1  n/a        0
-------  ----------------  -------  -------  ---  -------
=======  =======  =======  =======  =======  ===  ======="""

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testFormatter']
    unittest.main()