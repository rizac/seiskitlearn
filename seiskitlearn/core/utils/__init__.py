# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import sys
import numpy as np
from numpy import empty_like
from itertools import izip, count, cycle


class Formatter(object):
    """
    A class which allows pretty printing numpy matrices (2D numpy arrays) in rst, plain text or
    html, which features such as cells text alignment, headers, and colspans.
    An object of this class wraps a numpy matrix and has methods to print it such as
    `report(...)`. The string method of this class calls `report(format_=None)`
    """
    def __init__(self, colsep="  ", nanrepr='n/a'):
        """
            Initializes a new Formatter
            :param colsep: the column separator (defaults to two spaces). This is only for visual
            readability, it won't be rendered in not html or rst. Rst suggests to keep it at least
            two space wide
            :param nanrepr: the string to be used when encountering NaNs (defaults to "n/a")
        """
        self._colsep = colsep
        self._data = np.empty(shape=(), dtype=str)
        self._colspans = np.empty(shape=(), dtype=int)
        self._header_num = 0
        self._nanrepr = nanrepr
        self._ispy2 = sys.version_info[0] < 3

    @property
    def cols(self):
        """Returns the number of columns of the underlying numpy matrix"""
        return 0 if not len(self._data.shape) else self._data.shape[1]

    @property
    def rows(self):
        """Returns the number of rows of the underlying numpy matrix"""
        return 0 if not len(self._data.shape) else self._data.shape[0]

    def expand(self, numrows=0, numcols=0):
        """Expands the underlying matrix with empty cells. You can use `self.set` later
        to populate the newly created cells"""
        if numrows:
            k = np.full((numrows, 1), '', dtype=str)  # FIXME: str or unicode?
            self.addrow(k)
        if numcols:
            k = np.full((numcols, 1), '', dtype=str)  # FIXME: str or unicode?
            self.addcol(k)
        return self

    def set(self, numpy_array_or_matrix, rowindex, colindex, frmt=None):
        """Sets the data at a given rowindex, colindex
        :param numpy_array_or_matrix: a scalr, a list, or a numpy array. If array of shape > 0,
        rowindex and colindex refer to the upper left underlying matrix, and all values will be
        set. Use an array of single-values arrays, e.g. [[1], [2], [2]] to set a column of values,
        (as it normally happens for numpy arrays)
        :param rowindex: the (upper left) rowindex
        :param colindex the (upper left) column index
        :param frmt: the format used to convert each array value into string. It can be
        a callable (e.g., `str`, `int`, `float`) taking as argument each array value, a string
        with a given format (e.g., "{:d}", "{:.2f}"). If missing or None, defaults to `str`
        """
        ret, _ = self._format(numpy_array_or_matrix, frmt)
        rowlen = 1 if not len(ret.shape) > 0 else ret.shape[0]
        collen = 1 if not len(ret.shape) > 1 else ret.shape[1]
        self._data[rowindex: rowindex + rowlen, colindex: colindex + collen] = ret
        return self

    def addrow(self, numpy_array_or_matrix, frmt=None, colspans=None):
        """
        Adds a row to the end of the underlying matrix
        :param numpy_array_or_matrix: a numpy 2D, 1D array, or scalar. If 1D array, the array
        is converted to [array], if 0D (scalar), the scalar is converted to [array, ..., array]
        (`self.cols` times)
        :param frmt: the format used to convert each array value into string. It can be
        a callable (e.g., `str`, `int`, `float`) taking as argument each array value, a string
        with a given format (e.g., "{:d}", "{:.2f}"). If missing or None, defaults to `str`
        :param colspans: optional list of colspans, indicating which columns should span. The list
        should have elements of the form (start, length), where both values are integers
        indicating the column start (from 0) and the number of columns to span, respectively.
        The elements needs not to be a tuple, they can be any two-element array
        (lists/tuples/numpy arrays etcetera)

        :examples:
        ```
            f = Formatter()

            f.addrow([[1, 1.1, 5], [0, 0, 5]], frmt="{:.2f}")
            1.00  1.10  5.00
            0.00  0.00  5.00

            f.addrow([['a', 'b', 7]])  # same as f.addrow(['a', 'b', 7])
            1.00  1.10  5.00
            0.00  0.00  5.00
            a     b     7

            f.insertrow(float('nan'), -1)
            1.00  1.10  5.00
            0.00  0.00  5.00
            n/a   n/a   n/a
            a     b     7

            f.addcol([[1, 1.1, 5], [0, 0, 5]])  # raises an error (3-element arrays, 4- needed)

            f.addcol([1, 2, 3, 4])  # same as f.addcol([[1, 2, 3, 4]])
            1.00  1.10  5.00  1
            0.00  0.00  5.00  2
            n/a   n/a   n/a   3
            a     b     7     4
        ```
        """
        return self.insertrow(numpy_array_or_matrix, 0 if not self._data.shape else
                              self._data.shape[0], frmt, colspans)

    def insertrow(self, numpy_array_or_matrix, rowindex, frmt=None, col_spans=None):
        """
        Inserts a row at a specified index in the underlying matrix
        :param numpy_array_or_matrix: a numpy 2D, 1D array, or scalar. If 1D array, the array
        is converted to [array], if 0D (scalar), the scalar is converted to [array, ..., array]
        (`self.cols` times)
        :param rowindex: the row index. Negative values count from the end, as usual
        :param frmt: the format used to convert each array value into string. It can be
        a callable (e.g., `str`, `int`, `float`) taking as argument each array value, a string
        with a given format (e.g., "{:d}", "{:.2f}"). If missing or None, defaults to `str`
        :param colspans: optional list of colspans, indicating which columns should span. The list
        should have elements of the form (start, length), where both values are integers
        indicating the column start (from 0) and the number of columns to span, respectively.
        The elements needs not to be a tuple, they can be any two-element array
        (lists/tuples/numpy arrays etcetera)

        :examples:
        ```
            f = Formatter()

            f.addrow([[1, 1.1, 5], [0, 0, 5]], frmt="{:.2f}")
            1.00  1.10  5.00
            0.00  0.00  5.00

            f.addrow([['a', 'b', 7]])  # same as f.addrow(['a', 'b', 7])
            1.00  1.10  5.00
            0.00  0.00  5.00
            a     b     7

            f.insertrow(float('nan'), -1)
            1.00  1.10  5.00
            0.00  0.00  5.00
            n/a   n/a   n/a
            a     b     7

            f.addcol([[1, 1.1, 5], [0, 0, 5]])  # raises an error (3-element arrays, 4- needed)

            f.addcol([1, 2, 3, 4])  # same as f.addcol([[1, 2, 3, 4]])
            1.00  1.10  5.00  1
            0.00  0.00  5.00  2
            n/a   n/a   n/a   3
            a     b     7     4
        ```
        """
        if np.isscalar(numpy_array_or_matrix) and col_spans is not None:
            raise ValueError('Inserting row with scalar value needs col_spans=None or missing')
        ret, colspans = self._format(numpy_array_or_matrix, frmt)
        if col_spans:
            for start, length in sorted(col_spans, key=lambda c: c[0], reverse=True):
                colspans[:, start] = length
                slice_ = np.arange(start+1, start+length)
                colspans = np.insert(colspans, slice_, 0, axis=1)
                ret = np.insert(ret, slice_, '', axis=1)

        self._insert(ret, colspans, rowindex, axis=0)
        return self

#     def insertemptyrow(self, numpy_array_or_matrix, rowindex, render=True):
#         self.insertrow([''] * self.cols, rowindex)
#         if render:
#             self._colspans[rowindex, :] = 0
#             self._colspans[rowindex, 0] = self.cols

    def addcol(self, numpy_array_or_matrix, frmt=None):
        """
        Adds a column to the end of the underlying matrix
        :param numpy_array_or_matrix: a numpy 2D, 1D array, or scalar. **Note that the array ROWS
        will be considered matrix columns**. If 1D array, the array
        is converted to [array], if 0D (scalar), the scalar is converted to [array, ..., array]
        (`self.cols` times)
        :param frmt: the format used to convert each array value into string. It can be
        a callable (e.g., `str`, `int`, `float`) taking as argument each array value, a string
        with a given format (e.g., "{:d}", "{:.2f}"). If missing or None, defaults to `str`
        :param colspans: optional list of colspans, indicating which columns should span. The list
        should have elements of the form (start, length), where both values are integers
        indicating the column start (from 0) and the number of columns to span, respectively.
        The elements needs not to be a tuple, they can be any two-element array
        (lists/tuples/numpy arrays etcetera)

        :examples:
        ```
            f = Formatter()

            f.addrow([[1, 1.1, 5], [0, 0, 5]], frmt="{:.2f}")
            1.00  1.10  5.00
            0.00  0.00  5.00

            f.addrow([['a', 'b', 7]])  # same as f.addrow(['a', 'b', 7])
            1.00  1.10  5.00
            0.00  0.00  5.00
            a     b     7

            f.insertrow(float('nan'), -1)
            1.00  1.10  5.00
            0.00  0.00  5.00
            n/a   n/a   n/a
            a     b     7

            f.addcol([[1, 1.1, 5], [0, 0, 5]])  # raises an error (3-element arrays, 4- needed)

            f.addcol([1, 2, 3, 4])  # same as f.addcol([[1, 2, 3, 4]])
            1.00  1.10  5.00  1
            0.00  0.00  5.00  2
            n/a   n/a   n/a   3
            a     b     7     4
        ```
        """
        return self.insertcol(numpy_array_or_matrix, 0 if not self._data.shape else
                              self._data.shape[1], frmt)

    def insertcol(self, numpy_array_or_matrix, colindex, frmt=None):
        """
        Inserts a column at a specified index in the underlying matrix
        :param numpy_array_or_matrix: a numpy 2D, 1D array, or scalar. **Note that the array ROWS
        will be considered matrix columns**. If 1D array, the array
        is converted to [array], if 0D (scalar), the scalar is converted to [array, ..., array]
        (`self.cols` times)
        :param colindex: the column index. Negative values count from the end, as usual
        :param frmt: the format used to convert each array value into string. It can be
        a callable (e.g., `str`, `int`, `float`) taking as argument each array value, a string
        with a given format (e.g., "{:d}", "{:.2f}"). If missing or None, defaults to `str`
        :param colspans: optional list of colspans, indicating which columns should span. The list
        should have elements of the form (start, length), where both values are integers
        indicating the column start (from 0) and the number of columns to span, respectively.
        The elements needs not to be a tuple, they can be any two-element array
        (lists/tuples/numpy arrays etcetera)

        :examples:
        ```
            f = Formatter()

            f.addrow([[1, 1.1, 5], [0, 0, 5]], frmt="{:.2f}")
            1.00  1.10  5.00
            0.00  0.00  5.00

            f.addrow([['a', 'b', 7]])  # same as f.addrow(['a', 'b', 7])
            1.00  1.10  5.00
            0.00  0.00  5.00
            a     b     7

            f.insertrow(float('nan'), -1)
            1.00  1.10  5.00
            0.00  0.00  5.00
            n/a   n/a   n/a
            a     b     7

            f.addcol([[1, 1.1, 5], [0, 0, 5]])  # raises an error (3-element arrays, 4- needed)

            f.addcol([1, 2, 3, 4])  # same as f.addcol([[1, 2, 3, 4]])
            1.00  1.10  5.00  1
            0.00  0.00  5.00  2
            n/a   n/a   n/a   3
            a     b     7     4
        ```
        """
        ret, colspans = self._format(numpy_array_or_matrix, frmt)
        self._insert(ret, colspans, colindex, axis=1)
        return self

    def _format(self, array, frmt=None):
        """
        Formats the elements of the given array according to frmt and returns the tuple
        stringarray, colspans, where stringarray is the numpy string array of formatted elemtns
        and colspans is a numpy array of ones with the same shape as array (the latter will be
        used by the calling methods)"""
        array = np.asarray(array)
        if len(array.shape) == 1:
            array = array.reshape((1, array.shape[0]))
        shape = array.shape
        dtype = unicode
        # convert to string cause we might have passed frmt = int or float
        row = np.array([self._formatcell(value, frmt) for value in array.flatten()], dtype=dtype).\
            reshape(shape)
        colspans = np.ones(shape=shape)
        return row, colspans

    def _formatcell(self, value, frmt):
        """Formats value into string according to `frmt`"""
        ret = ''
        isnan = False
        try:
            isnan = np.isnan(value)
        except TypeError:
            pass
        if isnan:
            ret = self._nanrepr
        else:
            try:
                if frmt is None:
                    frmt = str
                ret = frmt(value) if hasattr(frmt, "__call__") else \
                    frmt.format(value) if hasattr(frmt, "format") else ''
            except (TypeError, ValueError):
                pass
        return ret.decode('utf8') if self._ispy2 and hasattr(ret, 'decode') else ret

    def _insert(self, data, colspans, index, axis):
        """insert method, private, called by public class methods. Updates the underlying numpy
        matrix"""
        if not self._data.shape:
            self._data = data.T if axis == 1 else data
            self._colspans = colspans.T if axis == 1 else colspans
        else:
            # insert (unless vstack and hstack) does
            # NOT account for string lengths, resulting in strings cutoff.
            if self._data.dtype.itemsize < data.dtype.itemsize:
                self._data = self._data.astype(data.dtype)
            self._data = np.insert(self._data, index, data, axis=axis)
            self._colspans = np.insert(self._colspans, index, colspans, axis=axis)

    def __str__(self):
        return self.report(None)

    def report(self, format_=None, aligns=None, numheaders=0, **table_attrs):
        """Prints a report of this object, i.e., a pretty print version of the underlying matrix
        in table form
        :param format_: The format of the output table. None by default, it can be either 'html',
        'rst' or plain text (anything else)
        :param aligns: string or array of strings (default None): the alignments of each column.
        If string, it can be either 'left' or 'right' or None, and will be applied to all table
        cells. If array, its length must match `self.cols` and must be made of strings equal to
        either 'left', 'right' or None
        :param numheaders: integer (0 by default): the number of headers
        :param table_attrs: used only if format_ is 'html', the optional table attributes. To
        specify a class attribute, being 'class' a reserved python keyword, type e.g.,
        **{'class': 'myclass'}
        """
        ret = self._report_html(aligns, numheaders, **table_attrs) if format_ == 'html' else \
            self._report_plain(format_ == 'rst', aligns, numheaders)

        if self._ispy2:  # re-convert to byte string if python2
            ret = ret.encode('utf8', errors='ignore')

        return ret

    def _report_html(self, aligns=None, numheaders=0, **table_attrs):
        if aligns is None or np.isscalar(aligns):
            aligns = cycle([aligns])
        ret = ["<table>" if not table_attrs else
               "<table%s>" % "".join(" %s=\"%s\"" % (k, v) for k, v in table_attrs.iteritems())]
        for i, row, colspans in izip(count(), self._data, self._colspans):
            rowstr = []
            for cell, colspan, align in izip(row, colspans, aligns):
                if not colspan:
                    continue
                attr = ""
                if colspan > 1:
                    attr = " colspan=\"%d\"" % colspan
                if align in ('left', 'right'):
                    attr += " style=\"text-align:%s\"" % align
                tag = "th" if i < numheaders else "td"
                rowstr.append("<%s%s>%s</%s>" % (tag, attr, cell, tag))
            ret.append("<tr>%s</tr>" % self._colsep.join(rowstr))
        ret.append("</table>")

        return "\n".join(ret)

    def _report_plain(self, is_rst, aligns=None, numheaders=0):
        if aligns is None or np.isscalar(aligns):
            aligns = cycle([aligns])
        colwidths = self._maxcolwidths(self._data)
        colwidths[0] = max(colwidths[0], 2)  # rst needs first col cells to escape spaces, if empty
        rst_header = self._rst_line(colwidths) if is_rst else ""
        ret = []
        for i, row, colspans in izip(count(), self._data, self._colspans):
            if is_rst and i in (0, numheaders):
                ret.append(rst_header)
            rowstr = []
            for j, cell, align in izip(count(), row, aligns):
                if not colspans[j]:
                    continue
                colwidth = colwidths[j:j+colspans[j]].sum() + len(self._colsep) * (colspans[j] - 1)
                frmt = "{:%s%d}" % (">" if align == 'right' else "<" if align == 'left' else '',
                                    colwidth)
                stringval = frmt.format(cell)
                # see above, rst needs escaping:
                if j == 0 and is_rst and not stringval.strip():
                    stringval = "\\ " + stringval[2:]
                rowstr.append(stringval)
            ret.append(self._colsep.join(rowstr))
            if colspans[colspans > 1].any():
                ret.append(self._rst_line(colwidths, colspans))
        if is_rst:
            ret.append(rst_header)

        return "\n".join(ret)

    def _rst_line(self, colwidths, colspans=None):
        colsep = self._colsep
        if colspans is None:
            return colsep.join("=" * l for l in colwidths)
        colspansep = '-' * len(colsep)
        char = '-'
        ret = []
        for i, cspan in enumerate(colspans):
            if not cspan:
                continue
            ret.append(colspansep.join(char * cwd for cwd in colwidths[i:i+cspan]))
        return colsep.join(ret)

    def _maxcolwidths(self, string_matrix, min_colwidth=1):
        """Returns the max column widths of each column in string_matrix, accounting for
        python2 strings to account for the right number of characters)
        Note mincolwidth = 1 cause otherwise string.format does not work (e.g. "{:0}".format(...))
        """
#         if sys.version_info[0] < 3:
#             return np.apply_along_axis(lambda r: max(min_colwidth,
#                                                      max(len(x.decode('utf8')) for x in r)), 0,
#                                        string_matrix)

        data1 = np.array(self._data, copy=True)
        dataN = np.array(self._data, copy=True)

        data1[self._colspans != 1] = ''
        dataN[self._colspans <= 1] = ''

        maxlens = np.apply_along_axis(self._maxlen, 0, data1)
        maxlenspans = np.apply_along_axis(self._maxlen, 0, dataN)

        for col, maxlen, maxlenspan in izip(count(), maxlens, maxlenspans):
            # nonzero is the same as argwhere, but it does NOT
            # transpose the matrix (we don't want it
            colspans = self._colspans[:, col]
            for colspan in colspans[colspans > 1]:
                total_colinterspace = (colspan-1) * len(self._colsep)
                reallen = maxlens[col:col+colspan].sum() + total_colinterspace
                if reallen < maxlenspan:
                    maxlens[col: col+colspan] = max(maxlen,
                                                    np.ceil(np.true_divide(maxlenspan -
                                                                           total_colinterspace,
                                                            colspan)))
        return maxlens

    def _maxlen(self, nparray, min_colwidth=1):
        return max(min_colwidth, max(len(x) for x in nparray))