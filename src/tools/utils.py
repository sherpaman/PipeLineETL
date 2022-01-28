from collections.abc import Iterable
from numbers import Number
import os
import re
import json

import pandas
import numpy
from tqdm import tqdm, notebook


def is_iterable(i):
    return isinstance(i, Iterable) and not isinstance(i, str)


def is_unique_list(lst):
    return len(lst) == len(set(lst))


def is_numeric(n):
    return isinstance(n, Number)


def any_in(l1, l0):
    """
    Return True if any element on l1 is in l0
    """
    return any([set([i]) <= set(l0) for i in l1])


def key_form_val(d: dict, val):
    return list(d.keys())[list(d.values()).index(val)]


def get_tqdm_iter(ls):
    if "JPY_PARENT_PID" in os.environ:
        gen_list = notebook.tqdm(ls)
    else:
        gen_list = tqdm(ls)
    return gen_list


def find_field(d, k: str, kk: str):
    if k in d:
        p = d[k]
        if type(p) is str:
            return [p]
        else:
            return p[kk]
    else:
        return []


def read_label(filename, idx):
    id_match = re.compile('[\S\\\\]*_([0-9]*)\.csv')
    with open(filename, 'r') as f:
        label = json.load(f)
    out = []
    for d in label:
        t = {'container_device_id': int(id_match.findall(d['csv'])[0]),
             'quality': d['quality'] }
        problem_list = find_field(d, 'problems', 'choices')
        for p in problem_list:
            t[p] = True
        out.append(t)
    out = pandas.DataFrame(out).fillna(False).set_index(idx)
    return out


def log_scale(p0, w, n):
    c = int(numpy.log10(p0))
    return numpy.logspace(c - w, c + w, n)


def p_scale(p0, w, n, min_p=0, max_p=100):
    a, b = max(min_p, p0 - w), min(max_p, p0 + w)
    return numpy.linspace(a, b, n)


def lin_scale(p0, w, n):
    return numpy.linspace(p0 - w, p0 + w, n)