from fastcore.dispatch import TypeDispatch

from types import LambdaType
from typing import TypeVar, Generic, List, Optional, Sequence
from numbers import *
import pytest
import numpy as np

"""
`TypeDispatch` has a bunch of functions, and you index into those functions
using a 1-tuple or 2-tuple of types. The function that matches will be returned
,which you can then call.  This is useful because it groups together a set of
functions that have similar outputs and semantics but vary by the type inputs.

the return type of the function is not considered.
This is like function overloading in other languages.

maps types from type annotations to functions

the fancy way of saying is this that the covariant types are accepted
"""

def f1(x: Number):   return 'f1'
def f2(x: Complex):  return 'f2'
def f3(x: Real):     return 'f3'
def f4(x: Rational): return 'f4'
def f5(x: Integral): return 'f5'

def test_real():
    def f(x: Real): return 'f1'
    td = TypeDispatch(f)

    # you can pass in instances because of lenient_issubclass
    assert td[Number]   is None
    assert td[Complex]  is None
    assert td[Real]     is f
    assert td[Rational] is f
    assert td[Integral] is f

    assert td[[]]        is None
    assert td[()]        is None
    assert td[{}]        is None
    assert td[{'1': 2}]  is None
    assert td[str]       is None

@pytest.mark.parametrize('type_,func', [
    (Number,   None),
    (Complex,  None),
    (Real,     f3),
    (Rational, f4),
    (Integral, f4),
    (np.uint8, f4),
    ])
def test_number(type_,func):
    td = TypeDispatch([f3, f4])
    assert td[type_] is func
