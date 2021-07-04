from fastcore.foundation import L
from fastcore.utils import gt
from pytest import fixture
from pytest import raises
from pytest import mark
from typing import (Sequence, Collection, Container, Sized,
        Reversible, Iterable)
from types import SimpleNamespace

@fixture
def l():
    return L(dict(a=1, b=-5, d=6, e=9).items())

def test_itemgot():
    """ like indexing into the iterable (of iterables) """
    l = L(dict(a=1, b=-5, d=6, e=9).items())
    assert l.itemgot(1) == [1, -5, 6, 9]
    assert l.itemgot(0) == ['a', 'b', 'd', 'e']

def test_itemgot2():
    """ like indexing into the iterable (of iterables) """
    l = L([ [1, 2, 3], [1, 2]])
    assert l.itemgot(0) == [1, 1]
    assert l.itemgot(1) == [2, 2,]

    with raises(IndexError): assert l.itemgot(2)
    with raises(IndexError): assert l.itemgot(200)

def test_itemgot3():
    l = L(dict(a=1, b=-5, d=6, e=9).items())
    assert l.itemgot(1).map(abs) == [1,5,6,9,]
    assert l.itemgot(1).map(abs).filter(gt(4)) == [5,6,9]
    assert l.itemgot(1).map(abs).filter(gt(4)).sum() == 20

def test_listy(): 
    assert L(range(10)) == list(range(10))

def test_listy2():
    l = L(range(10))
    l[3] = "YO"; l[8] = "HEY"
    assert l[3] == "YO"
    assert l[8] == "HEY"

@mark.parametrize('t', (Sequence, Collection, Container, Sized, Reversible,
    Iterable))
def test_subclass(t):
    assert issubclass(L, t)
    assert issubclass(list, t)

def test_plus():
    l = L(1, 2, 3)
    assert l + 1 == (1, 2, 3, 1)

    l = L(1, 2, 3)
    assert l + (4, 5, 6) == (1,2,3,4,5,6)

def test_attrgot():
    """ not a dict """
    a = [SimpleNamespace(a=3,b=4), SimpleNamespace(a=1,b=2)]
    assert L(a).attrgot('b') == [4, 2]
    assert L(a).attrgot('a') == [3, 1]

def test_attrgot2():
    a = [{'a': 1, 'b':2}, {'a': 3, 'b':4}]
    assert L(a).attrgot('b') == [2, 4]
    assert L(a).attrgot('a') == [1, 3]

def test_attrgot3():
    a = [{'a': 1, 'b':2}, {'a': 3, 'b':4, 'c':5}]
    assert L(a).attrgot('b') == [2, 4]
    assert L(a).attrgot('a') == [1, 3]
    assert L(a).attrgot('c') == [None, 5]
