from fastcore.dispatch import _TypeDict as TypeDict
import pytest
from typing import Container

@pytest.fixture
def td():
    return TypeDict()

@pytest.fixture
def objects():
    class A:    pass
    class B(A): pass
    class C(B): pass
    def f1():   pass
    def f2():   pass
    return A, B, C, f1, f2


def test_td_add(td):
    # add a func
    assert td.first() is None
    def f(x): return x
    td.add(int, f)
    assert td.first() is f

def test_td_add(td, objects):
    A, B, *_, f1, f2 = objects
    td.add(A, f1)
    td.add(B, f2)

    assert td[B] is f2
    assert td[A] is f1

def test_td_add2(td, objects):
    """ 
    the class you ask for doesn't have to be in the dictionary 
    only one of it's superclasses need to be there. 
    The first superclass is returned. 
    Each time you add an item, the dict gets sorted via sorted_topologically
    """

    A, B, C, f1, f2 = objects
    class D(A): pass
    class E(C): pass
    td.add(B, f2)
    td.add(A, f1)

    assert td[C]   is f2
    assert td[A]   is f1
    assert td[D]   is f1
    assert td[E]   is td[B]
    assert td[D]   is td[A]
    assert td[int] is None
