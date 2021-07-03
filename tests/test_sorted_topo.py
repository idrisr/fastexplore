from fastcore.dispatch import sorted_topologically as soto
from fastcore.dispatch import lenient_issubclass
from pytest import fixture

#  def sorted_topologically(iterable, *, cmp=operator.lt, reverse=False):

@fixture
def objects():
    class A:pass
    class B(A):pass
    class C(B):pass
    class D(A):pass
    class E(D):pass
    return A,B,C,D,E

def test_sort(objects):
    """
    you might get some ties
    and in that case the order you add the thing matters

    python's reduce automatically gets called with the first two elements of the
    iterable

    """

    A,B,C,D,E = objects
    cmp = lenient_issubclass
    assert soto( [A,C,B,D], cmp=cmp) == [C,B,D,A,]
    assert soto( [A,C,D,B], cmp=cmp) == [C,D,B,A,]
    assert soto( [E,A,C,D,B], cmp=cmp) == [E,C,D,B,A,]
    assert soto( [A,C,D,B,E], cmp=cmp) == [C,E,D,B,A,]

def test_sort2(objects):
    """
    if you use instances and not types, it's all a noop
    """

    A,B,C,D,E = objects
    a,b,c,d,e = [_() for _ in objects]
    cmp = lenient_issubclass
    assert soto( [a,c,b,d], cmp=cmp) == [a,c,b,d,]
    assert soto( [a,c,d,b], cmp=cmp) == [a,c,d,b]
    assert soto( [e,a,c,d,b], cmp=cmp) == [e,a,c,d,b]
    assert soto( [a,c,d,b,e], cmp=cmp) == [a,c,d,b,e]
