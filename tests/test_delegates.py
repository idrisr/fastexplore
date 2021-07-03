from fastcore.meta import delegates
from inspect import signature
import pytest

"""
replaces **kwargs in a signature with all of the variables from another
function or from the init of a super class
"""

# used as func and class decorator
# Q: what use when to is None?
# A: when used as a class decorator for a subclass

#    delegates()
# keep
# but


@pytest.fixture
def A():
    class A:
        def __init__(self, x=3, y=2, z=1): pass
    return A


@pytest.fixture()
def B(A):
    @delegates()
    class B(A):
        def __init__(self, w, **kwargs): pass
    return B

@pytest.fixture()
def B2():
    class A:
        # no default values
        def __init__(self, x, y, z): pass

    @delegates()
    class B(A):
        def __init__(self, w, **kwargs): pass

    return B

def test_A(A):
    # A has no kwargs in its init
    sig = signature(A.__init__)
    assert len(sig.parameters) == 4

def test_B(B):
    sig = signature(B.__init__)
    assert 'kwargs' not in sig.parameters
    assert 'x' in sig.parameters
    assert 'y' in sig.parameters
    assert 'z' in sig.parameters

def test_no_defaults(B2):
    # no defaults in subclass parameters
    sig = signature(B2.__init__)

    #kwargs is removed
    assert 'kwargs' not in sig.parameters

    # but nothing added
    assert 'x' not in sig.parameters
    assert 'y' not in sig.parameters
    assert 'z' not in sig.parameters

def test_as_func():
    """ only the params with default values get passed through """
    def a(a, b=2, c=1): pass
    @delegates(a)
    def b(d, **kwargs): pass

    sig = signature(b)
    assert 'kwargs' not in sig.parameters

    assert 'a' not in sig.parameters
    assert 'b' in sig.parameters
    assert 'c' in sig.parameters
    assert 'd' in sig.parameters
