from fastcore.basics import GetAttr
import pytest

"""
Everything in _xtra will be proxied over to default
"""

@pytest.fixture
def A():
    class A:
        def __init__(self, y, z):
            self.y=y
            self.z=z
    return A

@pytest.fixture
def B():
    class B(GetAttr):
        _default = 'a'
        _xtra = 'y'
        def __init__(self, a):
            self.a = a
    return B

@pytest.fixture
def C():
    class C(GetAttr):
        _default='b'
        def __init__(self, b): self.b=b
    return C

def test_B(A, B):
    """ pass y through in _xtra """
    a = A(1, 2)
    b = B(a)
    assert b.y == 1
    with pytest.raises(AttributeError): b.z

def test_B(A, B):
    """ empty _xtra, therefore everything gets passed through """
    B._xtra=None
    a = A(1, 2)
    b = B(a)
    assert b.y == 1
    assert b.z == 2
    with pytest.raises(AttributeError): b.x

def test_xtra(A, B):
    """ extra _xtra """
    B._xtra = 'x'
    a = A(1, 2)
    b = B(a)
    with pytest.raises(AttributeError): b.x

def test_two_deep(A, B, C):
    """ this works as deep as you want to go """
    B._xtra=None
    a = A(2, 1)
    b = B(a)
    c = C(b)
    assert c.y==2
    assert c.z==1
    assert c.a==a
