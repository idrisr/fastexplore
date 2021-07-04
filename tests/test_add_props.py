from fastcore.basics import add_props
from inspect import signature
import pytest

""" 
basically you create a callable, and then add_props
will create n properties, where each one differs by having some argument from
the elements in range n.

you can use it to iterate over some list, create a callable from each element,
via partial and property, and then bind those values to attributes.
""" 
from functools import partial

# ok, maybe I don't understand fully how to create a property
# and or how to do it with a partial

# come back to this one

@pytest.fixture
def partial_property(f, n):
    pass


class D:
    def __get__(self, obj, type): 
        return 1

class A:
    d = D()
    e = property(lambda x: 1)

# kind of insane.
# the key is to realize that self will be passed in when envoked
# and that self goes second, since the first argument is what gets curried
class Z:
    def __init__(self, l): self.l = l
    def __getitem__(self, i): return self.l[i]
    a,b,c, = (property(partial(lambda i,self: self[i], _)) for _ in range(3))


def test_property():
    """ it's a property. you can't pass anything in """
    z = Z('david foster wallace'.split())
    assert z.a == 'david' 
    assert z.b == 'foster' 
    assert z.c == 'wallace' 
    with pytest.raises(AttributeError): z.a = 'lallace' 


def test_property2():
    z = Z([5, 8, 13])
    assert z.a == 5
    assert z.b == 8
    assert z.c == 13
    with pytest.raises(AttributeError): z.a = 'lallace' 


def test_T():
    class T: 
        a,b,c,d = add_props(lambda i,x:i*2, n=4)
    t = T()
    assert t.a == 0
    assert t.b == 2
    assert t.c == 4
    assert t.d == 6
    print(type(t.d))
