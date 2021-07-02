import explore
from fastcore.basics import store_attr


""" 
Basically store_attr will store a bunch of attributes onto an object.
It uses magic like going one up the stack frame, and then reading the variabels
in that frame looking for what to attach.
"""

def test_with_name():
    """ if the args isnt passed in, you need to be explicit """
    class A:
        def __init__(self):
            b = 42
            store_attr(names='b')
    assert A().b == 42

def test_default():
    """ the args passed in will be added """
    class A:
        def __init__(self, b=42, c=69):
            store_attr()
    a = A()
    assert a.b == 42
    assert a.c == 69

def test_but():
    """ the args passed in will be added """
    class A:
        def __init__(self, b=42, c=69):
            store_attr(but='c')
    a = A()
    assert a.b == 42
    assert not hasattr(A(), 'c')

def test_self():
    """ the self command is passed in when used outside 
    and you need to be explicit about the names,
    otherwise self gets passed in twice
    """
    class B:...
    class A:
        def __init__(self):
            z=1;y=2;x='asdf'
            b = B()
            store_attr('x,y,z', self=b)
            self.b = b
    a = A()
    assert a.b.z==1
    assert a.b.y==2
    assert a.b.x=='asdf'
