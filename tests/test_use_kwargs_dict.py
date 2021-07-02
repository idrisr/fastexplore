import inspect
from fastcore.meta import use_kwargs_dict

""" 
use_kwargs_dict basically replaces kwargs in the signature, 
and then things like tab complete and function hints work
"""

def test_without():
    #  @use_kwargs_dict(b='yep')
    def f(a, **kwargs): pass

    sig = inspect.signature(f)
    assert 'kwargs' in sig.parameters
    assert 'b' not in sig.parameters

def test_with():
    @use_kwargs_dict(b='yep')
    def f(a, **kwargs): pass
    sig = inspect.signature(f)
    assert 'kwargs' not in sig.parameters
    assert 'b' in sig.parameters

def test_keep():
    """ keep adds more parameters and keeps kwargs """
    @use_kwargs_dict(keep=True, b='yep')
    def f(a, **kwargs): pass
    sig = inspect.signature(f)
    assert 'kwargs' in sig.parameters
    assert 'b' in sig.parameters
