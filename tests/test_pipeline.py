from fastcore.transform import Pipeline
from fastcore.transform import Transform
from fastcore.dispatch  import lenient_issubclass
from types import MethodType
from pytest import fixture
from pytest import mark 
from pytest import raises

"""
Pipeline : class
   -__call__ : function
   -__dir__ : function
   -__getattr__ : function
   -__getitem__ : function
   +__init__ : function
   -__repr__ : function
   -__setstate__ : function
   +_is_showable : function
   +add : function
   +decode : function
   +setup : function
   +show : function
"""

@fixture
def pipeline(): return Pipeline()

@mark.parametrize("attr", ['__call__',
                    '__dir__',
                    '__getattr__',
                    '__getitem__',
                    '__init__',
                    '__repr__',
                    '__setstate__',
                    '_is_showable',
                    'add',
                    'decode',
                    'setup',
                    'show',])
def test_attrs(pipeline, attr):
    """ this doesnt blow up, so I can access them even with an empty init """
    assert hasattr(pipeline, attr)
    assert getattr(pipeline, attr)

def test_tfm_dec(pipeline):
    assert pipeline.fs == []

    @Transform
    def f(x): return x+42 
    pipeline.add(f)
    assert pipeline(10) == 52
    assert pipeline[0] == f

    assert pipeline.decode(10) == 10
    assert pipeline.decode([]) == []
    assert pipeline.decode("NOOP") == "NOOP"
    msg = """
    No decode in the Transform created from the decorator. hence decode
    is a noop
    """
    assert pipeline.decode(msg) == msg

    # f is not returning a thing that can show, so ...
    assert pipeline.show(1) is None


def test_tfm_class(pipeline):
    class A(Transform):
        def encodes(self, x): return x+1
        def decodes(self, x): return x-1

    class B(Transform):
        def encodes(self, x): return x*2
        def decodes(self, x): return x/2

    p = Pipeline([A(), B()])
    assert p       (100) == 202
    assert p.decode(202) == 100
    assert lenient_issubclass(p.show, MethodType)
    assert pipeline.show(1) is None

    with raises(TypeError, match='concatenate'):
        p([]) # cant handle a list

def test_tfm_class_type(pipeline):
    class A(Transform):
        def encodes(self, x:int): return x+1
        def decodes(self, x:int): return x-1

    class B(Transform):
        def encodes(self, x:int): return x*2
        def decodes(self, x:int): return x//2

    p = Pipeline([A(), B()])
    assert p       (100) == 202
    assert p.decode(202) == 100
    assert lenient_issubclass(p.show, MethodType)
    assert pipeline.show(1) is None

    # now we get a noop
    assert pipeline([]) == []

def test_tfm_class_show(pipeline):
    class Z: 
        def show(self, ctx=None, **kwargs): 
            pass

    class A(Transform):
        def encodes(self, x:int): return x+1
        def decodes(self, x:int): return Z()

    class B(Transform):
        def encodes(self, x:int): return x*2
        def decodes(self, x:int): return x//2
    p = Pipeline([A(), B()])

    # how does a context gets passed in?
    assert p.show(100) is None
