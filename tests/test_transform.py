import os
import pytest
import tempfile
import torch
import math
import numpy as np

from matplotlib.artist import Artist
from fastcore.transform import Transform
from fastcore.dispatch import TypeDispatch
from fastai.vision.all import *
from fastai.data.transforms import ToTensor
from pathlib import Path
from torch import tensor
from torch import Tensor
from types import MethodType, FunctionType

from hypothesis.strategies import *
from hypothesis import given
import typing

tokill = [_ for _ in globals().keys() if _.startswith('test')]
for _ in tokill: del globals()[_]

#  https://docs.fast.ai/tutorial.pets.html

class ImageTransform(Transform):
    def encodes(self, fn: Path): return PILImage.create(fn)

@pytest.fixture
def test_fn(): return Path(__file__).parent.parent / "img/8533.png"

@pytest.fixture
def A():
    class A(Transform): 
        def encodes(self, a): return a+1
    return A

@pytest.fixture(scope="module")
def A1():
    class A(Transform): 
        def encodes(self, a:int): return a+1
    return A

@pytest.fixture
def A2():
    class A(Transform):
        def encodes(self, a:int): return a+1
        def encodes(self, a:list): a.append('z'); return a
    return A

def test_simple(A):
    """ no type annotations """
    a = A()
    assert a(1) == 2
    assert a(1.0) == 2.0

    # cant add an int with a list
    with pytest.raises(TypeError): a([])

def test_annotate(A1):
    a = A1()
    assert a(1) == 2

    # with a type annotation, the float is a noop
    assert a(1.0) == 1.0
    assert a([1, 2, 3]) == [1, 2, 3]
    assert a(None) == None

def test_more_annotations(A2):
    """ int and list """
    a = A2()
    assert a.encodes('1') == a('1')
    assert a.encodes(1) == 2
    assert a.encodes(['a']) == ['a', 'z']

def test_filename_to_image(test_fn):
    """
    start with a filename
    transform to an image
    pretty common type transform
    """
    timg = ImageTransform()(test_fn)
    assert isinstance(timg, PILImage)
    # int returns a noop 
    assert not isinstance(ImageTransform(2), int)

    # the output from the transform can be shown
    assert hasattr(timg, 'show')
    assert hasattr(ImageTransform, 'order')

def test_to_tensor(test_fn):
    """ go from filename, to PILImage, to somethign PyTorch can dig """

    timg = ImageTransform()(test_fn)

    # noop
    assert ToTensor().encodes([1, 2, 3]) == [1, 2, 3]
    t = ToTensor()(timg)
    assert t.shape == torch.Size([3, 28, 28])
    assert isinstance(t, Tensor)
    assert t.dtype == torch.uint8

    # transform can be ordered so it can go in a pipeline
    assert hasattr(ToTensor, 'order')
    assert hasattr(ImageTransform, 'order')

def test_can_show(test_fn):
    """ 
    I think all the show methods return a matplotlib axes / subaxes 
    by the way there are a lot of non-Transform classes that also have a
    show method 
    """
    timg = ImageTransform()(test_fn)
    t = ToTensor()(timg)

    # the output from the transform can be shown
    assert isinstance(timg.show(), Artist)
    assert isinstance(t.show(), Artist)

    assert isinstance(ToTensor.encodes, TypeDispatch)
    assert isinstance(ToTensor.decodes, TypeDispatch)
    assert isinstance(ToTensor.setups, TypeDispatch)


@pytest.mark.parametrize("input", [1, 'a', [], (1, 2, 3,), {'a':1, 'b':2}])
def test_bboxlabeler(input):
    """ 
    this thing doesn't have an encodes
    what's the logic behind that?  and it has a decode attribute...?
    """
    assert BBoxLabeler()(input) == input
    assert BBoxLabeler().encodes(input) == input


@pytest.mark.parametrize("type_", [BBoxLabeler, PointScaler, Transform])
def test_transform_decode(type_):
    """ 
    Q: What is with this decode function that comes from Transform?  It's not a
    TypeDispatch
    A: it's what you call to decode. decodes is what you use to implement. The
    difference is that you need a TypeDispatch in between?
    """ 
    b = type_
    assert isinstance(b.encodes, TypeDispatch)
    assert isinstance(b.decodes, TypeDispatch)
    assert isinstance(b.setups, TypeDispatch)
    assert not isinstance(b.decode, TypeDispatch)
    assert hasattr(b, 'decode')
    assert isinstance(b.decode, FunctionType)


@given(floats(), integers())
def test_tuples(A1, f, i):
    # only passing an int by itself passes. crazy
    assert A1.encodes((i)) == (i+1)
    assert not A1.encodes((i, f)) == (i+1, f)


@given(integers(), integers(), floats(allow_nan=False))
def test_tuples4(i1, i2, f):
    # you only pass in one thing. but that one thing can be a tuple.
    # if the element's type matches, youll get encoded.
    # otherwise you're just a noop. whoa, kind of mind blowing
    class A(Transform):
        def encodes(self, x: int): return x+1

    a = A()
    assert a(i1)          == i1+1
    assert a(f)           == f 
    assert a((i1))        == i1+1
    assert a((i1, i2))    == (i1+1, i2+1)
    assert a((i1, i2, f)) == (i1+1, i2+1, f)


@given(floats(), floats())
def test_tuples2(A1, f1, f2):
    # pass in two floats, both come back 
    assert A1.encodes((f1, f2)) == (f1, f2)

@given(floats(), floats())
def test_tuples3(A1, f1, f2):
    # pass in two floats to Resize. The googles do nothing
    Resize.goggles = Resize.encodes
    assert Resize(224).goggles((f1, f2)) == (f1, f2)

@given(floats(), floats())
def test_bbox(A1, f1, f2):
    # pass in two floats to Resize. The googles do nothing
    Resize.goggles = Resize.encodes
    assert Resize(224).goggles((f1, f2)) == (f1, f2)
