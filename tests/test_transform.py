import os
import pytest
import tempfile

from matplotlib.artist import Artist
from fastcore.transform import Transform
from fastai.vision.core import PILImage
from fastai.data.transforms import ToTensor
from pathlib import Path
from torch import tensor
from torch import Tensor
import torch

#  https://docs.fast.ai/tutorial.pets.html

#  @pytest.fixture
#  def ImageTransform():
class ImageTransform(Transform):
    def encodes(self, fn: Path): return PILImage.create(fn)
    #  return ImageTransform

@pytest.fixture
def test_fn(): return Path(os.getcwd()) / "img/8533.png"

@pytest.fixture
def A():
    class A(Transform):
        def encodes(self, a): return a+1
    return A

@pytest.fixture
def A1():
    class A(Transform):
        def encodes(self, a:int): return a+1
    return A

@pytest.fixture
def A2():
    class A(Transform):
        def encodes(self, a:int): return a+1
        def encodes(self, a:list): return a.append('z')
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
    # encodes == __call__
    a.encodes('1') == a('1')
    a.encodes == a.__call__
    a.encodes(1) == 2
    a.encodes(['a']) == ['a', 'z']


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
    """ I think all the show methods return a matplotlib axes / subaxes """

    timg = ImageTransform()(test_fn)
    t = ToTensor()(timg)

    # the output from the transform can be shown
    assert isinstance(timg.show(), Artist)
    assert isinstance(t.show(), Artist)

#  @ToTensor
#  def encodes(self, o:PILBase): return o._tensor_cls(image2tensor(o))
#  @ToTensor
#  def encodes(self, o:PILMask): return o._tensor_cls(image2tensor(o)[0])
