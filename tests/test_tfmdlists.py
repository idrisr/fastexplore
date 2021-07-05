from fastai.data.core import TfmdLists
from fastcore.transform import Transform
from pytest import fixture
from pytest import mark

"""
A `Pipeline` of `tfms` applied to a collection of `items`
lazily applies a collection of Transforms on a list
"""

@fixture
def T():
    class T(Transform):
        def __init__(self, x): self.x = x
        def encodes(self, o): return self.x + o, 'a'
        # not a real inverse, but so what
        def decodes(self, o): return self.x - o, 'b'
    return T

@fixture
def U():
    class U(Transform):
        def __init__(self, x): self.x = x
        def encodes(self, o): return self.x + o, 'a'
        # not a real inverse, but so what
        def decodes(self, o): return self.x - o, 'b'
    return U

@fixture
def valid_tl(T): return TfmdLists(list(range(10, 20)), T(2))
@fixture
def train_tl(T): return TfmdLists(list(range(10)),     T(1))

#  dls = DataLoaders.from_dsets(
    #  train_tl, 
    #  valid_tl, 
    #  after_batch=[Normalize.from_stats(*imagenet_stats), *aug_transforms()])
#  dls = dls.cuda()

#  used to create a data loader
def test_tfm(T):
    t = T(9)
    assert t(10) == (19, 'a')
    assert t.decode(19) == (-10, 'b')

@mark.parametrize('i,exp', [
    [0, (1,  'a')],
    [1, (2,  'a')],
    [2, (3,  'a')],
    [3, (4,  'a')],
    [9, (10, 'a')]])
def test_train(train_tl, i, exp):
    assert isinstance(train_tl, TfmdLists)
    assert train_tl[i] == exp

@mark.parametrize('i,exp', [
    [0,(12, 'a')],
    [5,(17, 'a')],
    [9,(21, 'a')]
    ])
def test_valid(valid_tl, i, exp):
    assert isinstance(valid_tl, TfmdLists)
    assert valid_tl[i] == exp


@mark.parametrize('i,exp', [
    [0,(1, 'b')], 
    [5,(-4, 'b')],
    [9,(-8, 'b')]
    ])
def test_train_decode(i,exp,train_tl):
    assert train_tl.decode(i) == exp
