import pytest
from fastcore.foundation import *
from fastai.data.all import *
from fastai.vision.all import *

mnist_fn = path = (untar_data(URLs.MNIST_TINY) / "train/3").ls()[0]

""" explorations of the point scaler """

def _pnt_lbl(x): return TensorPoint.create(pnts)
def _pnt_open(fn): return PILImage(PILImage.create(fn).resize((28,35)))

def make_dataset():
    return Datasets([mnist_fn], [_pnt_open, [_pnt_lbl]])

def make_tfmdlist():
    tls = None
    tfms = None
    items = [mnist_fn]
    kwargs = {}

    return L(tls if tls else [TfmdLists(items, t, **kwargs) for t in L(ifnone(tfms,[None]))])

pnts = np.array([[0,0], [0,35], [28,0], [28,35], [9, 17]])
pnt_tds = make_dataset()
pnt_tdl = TfmdDL(pnt_tds, bs=1, after_item=[PointScaler(), ToTensor()])

@pytest.fixture
def dataset():
    return Datasets([mnist_fn], [_pnt_open, [_pnt_lbl]])

@pytest.fixture
def tfmdlist():
    return make_tfmdlist()

def test_dataset_exists(dataset): 
    assert dataset is not None

def test_dataset_type_tfms(dataset): 
    # why is this type L?
    assert isinstance(dataset.tls, L)

def test_dataset_len(dataset):
    assert len(dataset) == 1

def test_dataset_len2(dataset):
    assert True
    #  assert len(dataset) == 2

def test_tfmdlists(tfmdlist):
    assert len(tfmdlist) == 1

def test_tfmdlists2(tfmdlist):
    assert len(tfmdlist[0]) == 1
