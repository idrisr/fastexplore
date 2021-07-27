from fastai.vision.data import PointScaler
from fastai.data.core import Datasets
from fastai.data.all import *
from fastai.vision.all import *
from pathlib import Path
import numpy as np

__all__ = ['make_dataset', 'make_tfmdlist']

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
