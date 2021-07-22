import pytest
from fastai.data.all import *
from fastai.vision.all import *
from hypothesis.strategies import lists, integers, composite, floats
from hypothesis import given

# removed imported test symbols so pytest doesnt pick them up
tokill = [_ for _ in globals().keys() if _.startswith('test')]
for _ in tokill: del globals()[_]


"""
class PointScaler(Transform):
    "Scale a tensor representing points"
    order = 1
    def __init__(self, do_scale=True, y_first=False): self.do_scale,self.y_first = do_scale,y_first
    def _grab_sz(self, x):
        self.sz = [x.shape[-1], x.shape[-2]] if isinstance(x, Tensor) else x.size
        return x

    def _get_sz(self, x): return getattr(x, 'img_size') if self.sz is None else self.sz

    def setups(self, dl):
        res = first(dl.do_item(None), risinstance(TensorPoint))
        if res is not None: self.c = res.numel()

    def encodes(self, x:(PILBase,TensorImageBase)): return self._grab_sz(x)
    def decodes(self, x:(PILBase,TensorImageBase)): return self._grab_sz(x)

    def encodes(self, x:TensorPoint): return _scale_pnts(x, self._get_sz(x), self.do_scale, self.y_first)
    def decodes(self, x:TensorPoint): return _unscale_pnts(x.view(-1, 2), self._get_sz(x))

"""

mnist_fn = path = (untar_data(URLs.MNIST_TINY) / "train/3").ls()[0]

""" explorations of the point scaler """

pnts = np.array([[0,0], [0,35], [28,0], [28,35], [9, 17]])


@composite
def list_mult(draw, x=1):
    i = draw(integers(min_value=1, max_value=1000//x))
    xs = draw(lists(elements=floats(), min_size=i*x, max_size=i*x))
    return xs


@pytest.mark.skip
@given(list_mult(2))
def test_tensor_point(l): 
    """ tensor point needs an even number of inputs """
    TensorPoint.create(l)


@pytest.mark.skip
@given(list_mult(4))
def test_tensor_bbox(l):
    """ tensor bbox needs an quad number of inputs """
    TensorBBox.create(l)


@pytest.mark.skip
@given(list_mult(4))
def test_Resize(l):
    bbox = TensorBBox.create(l)
    Resize(224)(bbox)

@given(list_mult(4))
def test_point_scale(l):
    # how to get a az attribute on this thing??
    img  = PILImage.create(mnist_fn)
    bbox = TensorBBox.create(img, img_size=(100, 100, ))
    ps = PointScaler(224)

    # how to get a sz attribute on this thing??
    with pytest.raises(AttributeError):
        ps(bbox)

    # pass the image in the tuple first
    ps((img, bbox))
