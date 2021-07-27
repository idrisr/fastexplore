from fastai.vision.all import *
from hypothesis.strategies import *
from random import choice

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

class BBoxTruth:
    """ get bounding box location from DataFrame """
    def __init__(self, df): self.df=df
    def __call__(self, o):
        size,x,y,*_ =self.df.iloc[int(o.stem)-1]
        return [[x,y, x+size, y+size]]

tokill = [_ for _ in globals().keys() if _.startswith('test')]
for _ in tokill: del globals()[_]

data_url = Path.home()/".fastai/data/chess"
df = pd.read_csv(data_url/'annotations.csv', index_col=0)
files = get_image_files(data_url)
bbt = BBoxTruth(df)

mpl.use('pdf')
def test_tensor_image():
    fn = choice(files)
    img = PILImage.create(fn)
    t = TensorImage(img)
    x0,y0,x1,y1=[i.item() for i in TensorBBox(bbt(fn)).squeeze(0)]

    show_image(t)
    plt.savefig("t")

    show_image(t[y0:y1, x0:x1, :])
    plt.savefig("t1")
