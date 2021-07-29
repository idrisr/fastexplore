from fastai.data.all import *
from fastai.vision.all import *
from fastcore.basics import patch

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use("pdf")

path = untar_data(URLs.COCO_TINY)
fnames, boxes = get_annotations(path/"train.json")

@patch
def __repr__(self:DataLoaders):
    return f"after_item:\n\t{self.after_item}" +\
           f"\nbefore_batch:\n\t {self.before_batch}" +\
           f"\nafter_batch:\n\t {self.after_batch}\n"

item_tfms = []
bbox_map = dict(zip(fnames, boxes))

def get_bb(x):
    return bbox_map[x.name][0]

def get_class_name(x):
    return bbox_map[x.name][1]


getters = [get_bb, get_class_name]

item_tfms = [FlipItem(), RandomResizedCrop(128, min_scale=0.35)]
batch_tfms = [
        #  FlipItem(),
              #  RandomResizedCrop(128, min_scale=0.35),
              IntToFloatTensor(),
              Normalize.from_stats(*imagenet_stats)]

db = DataBlock (blocks=(ImageBlock, BBoxBlock, BBoxLblBlock),
        splitter=RandomSplitter(seed=42),
        get_items=get_image_files,
        get_y=getters,
        item_tfms=item_tfms,
        batch_tfms=batch_tfms,
        n_inp=1)

dls = db.dataloaders(path, verbose=True)
dls.show_batch(max_n=25)
plt.savefig("batch")
