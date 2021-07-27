"""
This is all about drilling the different transforms.

Let's try to load data from examples that use DataBlock, but instead we'll use
the mid-level API which includes:

Transform
TfmdLists
DataLoader
DataBlock
DataSet

See the mid-level API chapter in the fastai book for more info.
"""

from fastai.data.all import *
from fastai.vision.all import *
from functools import partial
from random import choice


def is_cat(x): return x.name[0].isupper()
path = untar_data(URLs.PETS) / "images"


def l1():
    dls = ImageDataLoaders.from_path_func(
            path, 
            get_image_files(path),
            label_func=is_cat,
            valid_pct=0.2,
            seed=42,
            item_tfms=Resize(224)
            )
    return dls


def l2():
    seed = 42
    valid_pct=0.2
    cls = DataLoaders
    kwargs = {}

    dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                   splitter=RandomSplitter(valid_pct, seed=seed),
                   get_items = get_image_files,
                   get_y=is_cat,
                   item_tfms=Resize(224)
                   )

    return dblock, dblock.dataloaders(path)


def l3():
    # same as type_tfms for datablock
    tfms = [[PILImage.create], [Categorize(), Transform(is_cat)]]
    files = get_image_files(path)
    splits = RandomSplitter(0.2, seed=42)(files)
    dsets = Datasets(files, tfms, splits=splits)
    dls = dsets.dataloaders(after_item=[ToTensor, Resize(224), ],
            after_batch=[IntToFloatTensor])
    return dsets, dls


def repr(self):
    return f"after_item:\n\t{self.after_item}" +\
           f"\nbefore_batch:\n\t {self.before_batch}" +\
           f"\nafter_batch:\n\t {self.after_batch}\n"


if __name__ == '__main__':
    # create some dataloaders in different ways
    DataLoaders.__repr__ = repr
    d1 = l1()
    dblock, d2 = l2()
    dsets, d3 = l3()

    j = choice(range(len(d1.train_ds)))
    assert d1.train_ds[j] == d2.train_ds[j] == dsets.train[j], "nope"

    i = choice(range(len(d1.train_ds)))
    assert d1.train_ds[i] == d2.train_ds[i] == d3.train_ds[i], "nope"
