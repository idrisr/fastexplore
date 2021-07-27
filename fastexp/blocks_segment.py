from fastai.data.all import *
from fastai.vision.all import *
import random
import numpy as np


path = untar_data(URLs.CAMVID_TINY)
def label_func(o): return path/'labels'/f'{o.stem}_P{o.suffix}'


def l1():
    return SegmentationDataLoaders.from_label_func(
            path, bs=8, fnames=get_image_files(path/"images"),
            label_func = label_func,
            codes = np.loadtxt(path/'codes.txt', dtype=str), valid_pct=0.2,
            seed=42)


def l2():
    codes = np.loadtxt(path/'codes.txt', dtype=str)
    block = DataBlock(blocks=(ImageBlock, MaskBlock(codes=codes)),
                       splitter=RandomSplitter(0.2, seed=42),
                       get_items=get_image_files,
                       get_y=label_func,)
    return block, block.dataloaders(path/"images", bs=8)


def l3():
    tfms = [[PILImage.create], [Transform(label_func), PILMask.create]]
    files = get_image_files(path/"images")
    splits = RandomSplitter(0.2, seed=42)(files)
    dsets = Datasets(files, tfms, splits=splits)
    dls = dsets.dataloaders(after_item=[ToTensor, AddMaskCodes, ],
            after_batch=[IntToFloatTensor], bs=8)
    return dsets, dls


@patch
def __repr__(self:DataLoaders):
    return f"after_item:\n\t{self.after_item}" +\
           f"\nbefore_batch:\n\t {self.before_batch}" +\
           f"\nafter_batch:\n\t {self.after_batch}\n"


if __name__ == '__main__':
    d1 = l1()
    block, d2 = l2()
    dsets, d3 = l3()

    for _ in range(80):
        j = random.randint(0, len(d1.train_ds)-1)
        assert d1.train_ds[j] == d2.train_ds[j] == d3.train_ds[j], "nope"
