from fastai.data.all import *
from fastai.text.all import *


# skipped for now


def l1():
    path = untar_data(URLs.IMDB)
    dls = TextDataLoaders.from_folder(path, valid='test')
    return dls


def l2():
    path = untar_data(URLs.IMDB)
    b1 = TextBlock.from_folder(path, text_vocab=None, is_lm=False, seq_len=72, backwards=False, tok=None)
    b2 = CategoryBlock(vocab=vocab)
    blocks = [TextBlock.from_folder(path, text_vocab=None, is_lm=False,
        seq_len=72, backwards=False, tok=None), ]
    block = DataBlock(blocks=blocks, 


@patch
def __repr__(self:DataLoaders):
    return f"after_item:\n\t{self.after_item}" +\
           f"\nbefore_batch:\n\t {self.before_batch}" +\
           f"\nafter_batch:\n\t {self.after_batch}\n"


if __name__ == '__main__':
    d1 = l1()
    #  block, d2 = l2()
    #  dsets, d3 = l3()

    #  for _ in range(80):
        #  j = random.randint(0, len(d1.train_ds)-1)
        #  assert d1.train_ds[j] == d2.train_ds[j] == d3.train_ds[j], "nope"
