__version__ = '0.1.0'

from fastai.data.core import DataLoaders
from fastcore.basics import patch


@patch
def __repr__(self:DataLoaders):
    return f"after_item:\n\t{self.after_item}" +\
           f"\nbefore_batch:\n\t {self.before_batch}" +\
           f"\nafter_batch:\n\t {self.after_batch}\n"
