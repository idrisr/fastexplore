from fastai.data.core import FilteredBase
from fastai.data.core import TfmdDL
from fastai.data.core import DataLoader, DataLoaders
from types import MethodType
import pytest

"""
Base class for lists with subsets

Q:
what does that mean, exactly?

A:
you have a list, and then split the indices into subsets?  I think so.

"""

"""
FilteredBase : class
   +__init__ : function
   +_new : function
   +dataloaders : function
   +n_subsets : function
   +subset : function
"""

@pytest.fixture
def fb(): return FilteredBase()


def test_some_attrs(fb):
    assert fb._dl_type == TfmdDL
    assert fb._dbunch_type == DataLoaders

    # dataloaders here is a method. Why not just put the
    # @delegates decorator on it?
    assert isinstance(fb.dataloaders, MethodType)
    with pytest.raises(AttributeError, match='splits'):
        assert isinstance(fb.dataloaders(), DataLoaders)


def test_attr_errors(fb):
    with pytest.raises(AttributeError): fb.n_subsets
    with pytest.raises(Exception): fb.subset()

def test_splits(fb):
    """
    this wont work because subset is not implmented
    in a way this should be an abstract base class abc
    """

    fb.splits = [[0, 1, 2], [3, 4, 5]]
    with pytest.raises(TypeError, match="takes 1 positional argument"):
        isinstance(fb.dataloaders(), DataLoaders)
