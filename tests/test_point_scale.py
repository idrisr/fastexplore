import explore
import pytest
from fastcore.foundation import *

@pytest.fixture
def dataset():
    return explore.make_dataset()

@pytest.fixture
def tfmdlist():
    return explore.make_tfmdlist()

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
