from fastcore.transform import gather_attrs
from fastcore.foundation import L
import pytest

# used in Datasets and Transform
"""
i think it goes through an L and finds all the attributes with that name, per
each element in the L
"""

class A(L):
    def __init(self):
        self.z = 'Z'
        self.y = 'Y'
        self.x = 'X'

class B:
    def __init__(self):
        self.a = A()
    def __getattr__(self, k): 
        return gather_attrs(self, k, 'a')

@pytest.mark.skip(reason="maybe come back after datasets")
def test_gather_attrs():
    b = B()
    assert b.x == 'X'
    assert b.y == 'Y'
    assert b.z == 'Z'
