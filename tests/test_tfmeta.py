from debjig import log
from fastcore.transform import _TfmMeta
from fastai.vision.all import *
import logging
import sys
import pytest


logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger()
logger.setLevel(logging.WARNING)


#  _TfmMeta.__prepare__ = log()(_TfmMeta.__prepare__)


class MyMeta(type):
    def __new__(cls, name, bases, dict_):
        #  if name == 'B': breakpoint()

        if 'b' not in dict_: 
            pytest.fail("setup issue")
        if len(bases) != dict_['b']:
            pytest.fail("bases wrong") 
        return super().__new__(cls, name, bases, dict_)

class A(metaclass=MyMeta):
    b = 0
class B(metaclass=MyMeta):
    b = 0
class C(A, B):
    b = 2
class D(B):
    pass
    #  b = 2


def test_meta():
    assert A()
    assert B()
    assert C()
    assert D()


@pytest.mark.skip
def test_tfmmeta(caplog):
    with caplog.at_level(logging.DEBUG):
        A()
        assert "__call__" in caplog.text
        caplog.clear()

    with caplog.at_level(logging.DEBUG):
        A()
        assert "__new__" in caplog.text
        caplog.clear()
