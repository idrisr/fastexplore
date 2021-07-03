from fastcore.dispatch import lenient_issubclass
from typing import (Collection, Sequence, Container, ValuesView, KeysView,
        MutableSequence, ItemsView)
import pytest

# lenient_issubclass(cls, types)

@pytest.mark.parametrize("cls,types,expected", 
            [
                [list, Collection,      True],
                [list, Sequence,        True],
                [list, MutableSequence, True],
                [list, Container,       True],
                [list, ValuesView,      False],
                [list, KeysView,        False],
                [list, ItemsView,       False],
            ])
def test_issubclass(cls, types, expected):
    """ is cls a subset of types """
    assert issubclass(cls, types) is expected
    assert lenient_issubclass(cls, types) is expected

@pytest.mark.parametrize("cls,types,expected", 
            [
                [list, Collection,      False],
                [list, Sequence,        False],
                [list, MutableSequence, False],
                [list, Container,       False],
                [list, ValuesView,      False],
                [list, KeysView,        False],
                [list, ItemsView,       False],
            ])
def test_isinstance(cls, types, expected):
    assert isinstance(cls, types) is expected

@pytest.mark.parametrize("cls,types,expected", 
            [
                [[],          Collection,      True],
                [(),          Sequence,        True],
                [[],          MutableSequence, True],
                [(),          MutableSequence, False],
                [[],          Container,       True],
                [{},          ValuesView,      False],
                [{}.values(), ValuesView,      True],
                [{}.keys(),   KeysView,        True],
                [{}.items(),  ItemsView,       True],
            ])
def test_isinstance2(cls, types, expected):
    """ basically lenient_subclass checks 
    for either
    isubclass or
    isinstance

    that's it
    """

    assert isinstance(cls, types) is expected

    with pytest.raises(TypeError):
        assert issubclass(cls, types) is not expected
    assert lenient_issubclass(cls, types) is expected
