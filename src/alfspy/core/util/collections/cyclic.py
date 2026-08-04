from typing import SupportsIndex, TypeVar, Generic, Sequence

_T = TypeVar('_T')

class CyclicSequence(Generic[_T]):
    """
    Add a cyclic get index to any sequence; an out-of-bounds int index is handled via a modulo operation.
    This only applies when an integer is used for indexing.
    """
    def __init__(self, sequence: Sequence[_T]) -> None:
        self._sequence = sequence

    def __getitem__(self, index: SupportsIndex) -> _T:
        if isinstance(index, int):
            index %= len(self._sequence)
        return self._sequence[index]


class CyclicList(list[_T], CyclicSequence[_T]):
    """
    A list with a cyclic get index; an out-of-bounds int index is handled via a modulo operation.
    This only applies when an integer is used for indexing.
    Slice indices behave as implemented by the builtin list.
    """
    def __init__(self, *args , **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: SupportsIndex) -> _T:
        if isinstance(index, int):
            index %= len(self)
        return super().__getitem__(index)
