from typing import SupportsIndex, TypeVar, Generic

_T = TypeVar('_T')

class CyclicList(list[_T], Generic[_T]):
    """
    A list with a cyclic get index; an out-of-bounds int index is handled via a modulo operation.
    Slice indices behave as implemented by the builtin list.
    """

    def __getitem__(self, index: SupportsIndex) -> _T:
        if isinstance(index, int):
            index %= len(self)
        return super().__getitem__(index)
