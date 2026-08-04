from typing import Protocol, TypeVar

_T1 = TypeVar('_T1')
_T2 = TypeVar('_T2')

class Indexable(Protocol):

    def __getitem__(self, key: _T1) -> _T2:
        ...