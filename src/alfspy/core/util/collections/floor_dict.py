import bisect
from numbers import Number
from operator import index
from typing import Mapping, overload, Iterable, SupportsInt, SupportsFloat, SupportsIndex

from typing_extensions import TypeVar, Union

_KT_IDX = TypeVar('_KT_IDX', bound=SupportsIndex)
_KT_INT = TypeVar('_KT_INT', bound=SupportsInt)
_KT_FLOAT = TypeVar('_KT_FLOAT', bound=SupportsFloat)
_KT = TypeVar('_KT', bound=Union[_KT_INT, _KT_FLOAT])
_VT = TypeVar('_VT')

class FloorDict(dict[_KT, _VT]):
    """
    A variation of the built-in dictionary which overrides the get item method to floor round keys to the closest key
    present in the dictionary.
    """

    @overload
    def __init__(self, **kwargs: _VT) -> None: ...

    @overload
    def __init__(self, __map: Mapping[_KT, _VT]): ...

    @overload
    def __init__(self, __map: Mapping[_KT, _VT], **kwargs: _VT) -> None: ...

    @overload
    def __init__(self, __iterable: Iterable[tuple[_KT, _VT]]) -> None: ...

    @overload
    def __init__(self, __iterable: Iterable[tuple[_KT, _VT]], **kwargs: _VT) -> None: ...

    @overload
    def __init__(self, __iterable: Iterable[list[str]]) -> None: ...

    @overload
    def __init__(self, __iterable: Iterable[list[bytes]]) -> None: ...

    def __init__(self, seq=None, **kwargs) -> None:
        if seq is None:
            super().__init__(**kwargs)
        else:
            super().__init__(seq, **kwargs)

        self._sorted_keys = []

        if len(self) > 0:
            for key in self:
                bisect.insort(self._sorted_keys, self._numeric(key))

    @staticmethod
    def _numeric(value: _KT) -> Union[int, float]:
        if isinstance(value, SupportsFloat):
            return float(value)
        elif isinstance(value, SupportsInt):
            return int(value)
        else:
            return index(value)

    def __setitem__(self, key: _KT, value: _VT) -> None:
        if key not in self:
            bisect.insort(self._sorted_keys, self._numeric(key))
        super().__setitem__(key, value)

    def __getitem__(self, key: _KT) -> _VT:
        if key in self:
            return super().__getitem__(key)

        key = self._numeric(key)
        idx = bisect.bisect_right(self._sorted_keys, key) - 1
        if idx >= 0:
            closest_key = self._sorted_keys[idx]
            return super().__getitem__(closest_key)

        raise KeyError(f"Not suitable key found for {key}.")

    def __delitem__(self, key: _KT) -> None:
        if key in self:
            super().__delitem__(key)
            key = self._numeric(key)
            del self._sorted_keys[bisect.bisect_left(self._sorted_keys, key)]
        else:
            raise KeyError(key)


if __name__ == '__main__':
    def main() -> None:
        fd = FloorDict()
        for i in range(10):
            fd[i] = chr(65 + i)

        print(fd.get(1, None))
    main()
