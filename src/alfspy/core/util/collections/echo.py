from typing import TypeVar, Union, Final

_T = TypeVar('_T')

class Echo:
    """
    An echo object that returns the passed item unchanged as a result.
    This goes for calling the object or when using an indexer.
    """

    def __call__(self, item: _T) -> _T:
        return item

    def __getitem__(self, item: _T) -> _T:
        return item

_K = TypeVar('_K')
_V = TypeVar('_V')

class EchoDict(dict[_K, _V]):
    """
    An echo dictionary that returns the passed key unchanged as a result if there is no value associated with it.
    """
    _NIL_TOKEN: Final[object] = object()

    def __getitem__(self, item: _K) -> Union[_K, _V]:
        value = self.get(item, self._NIL_TOKEN)
        if value is not self._NIL_TOKEN:
            return value
        return item
