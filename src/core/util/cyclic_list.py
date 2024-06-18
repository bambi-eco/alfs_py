from typing import SupportsIndex, Any


class CyclicList(list):
    def __getitem__(self, index: SupportsIndex) -> Any:
        if isinstance(index, int):
            index %= len(self)
        return super().__getitem__(index)
