import os
from typing import Union, Iterable


def delete_all(files: Union[str, Iterable[str]]) -> None:
    """
    Tries to delete all given files. Skips over files that cannot be deleted.
    :param files: The files to be deleted.
    """
    if isinstance(files, str):
        files = (files,)

    for file in files:
        try:
            os.remove(file)
        except (FileNotFoundError, OSError):
            pass
