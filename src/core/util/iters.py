from typing import Optional, Iterator, Generator


def file_name_gen(filetype: str, prefix: Optional[str] = None, suffix: Optional[str] = None) -> Iterator[str]:
    """
    Generates a generator generating file names based on the following pattern:
    ([prefix]_)[i](_[suffix])[filetype] where i represents a counter variable starting at 0 counting up
    :param filetype: The filetype of the file names to create. If the filetype does not start with a period symbol, it
    will be prepended automatically
    :param prefix: The prefix to the counter variable (optional). If set an underscore symbol will always be appended
    :param suffix: The suffix to the counter variable (optional). If set an underscore symbol will always be prepended
    :return: An endless iterator returning file names
    """
    r_filetype = filetype if filetype.startswith('.') else f'.{filetype}'
    r_prefix = f'{prefix}_' if prefix is not None else ''
    r_suffix = f'_{suffix}' if suffix is not None else ''

    def gen() -> Generator[str, None, None]:
        i = 0
        while True:
            yield f'{r_prefix}{i}{r_suffix}{r_filetype}'
            i += 1

    return gen()
