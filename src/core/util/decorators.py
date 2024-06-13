import warnings
import functools
from typing import Callable, Optional


def incomplete(reason: Optional[str] = None, stack_level: int = 2) -> Callable:
    """
    Decorator marking a function or method as incomplete or in development in terms of implementation
    :param reason: The reason on why the wrapped function is incomplete (optional)
    :param stack_level: The amount of levels to be shown in the stack trace of the warning (defaults to 2)
    :return: The wrapped function
    """
    def incomplete_dec(func: Callable):
        @functools.wraps(func)
        def new_func(*args, **kwargs):
            warning = f'Call to incomplete function {func.__name__}'
            if reason is not None:
                warning = f'{warning}: {reason}'
            warnings.warn(warning, category=RuntimeWarning, stacklevel=stack_level)
            return func(*args, **kwargs)
        return new_func
    return incomplete_dec