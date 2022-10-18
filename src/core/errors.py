class MutuallyExclusiveArgsError(Exception):
    """
    Raised when multiple mutually exclusive arguments are passed to a callable
    """


class MissingArgError(Exception):
    """
    Raised when at least one required argument is not passed to a callable
    """
