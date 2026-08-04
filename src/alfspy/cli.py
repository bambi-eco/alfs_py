import argparse
from typing import Final

DEFAULT_METHOD: Final[str] = 'render'

def render() -> None:
    pass


def focus() -> None:
    raise NotImplemented()


def shutter() -> None:
    raise NotImplemented()


def main() -> None:

    arg_parser = argparse.ArgumentParser()
    sub_parsers = arg_parser.add_subparsers(dest='method', help='method to be executed', required=False)

    render_sub_parser = sub_parsers.add_parser(
        name='render',
        help='render a single light field'
    )


    focus_sub_parser = sub_parsers.add_parser(
        name='focus',
        help='render an animation of light fields where the focus linearly changes from one value to another'
    )


    shutter_sub_parser = sub_parsers.add_parser(
        name='shutter',
        help='render an animation of light fields where the number of input images changes according to a function'
    )

    args = arg_parser.parse_args()
    method = args.method
    if method is None:
        method = DEFAULT_METHOD

    if method == 'render':
        render()
    elif method == 'focus':
        focus()
    elif method == 'shutter':
        shutter()


if __name__ == '__main__':
    main()