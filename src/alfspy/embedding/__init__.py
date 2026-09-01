"""Embedded light fields.

An ALFS integral averages whatever the shots carry. Give it colour and you get a novel view;
give it dense visual descriptors and you get a novel view whose every pixel is a
1280-dimensional feature vector -- an *embedded* light field, useful for retrieval, matching
and similarity search over terrain rather than for looking at.

The pieces:

* :class:`~alfspy.embedding.extractor.DinoV3Extractor` turns a frame into a patch-resolution
  feature grid.
* :func:`~alfspy.render.field.render_field_integral` integrates those grids into a field.
* :class:`~alfspy.embedding.reduce.FieldReducer` projects a field back down to something
  viewable.

Everything here needs the optional dependencies: ``pip install "AlfsPy[embedding]"``. They
are imported lazily, so importing this package without them costs nothing until you use it.
"""

from typing import TYPE_CHECKING

__all__ = [
    'DEFAULT_MODEL',
    'DinoV3Extractor',
    'FieldReducer',
    'reduce_to_2d',
    'reduce_to_rgb',
]

if TYPE_CHECKING:  # pragma: no cover
    from .extractor import DEFAULT_MODEL, DinoV3Extractor
    from .reduce import FieldReducer, reduce_to_2d, reduce_to_rgb

_LAZY = {
    'DEFAULT_MODEL': '.extractor',
    'DinoV3Extractor': '.extractor',
    'FieldReducer': '.reduce',
    'reduce_to_rgb': '.reduce',
    'reduce_to_2d': '.reduce',
}


def __getattr__(name):
    """
    Resolves the public names on first access.

    Keeps ``import alfspy.embedding`` free of torch, transformers and scikit-learn, so the
    presence of this package does not force those on someone who only renders RGB.

    :param name: The attribute being looked up.
    :return: The requested object.
    :raises AttributeError: If the name is not part of the public API.
    """
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

    import importlib

    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
