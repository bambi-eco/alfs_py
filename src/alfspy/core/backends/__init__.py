"""Render backends.

Each backend implements the same three GPU operations -- render the textured DEM, project a
shot onto it, and integrate several shots -- against a different API. Backends are imported
lazily, so a missing optional dependency only fails when that backend is actually requested.

The registry and the shared ``Engine`` protocol land here next; for now this package holds
the concrete implementations.
"""
