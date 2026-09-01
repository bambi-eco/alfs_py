"""Test suite for the PyTorch port of ALFSPy.

Layout:

* ``unit/``        -- pure functions, no rendering.
* ``raster/``      -- the torch rasteriser against analytically known answers. This is where
                      the ModernGL replacement is actually proved correct.
* ``integration/`` -- end-to-end pipelines on a generated synthetic flight.
* ``parity/``      -- opt-in comparison against a reference ModernGL install.
* ``bench/``       -- throughput measurements, excluded from the default run.
"""
