# Copyright (c) 2023 Nathan Wendt.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for encoding GEMPAK grid files."""

from pathlib import Path
import tempfile

import numpy as np
import pyproj

from gempakio import GempakGrid, GridFile


def test_grid_write():
    """Test writing grid."""
    proj = pyproj.Proj('+proj=lcc +lon_0=-95.0 +lat_1=25.0 '
                       '+lat_2=25.0 +ellps=sphere +R=6371200.0')

    grid = Path(__file__).parent / 'data' / 'surface_temp.npz'

    with np.load(grid) as dat:
        tmpc = dat['tmpc']
        lat = dat['lat']
        lon = dat['lon']

    out_grid = GridFile(lon, lat, proj)
    out_grid.add_grid(tmpc, 'tmpc', None, 0, '202211130200')

    kwargs = {'dir': '.', 'suffix': '.gem', 'delete': False}
    try:
        with tempfile.NamedTemporaryFile(**kwargs) as tmp:
            out_grid.to_gempak(tmp.name)
            gem = Path(tmp.name)

        in_grid = GempakGrid(gem)
        test_tmpc = in_grid.gdxarray(parameter='tmpc', date_time='202211130200',
                                     coordinate=None, level=0)[0].squeeze()
        test_lat = test_tmpc.lat
        test_lon = test_tmpc.lon

        np.testing.assert_allclose(test_tmpc, tmpc, rtol=1e-6, atol=0)
        np.testing.assert_allclose(test_lat, lat, rtol=1e-6, atol=0)
        np.testing.assert_allclose(test_lon, lon, rtol=1e-6, atol=0)
    finally:
        gem.unlink()
