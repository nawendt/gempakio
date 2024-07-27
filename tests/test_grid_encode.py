# Copyright (c) 2023 Nathan Wendt.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for encoding GEMPAK grid files."""

from pathlib import Path
import tempfile

import numpy as np
import pyproj
import pytest

from gempakio import GempakGrid, GridFile


def test_grid_type_mismatch():
    """Test for grid type mismatches."""
    proj = pyproj.Proj('+proj=lcc +lon_0=-95.0 +lat_1=25.0 '
                       '+lat_2=25.0 +ellps=sphere +R=6371200.0')

    grid = Path(__file__).parent / 'data' / 'surface_temp.npz'

    with np.load(grid) as dat:
        tmpc = dat['tmpc']
        lat = dat['lat']
        lon = dat['lon']
    out_grid = GridFile(lon, lat, proj)

    with pytest.raises(ValueError):
        out_grid.add_grid(tmpc, 'tmpc', None, 0, '202211130200', 0, '202211130000F002')


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

        assert test_tmpc.grid_type == 'analysis'
        np.testing.assert_allclose(test_tmpc, tmpc, rtol=1e-6, atol=0)
        np.testing.assert_allclose(test_lat, lat, rtol=1e-6, atol=0)
        np.testing.assert_allclose(test_lon, lon, rtol=1e-6, atol=0)
    finally:
        gem.unlink()


def test_grid_write_incorrect_datetime_format():
    """Test writing grid with incorrect datetime string input."""
    proj = pyproj.Proj('+proj=lcc +lon_0=-95.0 +lat_1=25.0 '
                       '+lat_2=25.0 +ellps=sphere +R=6371200.0')

    grid = Path(__file__).parent / 'data' / 'surface_temp.npz'

    with np.load(grid) as dat:
        tmpc = dat['tmpc']
        lat = dat['lat']
        lon = dat['lon']

    out_grid = GridFile(lon, lat, proj)

    with pytest.raises(ValueError):
        out_grid.add_grid(tmpc, 'tmpc', None, 0, '20221113')


def test_grid_write_incomplete_datetime():
    """Test writing grid with incomplete datetime string input."""
    proj = pyproj.Proj('+proj=lcc +lon_0=-95.0 +lat_1=25.0 '
                       '+lat_2=25.0 +ellps=sphere +R=6371200.0')

    grid = Path(__file__).parent / 'data' / 'surface_temp.npz'

    with np.load(grid) as dat:
        tmpc = dat['tmpc']
        lat = dat['lat']
        lon = dat['lon']

    out_grid = GridFile(lon, lat, proj)
    out_grid.add_grid(tmpc, 'tmpc', None, 0, '202211130000A')

    kwargs = {'dir': '.', 'suffix': '.gem', 'delete': False}
    try:
        with tempfile.NamedTemporaryFile(**kwargs) as tmp:
            out_grid.to_gempak(tmp.name)
            gem = Path(tmp.name)

        in_grid = GempakGrid(gem)
        test_tmpc = in_grid.gdxarray(parameter='tmpc', date_time='202211130000',
                                     coordinate=None, level=0)[0].squeeze()
        test_lat = test_tmpc.lat
        test_lon = test_tmpc.lon

        assert test_tmpc.grid_type == 'analysis'
        np.testing.assert_allclose(test_tmpc, tmpc, rtol=1e-6, atol=0)
        np.testing.assert_allclose(test_lat, lat, rtol=1e-6, atol=0)
        np.testing.assert_allclose(test_lon, lon, rtol=1e-6, atol=0)
    finally:
        gem.unlink()


def test_grid_write_incomplete_datetime_forecast():
    """Test writing grid with incomplete datetime string input on forecast grid."""
    proj = pyproj.Proj('+proj=lcc +lon_0=-95.0 +lat_1=25.0 '
                       '+lat_2=25.0 +ellps=sphere +R=6371200.0')

    grid = Path(__file__).parent / 'data' / 'surface_temp.npz'

    with np.load(grid) as dat:
        tmpc = dat['tmpc']
        lat = dat['lat']
        lon = dat['lon']

    out_grid = GridFile(lon, lat, proj)
    out_grid.add_grid(tmpc, 'tmpc', None, 0, '202211130000F')

    kwargs = {'dir': '.', 'suffix': '.gem', 'delete': False}
    try:
        with tempfile.NamedTemporaryFile(**kwargs) as tmp:
            out_grid.to_gempak(tmp.name)
            gem = Path(tmp.name)

        in_grid = GempakGrid(gem)
        test_tmpc = in_grid.gdxarray(parameter='tmpc', date_time='202211130000',
                                     coordinate=None, level=0)[0].squeeze()
        test_lat = test_tmpc.lat
        test_lon = test_tmpc.lon

        assert test_tmpc.grid_type == 'forecast'
        np.testing.assert_allclose(test_tmpc, tmpc, rtol=1e-6, atol=0)
        np.testing.assert_allclose(test_lat, lat, rtol=1e-6, atol=0)
        np.testing.assert_allclose(test_lon, lon, rtol=1e-6, atol=0)
    finally:
        gem.unlink()


def test_grid_write_minutes():
    """Test writing grid with forecast grid type with minutes."""
    proj = pyproj.Proj('+proj=lcc +lon_0=-95.0 +lat_1=25.0 '
                       '+lat_2=25.0 +ellps=sphere +R=6371200.0')

    grid = Path(__file__).parent / 'data' / 'surface_temp.npz'

    with np.load(grid) as dat:
        tmpc = dat['tmpc']
        lat = dat['lat']
        lon = dat['lon']

    out_grid = GridFile(lon, lat, proj)
    out_grid.add_grid(tmpc, 'tmpc', None, 0, '202211130000F00215')

    kwargs = {'dir': '.', 'suffix': '.gem', 'delete': False}
    try:
        with tempfile.NamedTemporaryFile(**kwargs) as tmp:
            out_grid.to_gempak(tmp.name)
            gem = Path(tmp.name)

        in_grid = GempakGrid(gem)
        test_tmpc = in_grid.gdxarray(parameter='tmpc', date_time='202211130215',
                                     coordinate=None, level=0)[0].squeeze()
        test_lat = test_tmpc.lat
        test_lon = test_tmpc.lon

        assert test_tmpc.grid_type == 'forecast'
        np.testing.assert_allclose(test_tmpc, tmpc, rtol=1e-6, atol=0)
        np.testing.assert_allclose(test_lat, lat, rtol=1e-6, atol=0)
        np.testing.assert_allclose(test_lon, lon, rtol=1e-6, atol=0)
    finally:
        gem.unlink()


def test_grid_write_multiple_grid_types():
    """Test writing grid with multiple grid types supplied."""
    proj = pyproj.Proj('+proj=lcc +lon_0=-95.0 +lat_1=25.0 '
                       '+lat_2=25.0 +ellps=sphere +R=6371200.0')

    grid = Path(__file__).parent / 'data' / 'surface_temp.npz'

    with np.load(grid) as dat:
        tmpc = dat['tmpc']
        lat = dat['lat']
        lon = dat['lon']

    out_grid = GridFile(lon, lat, proj)

    with pytest.raises(ValueError):
        out_grid.add_grid(tmpc, 'tmpc', None, 0, '202211130000AAA')


def test_grid_write_multiple_times_levels():
    """Test writing grid with forecast grid type and multiple times and levels."""
    proj = pyproj.Proj('+proj=lcc +lon_0=-95.0 +lat_1=25.0 '
                       '+lat_2=25.0 +ellps=sphere +R=6371200.0')

    grid = Path(__file__).parent / 'data' / 'surface_temp.npz'

    with np.load(grid) as dat:
        tmpc = dat['tmpc']
        lat = dat['lat']
        lon = dat['lon']

    out_grid = GridFile(lon, lat, proj)
    out_grid.add_grid(tmpc, 'tmpc', 'hght', 0, '202211130000F002', 10,
                      date_time2='202211130000F003')

    kwargs = {'dir': '.', 'suffix': '.gem', 'delete': False}
    try:
        with tempfile.NamedTemporaryFile(**kwargs) as tmp:
            out_grid.to_gempak(tmp.name)
            gem = Path(tmp.name)

        in_grid = GempakGrid(gem)
        test_tmpc = in_grid.gdxarray(parameter='tmpc', date_time='202211130200',
                                     coordinate='hght', level=0, date_time2='202211130300',
                                     level2=10)[0].squeeze()
        test_lat = test_tmpc.lat
        test_lon = test_tmpc.lon

        assert test_tmpc.grid_type == 'forecast'
        np.testing.assert_allclose(test_tmpc, tmpc, rtol=1e-6, atol=0)
        np.testing.assert_allclose(test_lat, lat, rtol=1e-6, atol=0)
        np.testing.assert_allclose(test_lon, lon, rtol=1e-6, atol=0)
    finally:
        gem.unlink()


def test_grid_write_multiple_times_minutes():
    """Test writing grid with forecast grid type and multiple times and levels."""
    proj = pyproj.Proj('+proj=lcc +lon_0=-95.0 +lat_1=25.0 '
                       '+lat_2=25.0 +ellps=sphere +R=6371200.0')

    grid = Path(__file__).parent / 'data' / 'surface_temp.npz'

    with np.load(grid) as dat:
        tmpc = dat['tmpc']
        lat = dat['lat']
        lon = dat['lon']

    out_grid = GridFile(lon, lat, proj)
    out_grid.add_grid(tmpc, 'tmpc', 'hght', 0, '202211130000F00215', 10,
                      date_time2='202211130000F00315')

    kwargs = {'dir': '.', 'suffix': '.gem', 'delete': False}
    try:
        with tempfile.NamedTemporaryFile(**kwargs) as tmp:
            out_grid.to_gempak(tmp.name)
            gem = Path(tmp.name)

        in_grid = GempakGrid(gem)
        test_tmpc = in_grid.gdxarray(parameter='tmpc', date_time='202211130215',
                                     coordinate='hght', level=0, date_time2='202211130315',
                                     level2=10)[0].squeeze()
        test_lat = test_tmpc.lat
        test_lon = test_tmpc.lon

        assert test_tmpc.grid_type == 'forecast'
        np.testing.assert_allclose(test_tmpc, tmpc, rtol=1e-6, atol=0)
        np.testing.assert_allclose(test_lat, lat, rtol=1e-6, atol=0)
        np.testing.assert_allclose(test_lon, lon, rtol=1e-6, atol=0)
    finally:
        gem.unlink()


def test_grid_write_analysis():
    """Test writing grid with analysis grid type."""
    proj = pyproj.Proj('+proj=lcc +lon_0=-95.0 +lat_1=25.0 '
                       '+lat_2=25.0 +ellps=sphere +R=6371200.0')

    grid = Path(__file__).parent / 'data' / 'surface_temp.npz'

    with np.load(grid) as dat:
        tmpc = dat['tmpc']
        lat = dat['lat']
        lon = dat['lon']

    out_grid = GridFile(lon, lat, proj)
    out_grid.add_grid(tmpc, 'tmpc', None, 0, '202211130000A2')

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

        assert test_tmpc.grid_type == 'analysis'
        np.testing.assert_allclose(test_tmpc, tmpc, rtol=1e-6, atol=0)
        np.testing.assert_allclose(test_lat, lat, rtol=1e-6, atol=0)
        np.testing.assert_allclose(test_lon, lon, rtol=1e-6, atol=0)
    finally:
        gem.unlink()


def test_grid_write_valid():
    """Test writing grid with valid grid type."""
    proj = pyproj.Proj('+proj=lcc +lon_0=-95.0 +lat_1=25.0 '
                       '+lat_2=25.0 +ellps=sphere +R=6371200.0')

    grid = Path(__file__).parent / 'data' / 'surface_temp.npz'

    with np.load(grid) as dat:
        tmpc = dat['tmpc']
        lat = dat['lat']
        lon = dat['lon']

    out_grid = GridFile(lon, lat, proj)
    out_grid.add_grid(tmpc, 'tmpc', None, 0, '202211130000V2')

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

        assert test_tmpc.grid_type == 'forecast'
        np.testing.assert_allclose(test_tmpc, tmpc, rtol=1e-6, atol=0)
        np.testing.assert_allclose(test_lat, lat, rtol=1e-6, atol=0)
        np.testing.assert_allclose(test_lon, lon, rtol=1e-6, atol=0)
    finally:
        gem.unlink()


def test_grid_write_forecast():
    """Test writing grid with forecast grid type."""
    proj = pyproj.Proj('+proj=lcc +lon_0=-95.0 +lat_1=25.0 '
                       '+lat_2=25.0 +ellps=sphere +R=6371200.0')

    grid = Path(__file__).parent / 'data' / 'surface_temp.npz'

    with np.load(grid) as dat:
        tmpc = dat['tmpc']
        lat = dat['lat']
        lon = dat['lon']

    out_grid = GridFile(lon, lat, proj)
    out_grid.add_grid(tmpc, 'tmpc', None, 0, '202211130000F002')

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

        assert test_tmpc.grid_type == 'forecast'
        np.testing.assert_allclose(test_tmpc, tmpc, rtol=1e-6, atol=0)
        np.testing.assert_allclose(test_lat, lat, rtol=1e-6, atol=0)
        np.testing.assert_allclose(test_lon, lon, rtol=1e-6, atol=0)
    finally:
        gem.unlink()


def test_grid_write_guess():
    """Test writing grid with guess grid type."""
    proj = pyproj.Proj('+proj=lcc +lon_0=-95.0 +lat_1=25.0 '
                       '+lat_2=25.0 +ellps=sphere +R=6371200.0')

    grid = Path(__file__).parent / 'data' / 'surface_temp.npz'

    with np.load(grid) as dat:
        tmpc = dat['tmpc']
        lat = dat['lat']
        lon = dat['lon']

    out_grid = GridFile(lon, lat, proj)
    out_grid.add_grid(tmpc, 'tmpc', None, 0, '202211130000G002')

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

        assert test_tmpc.grid_type == 'guess'
        np.testing.assert_allclose(test_tmpc, tmpc, rtol=1e-6, atol=0)
        np.testing.assert_allclose(test_lat, lat, rtol=1e-6, atol=0)
        np.testing.assert_allclose(test_lon, lon, rtol=1e-6, atol=0)
    finally:
        gem.unlink()


def test_grid_write_initial():
    """Test writing grid with initial grid type."""
    proj = pyproj.Proj('+proj=lcc +lon_0=-95.0 +lat_1=25.0 '
                       '+lat_2=25.0 +ellps=sphere +R=6371200.0')

    grid = Path(__file__).parent / 'data' / 'surface_temp.npz'

    with np.load(grid) as dat:
        tmpc = dat['tmpc']
        lat = dat['lat']
        lon = dat['lon']

    out_grid = GridFile(lon, lat, proj)
    out_grid.add_grid(tmpc, 'tmpc', None, 0, '202211130000I002')

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

        assert test_tmpc.grid_type == 'initial'
        np.testing.assert_allclose(test_tmpc, tmpc, rtol=1e-6, atol=0)
        np.testing.assert_allclose(test_lat, lat, rtol=1e-6, atol=0)
        np.testing.assert_allclose(test_lon, lon, rtol=1e-6, atol=0)
    finally:
        gem.unlink()


@pytest.mark.parametrize('proj_type', ['conical', 'cylindrical', 'azimuthal'])
def test_grid_write_projected_using_xy(proj_type):
    """Test projections and coordinates."""
    g = Path(__file__).parent / 'data' / f'{proj_type}.grd'
    d = Path(__file__).parent / 'data' / f'{proj_type}.npz'

    grid = GempakGrid(g)
    y = grid.y
    x = grid.x
    tmpk = grid.gdxarray()[0].squeeze().values
    proj = pyproj.Proj(grid.crs)

    out_grid = GridFile(x, y, proj, use_xy=True)
    out_grid.add_grid(tmpk, 'tmpk', 'pres', 850, '199108190000F024')

    kwargs = {'dir': '.', 'suffix': '.gem', 'delete': False}
    try:
        with tempfile.NamedTemporaryFile(**kwargs) as tmp:
            out_grid.to_gempak(tmp.name)
            gem = Path(tmp.name)

        in_grid = GempakGrid(gem)
        test_tmpk = in_grid.gdxarray(parameter='tmpk', date_time='199108200000',
                                     coordinate='pres', level=850)[0].squeeze()
        test_lat = test_tmpk.lat
        test_lon = test_tmpk.lon

        gempak = np.load(d)
        true_lat = gempak['lat']
        true_lon = gempak['lon']

        np.testing.assert_allclose(test_lat, true_lat, rtol=1e-3, atol=0)
        np.testing.assert_allclose(test_lon, true_lon, rtol=1e-3, atol=0)
    finally:
        gem.unlink()
