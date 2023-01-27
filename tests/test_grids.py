# Copyright (c) 2021 Nathan Wendt.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for decoding GEMPAK grid files."""


from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from gempakio import GempakGrid


@pytest.mark.parametrize('grid_name', ['none', 'diff', 'dec', 'grib'])
def test_grid_loading(grid_name):
    """Test reading grids with various packing."""
    g = Path(__file__).parent / 'data' / f'{grid_name}.grd'
    d = Path(__file__).parent / 'data' / f'{grid_name}.npz'

    grid = GempakGrid(g).gdxarray(parameter='TMPK', level=850)[0]
    gio = grid.values.squeeze()

    gempak = np.load(d)['values']

    np.testing.assert_allclose(gio, gempak, rtol=1e-6, atol=0)


@pytest.mark.parametrize('keyword,date_time', [
    ('FIRST', '201204141200'), ('LAST', '201204150000')
])
def test_time_keywords(keyword, date_time):
    """Test time keywords FIRST and LAST."""
    g = Path(__file__).parent / 'data' / 'multi_date.grd'

    grid = GempakGrid(g).gdxarray(date_time=keyword)[0]
    dt64 = grid.time.values[0]
    epoch_seconds = int(dt64) / 1e9
    grid_dt = datetime(1970, 1, 1) + timedelta(seconds=epoch_seconds)
    expected = datetime.strptime(date_time, '%Y%m%d%H%M')

    assert grid_dt == expected


def test_multi_time_grid():
    """Test files with multiple times on a single grid."""
    g = Path(__file__).parent / 'data' / 'multi_time.grd'

    grid = GempakGrid(g)
    grid_info = grid.gdinfo()[0]
    dattim1 = grid_info.DATTIM1
    dattim2 = grid_info.DATTIM2

    assert dattim1 == datetime(1991, 8, 19, 0, 0)
    assert dattim2 == datetime(1991, 8, 20, 0, 0)
