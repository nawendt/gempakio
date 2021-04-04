# Copyright (c) 2021 Nathan Wendt.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for decoding GEMPAK grid files."""

import numpy as np
from pathlib import Path
import pytest

from gempakio import GempakGrid


@pytest.mark.parametrize('grid_name', ['none', 'diff', 'dec', 'grib'])
def test_grid_loading(grid_name):
    """Test reading grids."""
    g = Path(__file__).parent / 'data' / f'{grid_name}.grd'
    d = Path(__file__).parent / 'data' / f'{grid_name}.npz'

    grid = GempakGrid(g).gdxarray(parameter='TMPK', level=850)[0]
    gio = grid.values.squeeze()

    gempak = np.load(d)['values']

    np.testing.assert_allclose(gio, gempak, rtol=1e-6, atol=0)
