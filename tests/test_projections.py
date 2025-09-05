# Copyright (c) 2025 Nathan Wendt.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for decoding GEMPAK grid files."""

from pathlib import Path

import numpy as np
import pytest

from gempakio import GempakGrid


@pytest.mark.parametrize(
    'proj_type',
    ['lcc', 'ced', 'stereographic', 'aed', 'gnomonic', 'lea', 'mercator', 'orthographic'],
)
def test_coordinates_creation(proj_type):
    """Test projections and coordinates."""
    g = Path(__file__).parent / 'data' / f'{proj_type}.grd'
    d = Path(__file__).parent / 'data' / f'{proj_type}.npz'

    grid = GempakGrid(g)
    decode_lat = grid.lat
    decode_lon = grid.lon

    gempak = np.load(d)
    true_lat = gempak['lat']
    true_lon = gempak['lon']

    np.testing.assert_allclose(decode_lat, true_lat, rtol=1e-6, atol=1e-2)
    np.testing.assert_allclose(decode_lon, true_lon, rtol=1e-6, atol=1e-2)
