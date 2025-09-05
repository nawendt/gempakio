# Copyright (c) 2025 Nathan Wendt.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for encoding GEMPAK sounding files."""

from pathlib import Path
import tempfile

from gempakio import GempakSurface, SurfaceFile


def test_surface_write():
    """Test writing surface."""
    sfc = Path(__file__).parent / 'data' / 'oun_sfc.gem'

    station = {
        'station_id': 'OUN',
        'station_number': 723570,
        'elevation': 362,
        'state': 'OK',
        'country': 'US',
    }
    oun_lat = 35.22
    oun_lon = -97.47

    surface = GempakSurface(sfc)
    sfc_obj = surface.sfjson()

    variables = list(sfc_obj[0]['values'].keys())

    out_sfc = SurfaceFile(variables)

    for report in sfc_obj:
        out_sfc.add_station(
            {
                'skyc': report['values']['skyc'],
                'tmpf': report['values']['tmpf'],
                'wsym': report['values']['wsym'],
                'pmsl': report['values']['pmsl'],
                'dwpf': report['values']['dwpf'],
                'brbk': report['values']['brbk'],
            },
            oun_lat,
            oun_lon,
            report['properties']['date_time'],
            station,
        )

    kwargs = {'dir': '.', 'suffix': '.gem', 'delete': False}
    try:
        with tempfile.NamedTemporaryFile(**kwargs) as tmp:
            out_sfc.to_gempak(tmp.name)
            gem = Path(tmp.name)

        in_sfc = GempakSurface(gem)
        test_obj = in_sfc.sfjson()

        for ob1, ob2 in zip(sfc_obj, test_obj, strict=True):
            for prop in ob1['properties']:
                assert ob1['properties'][prop] == ob2['properties'][prop]
            for param in ob1['values']:
                assert ob1['values'][param] == ob2['values'][param]
    finally:
        gem.unlink()
