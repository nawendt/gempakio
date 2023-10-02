# Copyright (c) 2023 Nathan Wendt.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for encoding GEMPAK sounding files."""

from datetime import datetime
from pathlib import Path
import tempfile

import numpy as np

from gempakio import GempakSounding, SoundingFile


def test_sounding_write():
    """Test writing sounding."""
    station_info = {
        'station_id': 'OUN',
        'station_number': 72357,
        'elevation': 357,
        'state': 'OK',
        'country': 'US'
    }
    oun_lat = 35.21
    oun_lon = -97.45
    dt = datetime(1999, 5, 4, 0, 0)

    snd = Path(__file__).parent / 'data' / 'sounding.npz'

    with np.load(snd) as dat:
        pres = dat['pres']
        hght = dat['hght']
        temp = dat['temp']
        dwpt = dat['dwpt']
        drct = dat['drct']
        sped = dat['sped']

    out_snd = SoundingFile(['pres', 'hght', 'temp', 'dwpt', 'drct', 'sped'])
    out_snd.add_sounding({
        'pres': pres,
        'hght': hght,
        'temp': temp,
        'dwpt': dwpt,
        'drct': drct,
        'sped': sped
    }, oun_lat, oun_lon, dt, station_info)

    kwargs = {'dir': '.', 'suffix': '.gem', 'delete': False}
    try:
        with tempfile.NamedTemporaryFile(**kwargs) as tmp:
            out_snd.to_gempak(tmp.name)
            gem = Path(tmp.name)

        in_snd = GempakSounding(gem)
        oun_data = in_snd.snxarray('OUN', date_time='199905040000')[0]
        test_pres = oun_data.pres
        test_hght = oun_data.hght.squeeze()
        test_temp = oun_data.temp.squeeze()
        test_dwpt = oun_data.dwpt.squeeze()
        test_drct = oun_data.drct.squeeze()
        test_sped = oun_data.sped.squeeze()

        np.testing.assert_allclose(test_pres, pres, rtol=1e-6, atol=0)
        np.testing.assert_allclose(test_hght, hght, rtol=1e-6, atol=0)
        np.testing.assert_allclose(test_temp, temp, rtol=1e-6, atol=0)
        np.testing.assert_allclose(test_dwpt, dwpt, rtol=1e-6, atol=0)
        np.testing.assert_allclose(test_drct, drct, rtol=1e-6, atol=0)
        np.testing.assert_allclose(test_sped, sped, rtol=1e-6, atol=0)
    finally:
        gem.unlink()
