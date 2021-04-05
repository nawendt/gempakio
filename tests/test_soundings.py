# Copyright (c) 2021 Nathan Wendt.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for decoding GEMPAK grid files."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gempakio import GempakSounding


@pytest.mark.parametrize('gem,gio,station', [
    ('top_sigw_hght_unmrg.csv', 'top_sigw_hght_unmrg.snd', 'TOP'),
    ('waml_sigw_pres_unmrg.csv', 'waml_sigw_pres_unmrg.snd', 'WAML')
])
def test_loading_unmerged_sigw_hght(gem, gio, station):
    """Test loading an unmerged sounding.

    PPBB and PPDD groups will be in height coordinates.
    """
    g = Path(__file__).parent / 'data' / f'{gio}'
    d = Path(__file__).parent / 'data' / f'{gem}'

    gso = GempakSounding(g).snxarray(station_id=f'{station}')[0]
    gpres = gso.pres.values
    gtemp = gso.temp.values.squeeze()
    gdwpt = gso.dwpt.values.squeeze()
    gdrct = gso.drct.values.squeeze()
    gsped = gso.sped.values.squeeze()
    ghght = gso.hght.values.squeeze()

    gempak = pd.read_csv(d, na_values=-9999)
    dpres = gempak.PRES.values
    dtemp = gempak.TEMP.values
    ddwpt = gempak.DWPT.values
    ddrct = gempak.DRCT.values
    dsped = gempak.SPED.values
    dhght = gempak.HGHT.values

    np.testing.assert_allclose(gpres, dpres, rtol=1e-10, atol=1e-2)
    np.testing.assert_allclose(gtemp, dtemp, rtol=1e-10, atol=1e-2)
    np.testing.assert_allclose(gdwpt, ddwpt, rtol=1e-10, atol=1e-2)
    np.testing.assert_allclose(gdrct, ddrct, rtol=1e-10, atol=1e-2)
    np.testing.assert_allclose(gsped, dsped, rtol=1e-10, atol=1e-2)
    np.testing.assert_allclose(ghght, dhght, rtol=1e-10, atol=1e-2)
