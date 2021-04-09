# Copyright (c) 2021 Nathan Wendt.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for decoding GEMPAK grid files."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gempakio import GempakSounding


def test_merged():
    """Test loading a merged sounding.

    These are most often from models.
    """
    g = Path(__file__).parent / 'data' / 'msn_hrrr_mrg.snd'
    d = Path(__file__).parent / 'data' / 'msn_hrrr_mrg.csv'

    gso = GempakSounding(g).snxarray(station_id='KMSN')[0]
    gpres = gso.pres.values
    gtemp = gso.tmpc.values.squeeze()
    gdwpt = gso.dwpc.values.squeeze()
    gdrct = gso.drct.values.squeeze()
    gsped = gso.sped.values.squeeze()
    ghght = gso.hght.values.squeeze()
    gomeg = gso.omeg.values.squeeze()
    gcwtr = gso.cwtr.values.squeeze()
    gdtcp = gso.dtcp.values.squeeze()
    gdtgp = gso.dtgp.values.squeeze()
    gdtsw = gso.dtsw.values.squeeze()
    gdtlw = gso.dtlw.values.squeeze()
    gcfrl = gso.cfrl.values.squeeze()
    gtkel = gso.tkel.values.squeeze()
    gimxr = gso.imxr.values.squeeze()
    gdtar = gso.dtar.values.squeeze()

    gempak = pd.read_csv(d, na_values=-9999)
    dpres = gempak.PRES.values
    dtemp = gempak.TMPC.values
    ddwpt = gempak.DWPC.values
    ddrct = gempak.DRCT.values
    dsped = gempak.SPED.values
    dhght = gempak.HGHT.values
    domeg = gempak.OMEG.values
    dcwtr = gempak.CWTR.values
    ddtcp = gempak.DTCP.values
    ddtgp = gempak.DTGP.values
    ddtsw = gempak.DTSW.values
    ddtlw = gempak.DTLW.values
    dcfrl = gempak.CFRL.values
    dtkel = gempak.TKEL.values
    dimxr = gempak.IMXR.values
    ddtar = gempak.DTAR.values

    np.testing.assert_allclose(gpres, dpres, rtol=1e-10, atol=1e-2)
    np.testing.assert_allclose(gtemp, dtemp, rtol=1e-10, atol=1e-2)
    np.testing.assert_allclose(gdwpt, ddwpt, rtol=1e-10, atol=1e-2)
    np.testing.assert_allclose(gdrct, ddrct, rtol=1e-10, atol=1e-2)
    np.testing.assert_allclose(gsped, dsped, rtol=1e-10, atol=1e-2)
    np.testing.assert_allclose(ghght, dhght, rtol=1e-10, atol=1e-2)
    np.testing.assert_allclose(gomeg, domeg, rtol=1e-10, atol=1e-2)
    np.testing.assert_allclose(gcwtr, dcwtr, rtol=1e-10, atol=1e-2)
    np.testing.assert_allclose(gdtcp, ddtcp, rtol=1e-10, atol=1e-2)
    np.testing.assert_allclose(gdtgp, ddtgp, rtol=1e-10, atol=1e-2)
    np.testing.assert_allclose(gdtsw, ddtsw, rtol=1e-10, atol=1e-2)
    np.testing.assert_allclose(gdtlw, ddtlw, rtol=1e-10, atol=1e-2)
    np.testing.assert_allclose(gcfrl, dcfrl, rtol=1e-10, atol=1e-2)
    np.testing.assert_allclose(gtkel, dtkel, rtol=1e-10, atol=1e-2)
    np.testing.assert_allclose(gimxr, dimxr, rtol=1e-10, atol=1e-2)
    np.testing.assert_allclose(gdtar, ddtar, rtol=1e-10, atol=1e-2)


@pytest.mark.parametrize('gem,gio,station', [
    ('top_sigw_hght_unmrg.csv', 'top_sigw_hght_unmrg.snd', 'TOP'),
    ('waml_sigw_pres_unmrg.csv', 'waml_sigw_pres_unmrg.snd', 'WAML')
])
def test_unmerged(gem, gio, station):
    """Test loading an unmerged sounding.

    Test PPBB and PPDD groups in height and pressure coordinates.
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
