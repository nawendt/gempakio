# Copyright (c) 2025 Nathan Wendt.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for decoding GEMPAK grid files."""

from datetime import datetime, timedelta
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


@pytest.mark.parametrize('access_type', ['STID', 'STNM'])
def test_sounding_access(access_type):
    """Test for proper sounding retrieval with multi-parameter filter."""
    g = Path(__file__).parent / 'data' / 'merged_nopack.snd'
    gso = GempakSounding(g)

    if access_type == 'STID':
        gso.snxarray(station_id='OUN', country='US', state='OK', date_time='202101200000')
    elif access_type == 'STNM':
        gso.snxarray(station_number=72357, country='US', state='OK', date_time='202101200000')


@pytest.mark.parametrize('text_type', ['txta', 'txtb', 'txtc', 'txpb'])
def test_sounding_text(text_type):
    """Test for proper decoding of coded message text."""
    g = Path(__file__).parent / 'data' / 'unmerged_with_text.snd'
    d = Path(__file__).parent / 'data' / 'unmerged_with_text.csv'

    gso = GempakSounding(g).snxarray(station_id='OUN')[0]
    gempak = pd.read_csv(d)

    text = gso.attrs['wmo_codes'][text_type]
    gem_text = gempak.loc[:, text_type.upper()][0]

    assert text == gem_text


@pytest.mark.parametrize(
    'gem,gio,station',
    [
        ('top_sigw_hght_unmrg.csv', 'top_sigw_hght_unmrg.snd', 'TOP'),
        ('waml_sigw_pres_unmrg.csv', 'waml_sigw_pres_unmrg.snd', 'WAML'),
    ],
)
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


def test_unmerged_sigw_pressure_sounding():
    """Test loading an unmerged sounding.

    PPBB and PPDD groups will be in pressure coordinates and there will
    be MAN levels below the surface.
    """
    g = Path(__file__).parent / 'data' / 'dl10548_sigw_pres_unmrg_man_bgl.snd'
    d = Path(__file__).parent / 'data' / 'dl10548_sigw_pres_unmrg_man_bgl.csv'

    gso = GempakSounding(g).snxarray()
    gpres = gso[0].pres.values
    gtemp = gso[0].temp.values.squeeze()
    gdwpt = gso[0].dwpt.values.squeeze()
    gdrct = gso[0].drct.values.squeeze()
    gsped = gso[0].sped.values.squeeze()
    ghght = gso[0].hght.values.squeeze()

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
    np.testing.assert_allclose(ghght, dhght, rtol=1e-10, atol=1e-1)


def test_unmerged_no_ttcc():
    """Test loading an unmerged sounding.

    Sounding will have a PPCC group, but no TTCC group. This
    tests for proper handling of MAN winds on pressure surfaces
    without any temp/dewpoint/height data.
    """
    g = Path(__file__).parent / 'data' / 'nzwp_no_ttcc.snd'
    d = Path(__file__).parent / 'data' / 'nzwp_no_ttcc.csv'

    gso = GempakSounding(g).snxarray()
    gpres = gso[0].pres.values
    gtemp = gso[0].temp.values.squeeze()
    gdwpt = gso[0].dwpt.values.squeeze()
    gdrct = gso[0].drct.values.squeeze()
    gsped = gso[0].sped.values.squeeze()
    ghght = gso[0].hght.values.squeeze()

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
    np.testing.assert_allclose(ghght, dhght, rtol=1e-10, atol=1e-1)


@pytest.mark.parametrize(
    'keyword,date_time', [('FIRST', '202011070000'), ('LAST', '202011070100')]
)
def test_time_keywords(keyword, date_time):
    """Test time keywords FIRST and LAST."""
    g = Path(__file__).parent / 'data' / 'unmerged_with_text.snd'

    gso = GempakSounding(g).snxarray(date_time=keyword)[0]
    expected = datetime.strptime(date_time, '%Y%m%d%H%M')
    dt64 = gso.time.values[0]
    epoch_seconds = int(dt64) / 1e9
    sounding_dt = datetime(1970, 1, 1) + timedelta(seconds=epoch_seconds)

    assert sounding_dt == expected
