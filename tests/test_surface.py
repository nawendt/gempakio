# Copyright (c) 2021 Nathan Wendt.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for decoding GEMPAK surface files."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gempakio import GempakSurface


def test_standard_surface():
    """Test to read a standard surface file."""
    def dtparse(string):
        return datetime.strptime(string, '%y%m%d/%H%M')

    skip = ['text']

    g = Path(__file__).parent / 'data' / 'lwc_std_sfc.sfc'
    d = Path(__file__).parent / 'data' / 'lwc_std_sfc.csv'

    gsf = GempakSurface(g)
    gstns = gsf.sfjson()

    gempak = pd.read_csv(d, index_col=['STN', 'YYMMDD/HHMM'],
                         parse_dates=['YYMMDD/HHMM'],
                         date_parser=dtparse)
    if not gempak.index.is_monotonic_increasing:
        gempak.sort_index(inplace=True)

    for stn in gstns:
        idx_key = (stn['properties']['station_id'],
                   stn['properties']['date_time'])
        gemsfc = gempak.loc[idx_key, :]

        for param, val in stn['values'].items():
            if param not in skip:
                assert val == pytest.approx(gemsfc[param.upper()])


def test_ship_surface():
    """Test to read a ship surface file."""
    def dtparse(string):
        return datetime.strptime(string, '%y%m%d/%H%M')

    skip = ['text']

    g = Path(__file__).parent / 'data' / 'ship_sfc.sfc'
    d = Path(__file__).parent / 'data' / 'ship_sfc.csv'

    gsf = GempakSurface(g)

    gempak = pd.read_csv(d, index_col=['STN', 'YYMMDD/HHMM'],
                         parse_dates=['YYMMDD/HHMM'],
                         date_parser=dtparse)
    if not gempak.index.is_monotonic_increasing:
        gempak.sort_index(inplace=True)

    uidx = gempak.index.unique()

    for stn, dt in uidx:
        ugem = gempak.loc[(stn, dt), ]
        gstns = gsf.sfjson(station_id=stn, date_time=dt)

        assert len(ugem) == len(gstns)

        params = gempak.columns
        for param in params:
            if param not in skip:
                decoded_vals = [d['values'][param.lower()] for d in gstns]
                actual_vals = ugem.loc[:, param].values
                np.testing.assert_allclose(decoded_vals, actual_vals)
