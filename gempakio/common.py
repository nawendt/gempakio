# Copyright (c) 2023 Nathan Wendt.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""GEMPAK comman data definitions and structures."""

from enum import Enum

ANLB_SIZE = 128
MBLKSZ = 128
MMFREE = 62
NAVB_SIZE = 256
GEMPAK_HEADER = 'GEMPAK DATA MANAGEMENT FILE '


class FileTypes(Enum):
    """GEMPAK file type."""

    surface = 1
    sounding = 2
    grid = 3


class DataTypes(Enum):
    """Data management library data types."""

    real = 1
    integer = 2
    character = 3
    realpack = 4
    grid = 5


class VerticalCoordinates(Enum):
    """Veritical coordinates."""

    none = 0
    pres = 1
    thta = 2
    hght = 3
    sgma = 4
    dpth = 5
    hybd = 6
    pvab = 7
    pvbl = 8


class PackingType(Enum):
    """GRIB packing type."""

    none = 0
    grib = 1
    nmc = 2
    diff = 3
    dec = 4
    grib2 = 5


class ForecastType(Enum):
    """Forecast type."""

    analysis = 0
    forecast = 1
    guess = 2
    initial = 3


class DataSource(Enum):
    """Data source."""

    model = 0
    airway_surface = 1
    metar = 2
    ship = 3
    raob_buoy = 4
    synop_raob_vas = 5
    grid = 6
    watch_by_county = 7
    unknown = 99
    text = 100
    metar2 = 102
    ship2 = 103
    raob_buoy2 = 104
    synop_raob_vas2 = 105
