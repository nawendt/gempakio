# Copyright (c) 2022 Nathan Wendt.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Classes for decoding various GEMPAK file formats."""

import bisect
from collections import namedtuple
from collections.abc import Iterable
import contextlib
from copy import deepcopy
import ctypes
from datetime import datetime, timedelta
from itertools import product
import logging
import math
from pathlib import Path
import re
import struct
import sys

import numpy as np
import pyproj
import xarray as xr

from gempakio.common import (ANLB_SIZE, DataSource, DataTypes, FileTypes, ForecastType,
                             GEMPAK_HEADER, NAVB_SIZE, PackingType, VerticalCoordinates)
from gempakio.gemcalc import (interp_logp_height, interp_logp_pressure, interp_missing_data,
                              interp_moist_height)
from gempakio.tools import IOBuffer, NamedStruct

logger = logging.getLogger(__name__)

BYTES_PER_WORD = 4
PARAM_ATTR = [('name', (4, 's')), ('scale', (1, 'i')),
              ('offset', (1, 'i')), ('bits', (1, 'i'))]
USED_FLAG = 9999
UNUSED_FLAG = -9999
GEMPROJ_TO_PROJ = {
    'MER': ('merc', 'cyl'),
    'NPS': ('stere', 'azm'),
    'SPS': ('stere', 'azm'),
    'LCC': ('lcc', 'con'),
    'SCC': ('lcc', 'con'),
    'CED': ('eqc', 'cyl'),
    'MCD': ('eqc', 'cyl'),
    'NOR': ('ortho', 'azm'),
    'SOR': ('ortho', 'azm'),
    'STR': ('stere', 'azm'),
    'AED': ('aeqd', 'azm'),
    'ORT': ('ortho', 'azm'),
    'LEA': ('laea', 'azm'),
    'GNO': ('gnom', 'azm'),
    'TVM': ('tmerc', 'obq'),
    'UTM': ('utm', 'obq'),
}
GVCORD_TO_VAR = {
    'PRES': 'p',
    'HGHT': 'z',
    'THTA': 'theta',
}
METAR_STATION_RE = re.compile(
    r'^(?P<station>[A-Z][A-Z0-9]{3})\s+'
)
METAR_TIME_RE = re.compile(
    r'(?P<day>\d\d)(?P<hour>\d\d)(?P<minute>\d\d)Z'
)

Grid = namedtuple('Grid', [
    'GRIDNO',
    'TYPE',
    'DATTIM1',
    'DATTIM2',
    'PARM',
    'LEVEL1',
    'LEVEL2',
    'COORD',
])

Sounding = namedtuple('Sounding', [
    'DTNO',
    'SNDNO',
    'DATTIM',
    'ID',
    'NUMBER',
    'LAT',
    'LON',
    'ELEV',
    'STATE',
    'COUNTRY',
])

Surface = namedtuple('Surface', [
    'ROW',
    'COL',
    'DATTIM',
    'ID',
    'NUMBER',
    'LAT',
    'LON',
    'ELEV',
    'STATE',
    'COUNTRY',
])


def _bbox_filter(lat, lon, bbox):
    """Filter stations by bbox.

    Notes
    -----
    Expects lon in [-180, 180].
    """
    left, right, bottom, top = bbox

    return (lon >= left and lon <= right
            and lat >= bottom and lat <= top)


def _data_source(source):
    """Get data source from stored integer."""
    try:
        DataSource(source)
    except ValueError:
        logger.warning('Could not interpret data source `%s`. '
                       'Setting to `Unknown`.', source)
        return DataSource(99)
    else:
        return DataSource(source)


def _word_to_position(word, bytes_per_word=BYTES_PER_WORD):
    """Return beginning position of a word in bytes."""
    return (word * bytes_per_word) - bytes_per_word


class GempakFile:
    """Base class for GEMPAK files.

    Reads ubiquitous GEMPAK file headers (i.e., the data managment portion of
    each file).
    """

    prod_desc_fmt = [('version', 'i'), ('file_headers', 'i'),
                     ('file_keys_ptr', 'i'), ('rows', 'i'),
                     ('row_keys', 'i'), ('row_keys_ptr', 'i'),
                     ('row_headers_ptr', 'i'), ('columns', 'i'),
                     ('column_keys', 'i'), ('column_keys_ptr', 'i'),
                     ('column_headers_ptr', 'i'), ('parts', 'i'),
                     ('parts_ptr', 'i'), ('data_mgmt_ptr', 'i'),
                     ('data_mgmt_length', 'i'), ('data_block_ptr', 'i'),
                     ('file_type', 'i', FileTypes),
                     ('data_source', 'i', _data_source),
                     ('machine_type', 'i'), ('missing_int', 'i'),
                     (None, '12x'), ('missing_float', 'f')]

    grid_nav_fmt = [('grid_definition_type', 'f'),
                    ('projection', '3sx', bytes.decode),
                    ('left_grid_number', 'f'), ('bottom_grid_number', 'f'),
                    ('right_grid_number', 'f'), ('top_grid_number', 'f'),
                    ('lower_left_lat', 'f'), ('lower_left_lon', 'f'),
                    ('upper_right_lat', 'f'), ('upper_right_lon', 'f'),
                    ('proj_angle1', 'f'), ('proj_angle2', 'f'),
                    ('proj_angle3', 'f'), (None, '972x')]

    grid_anl_fmt1 = [('analysis_type', 'f'), ('delta_n', 'f'),
                     ('delta_x', 'f'), ('delta_y', 'f'),
                     (None, '4x'), ('garea_llcr_lat', 'f'),
                     ('garea_llcr_lon', 'f'), ('garea_urcr_lat', 'f'),
                     ('garea_urcr_lon', 'f'), ('extarea_llcr_lat', 'f'),
                     ('extarea_llcr_lon', 'f'), ('extarea_urcr_lat', 'f'),
                     ('extarea_urcr_lon', 'f'), ('datarea_llcr_lat', 'f'),
                     ('datarea_llcr_lon', 'f'), ('datarea_urcr_lat', 'f'),
                     ('datarea_urcrn_lon', 'f'), (None, '444x')]

    grid_anl_fmt2 = [('analysis_type', 'f'), ('delta_n', 'f'),
                     ('grid_ext_left', 'f'), ('grid_ext_down', 'f'),
                     ('grid_ext_right', 'f'), ('grid_ext_up', 'f'),
                     ('garea_llcr_lat', 'f'), ('garea_llcr_lon', 'f'),
                     ('garea_urcr_lat', 'f'), ('garea_urcr_lon', 'f'),
                     ('extarea_llcr_lat', 'f'), ('extarea_llcr_lon', 'f'),
                     ('extarea_urcr_lat', 'f'), ('extarea_urcr_lon', 'f'),
                     ('datarea_llcr_lat', 'f'), ('datarea_llcr_lon', 'f'),
                     ('datarea_urcr_lat', 'f'), ('datarea_urcrn_lon', 'f'),
                     (None, '440x')]

    data_management_fmt = ([('next_free_word', 'i'), ('max_free_pairs', 'i'),
                           ('actual_free_pairs', 'i'), ('last_word', 'i')]
                           + [(f'free_word{n}', 'i') for n in range(1, 29)])

    def __init__(self, file):
        """Instantiate GempakFile object from file."""
        if isinstance(file, Path):
            file = str(file)

        with contextlib.closing(open(file, 'rb')) as fobj:  # noqa: SIM115
            self._buffer = IOBuffer.fromfile(fobj)

        # Save file start position as pointers use this as reference
        self._start = self._buffer.set_mark()

        # Process the main GEMPAK header to verify file format
        self._process_gempak_header()
        meta = self._buffer.set_mark()

        # # Check for byte swapping
        self._swap_bytes(bytes(self._buffer.read_binary(4)))
        self._buffer.jump_to(meta)

        # Process main metadata header
        self.prod_desc = self._buffer.read_struct(NamedStruct(self.prod_desc_fmt,
                                                              self.prefmt,
                                                              'ProductDescription'))

        # File Keys
        # Surface and upper-air files will not have the file headers, so we need to check.
        if self.prod_desc.file_headers > 0:
            # This would grab any file headers, but NAVB and ANLB are the only ones used.
            fkey_prod = product(['header_name', 'header_length', 'header_type'],
                                range(1, self.prod_desc.file_headers + 1))
            fkey_names = [f'{x[0]}{x[1]}' for x in fkey_prod]
            fkey_info = list(zip(fkey_names, np.repeat(('4s', 'i', 'i'),
                                                       self.prod_desc.file_headers)))
            self.file_keys_format = NamedStruct(fkey_info, self.prefmt, 'FileKeys')

            self._buffer.jump_to(self._start, _word_to_position(self.prod_desc.file_keys_ptr))
            self.file_keys = self._buffer.read_struct(self.file_keys_format)

            # file_key_blocks = self._buffer.set_mark()
            # Navigation Block
            navb_size = self._buffer.read_int(4, self.endian, False)
            if navb_size != NAVB_SIZE:
                raise ValueError('Navigation block size does not match GEMPAK specification')
            else:
                self.navigation_block = (
                    self._buffer.read_struct(NamedStruct(self.grid_nav_fmt,
                                                         self.prefmt,
                                                         'NavigationBlock'))
                )
            self.kx = int(self.navigation_block.right_grid_number)
            self.ky = int(self.navigation_block.top_grid_number)

            # Analysis Block
            anlb_size = self._buffer.read_int(4, self.endian, False)
            anlb_start = self._buffer.set_mark()
            if anlb_size != ANLB_SIZE:
                raise ValueError('Analysis block size does not match GEMPAK specification')
            else:
                anlb_type = self._buffer.read_struct(struct.Struct(self.prefmt + 'f'))[0]
                self._buffer.jump_to(anlb_start)
                if anlb_type == 1:
                    self.analysis_block = (
                        self._buffer.read_struct(NamedStruct(self.grid_anl_fmt1,
                                                             self.prefmt,
                                                             'AnalysisBlock'))
                    )
                elif anlb_type == 2:
                    self.analysis_block = (
                        self._buffer.read_struct(NamedStruct(self.grid_anl_fmt2,
                                                             self.prefmt,
                                                             'AnalysisBlock'))
                    )
                else:
                    self.analysis_block = None
        else:
            self.analysis_block = None
            self.navigation_block = None

        # Data Management
        self._buffer.jump_to(self._start, _word_to_position(self.prod_desc.data_mgmt_ptr))
        self.data_management = self._buffer.read_struct(NamedStruct(self.data_management_fmt,
                                                                    self.prefmt,
                                                                    'DataManagement'))

        # Row Keys
        self._buffer.jump_to(self._start, _word_to_position(self.prod_desc.row_keys_ptr))
        row_key_info = [(f'row_key{n}', '4s', self._decode_strip)
                        for n in range(1, self.prod_desc.row_keys + 1)]
        row_key_info.extend([(None, None)])
        row_keys_fmt = NamedStruct(row_key_info, self.prefmt, 'RowKeys')
        self.row_keys = self._buffer.read_struct(row_keys_fmt)

        # Column Keys
        self._buffer.jump_to(self._start, _word_to_position(self.prod_desc.column_keys_ptr))
        column_key_info = [(f'column_key{n}', '4s', self._decode_strip)
                           for n in range(1, self.prod_desc.column_keys + 1)]
        column_key_info.extend([(None, None)])
        column_keys_fmt = NamedStruct(column_key_info, self.prefmt, 'ColumnKeys')
        self.column_keys = self._buffer.read_struct(column_keys_fmt)

        # Parts
        self._buffer.jump_to(self._start, _word_to_position(self.prod_desc.parts_ptr))
        # parts = self._buffer.set_mark()
        self.parts = []
        parts_info = [('name', '4s', self._decode_strip),
                      (None, f'{(self.prod_desc.parts - 1) * BYTES_PER_WORD}x'),
                      ('header_length', 'i'),
                      (None, f'{(self.prod_desc.parts - 1) * BYTES_PER_WORD}x'),
                      ('data_type', 'i', DataTypes),
                      (None, f'{(self.prod_desc.parts - 1) * BYTES_PER_WORD}x'),
                      ('parameter_count', 'i')]
        parts_info.extend([(None, None)])
        parts_fmt = NamedStruct(parts_info, self.prefmt, 'Parts')
        for n in range(1, self.prod_desc.parts + 1):
            self.parts.append(self._buffer.read_struct(parts_fmt))
            self._buffer.jump_to(self._start, _word_to_position(self.prod_desc.parts_ptr + n))

        # Parameters
        # No need to jump to any position as this follows parts information
        self._buffer.jump_to(self._start, _word_to_position(self.prod_desc.parts_ptr
                                                            + self.prod_desc.parts * 4))
        self.parameters = [{key: [] for key, _ in PARAM_ATTR}
                           for n in range(self.prod_desc.parts)]
        for attr, fmt in PARAM_ATTR:
            fmt = (fmt[0], self.prefmt + fmt[1] if fmt[1] != 's' else fmt[1])
            for n, part in enumerate(self.parts):
                for _ in range(part.parameter_count):
                    if 's' in fmt[1]:
                        self.parameters[n][attr] += [
                            self._decode_strip(self._buffer.read_binary(*fmt)[0])
                        ]
                    else:
                        self.parameters[n][attr] += self._buffer.read_binary(*fmt)

    def _swap_bytes(self, binary):
        """Swap between little and big endian."""
        self.swaped_bytes = (struct.pack('@i', 1) != binary)

        if self.swaped_bytes:
            if sys.byteorder == 'little':
                self.prefmt = '>'
                self.endian = 'big'
            elif sys.byteorder == 'big':
                self.prefmt = '<'
                self.endian = 'little'
        else:
            self.prefmt = ''
            self.endian = sys.byteorder

    def _process_gempak_header(self):
        """Read the GEMPAK header from the file."""
        fmt = [('text', '28s', bytes.decode), (None, None)]

        header = self._buffer.read_struct(NamedStruct(fmt, '', 'GempakHeader'))
        if header.text != GEMPAK_HEADER:
            raise TypeError('Unknown file format or invalid GEMPAK file')

    @staticmethod
    def _convert_dattim(dattim):
        """Convert GEMPAK DATTIM integer to datetime object."""
        if dattim:
            if dattim < 100000000:
                dt = datetime.strptime(f'{dattim:06d}', '%y%m%d')
            else:
                dt = datetime.strptime(f'{dattim:010d}', '%m%d%y%H%M')
        else:
            dt = None
        return dt

    @staticmethod
    def _convert_ftime(ftime):
        """Convert GEMPAK forecast time and type integer."""
        if ftime >= 0:
            iftype = ForecastType(ftime // 100000)
            iftime = ftime - iftype.value * 100000
            hours = iftime // 100
            minutes = iftime - hours * 100
            out = (iftype.name, timedelta(hours=hours, minutes=minutes))
        else:
            out = None
        return out

    @staticmethod
    def _convert_level(level):
        """Convert levels."""
        if isinstance(level, (int, float)) and level >= 0:
            return level
        else:
            return None

    @staticmethod
    def _convert_vertical_coord(coord):
        """Convert integer vertical coordinate to name."""
        if coord <= 8:
            return VerticalCoordinates(coord).name.upper()
        else:
            return struct.pack('i', coord).decode()

    @staticmethod
    def _fortran_ishift(i, shift):
        """Python-friendly bit shifting."""
        mask = 0xffffffff
        if shift > 0:
            shifted = ctypes.c_int32(i << shift).value
        elif shift < 0:
            if i < 0:
                shifted = (i & mask) >> abs(shift)
            else:
                shifted = i >> abs(shift)
        elif shift == 0:
            shifted = i
        else:
            raise ValueError(f'Bad shift value {shift}.')
        return shifted

    @staticmethod
    def _decode_strip(b):
        """Decode bytes to string and strip whitespace."""
        return b.decode().strip()

    @staticmethod
    def _make_date(dattim):
        """Make a date object from GEMPAK DATTIM integer."""
        return GempakFile._convert_dattim(dattim).date()

    @staticmethod
    def _make_time(t):
        """Make a time object from GEMPAK FTIME integer."""
        string = f'{t:04d}'
        return datetime.strptime(string, '%H%M').time()

    def _unpack_real(self, buffer, parameters, length):
        """Unpack floating point data packed in integers.

        Similar to DP_UNPK subroutine in GEMPAK.
        """
        nparms = len(parameters['name'])
        mskpat = 0xffffffff

        pwords = (sum(parameters['bits']) - 1) // 32 + 1
        npack = (length - 1) // pwords + 1
        unpacked = np.ones(npack * nparms, dtype=np.float32) * self.prod_desc.missing_float
        if npack * pwords != length:
            raise ValueError('Unpacking length mismatch.')

        ir = 0
        ii = 0
        for _i in range(npack):
            pdat = buffer[ii:(ii + pwords)]
            rdat = unpacked[ir:(ir + nparms)]
            itotal = 0
            for idata in range(nparms):
                scale = 10**parameters['scale'][idata]
                offset = parameters['offset'][idata]
                bits = parameters['bits'][idata]
                isbitc = (itotal % 32) + 1
                iswrdc = (itotal // 32)
                imissc = self._fortran_ishift(mskpat, bits - 32)

                jbit = bits
                jsbit = isbitc
                jshift = 1 - jsbit
                jsword = iswrdc
                jword = pdat[jsword]
                mask = self._fortran_ishift(mskpat, jbit - 32)
                ifield = self._fortran_ishift(jword, jshift)
                ifield &= mask

                if (jsbit + jbit - 1) > 32:
                    jword = pdat[jsword + 1]
                    jshift += 32
                    iword = self._fortran_ishift(jword, jshift)
                    iword &= mask
                    ifield |= iword

                if ifield == imissc:
                    rdat[idata] = self.prod_desc.missing_float
                else:
                    rdat[idata] = (ifield + offset) * scale
                itotal += bits
            unpacked[ir:(ir + nparms)] = rdat
            ir += nparms
            ii += pwords

        return unpacked.tolist()


class GempakGrid(GempakFile):
    """Subclass of GempakFile specific to GEMPAK gridded data."""

    def __init__(self, file, *args, **kwargs):
        """Instantiate GempakGrid object from file."""
        super().__init__(file)

        datetime_names = ['GDT1', 'GDT2']
        level_names = ['GLV1', 'GLV2']
        ftime_names = ['GTM1', 'GTM2']
        string_names = ['GPM1', 'GPM2', 'GPM3']

        # Row Headers
        # Based on GEMPAK source, row/col headers have a 0th element in their Fortran arrays.
        # This appears to be a flag value to say a header is used or not. 9999
        # means its in use, otherwise -9999. GEMPAK allows empty grids, etc., but
        # no real need to keep track of that in Python.
        self._buffer.jump_to(self._start, _word_to_position(self.prod_desc.row_headers_ptr))
        self.row_headers = []
        row_headers_info = [(key, 'i') for key in self.row_keys]
        row_headers_info.extend([(None, None)])
        row_headers_fmt = NamedStruct(row_headers_info, self.prefmt, 'RowHeaders')
        for _ in range(1, self.prod_desc.rows + 1):
            if self._buffer.read_int(4, self.endian, False) == USED_FLAG:
                self.row_headers.append(self._buffer.read_struct(row_headers_fmt))

        # Column Headers
        self._buffer.jump_to(self._start, _word_to_position(self.prod_desc.column_headers_ptr))
        self.column_headers = []
        column_headers_info = [(key, 'i', self._convert_level) if key in level_names
                               else (key, 'i', self._convert_vertical_coord) if key == 'GVCD'
                               else (key, 'i', self._convert_dattim) if key in datetime_names
                               else (key, 'i', self._convert_ftime) if key in ftime_names
                               else (key, '4s', self._decode_strip) if key in string_names
                               else (key, 'i')
                               for key in self.column_keys]
        column_headers_info.extend([(None, None)])
        column_headers_fmt = NamedStruct(column_headers_info, self.prefmt, 'ColumnHeaders')
        for _ in range(1, self.prod_desc.columns + 1):
            if self._buffer.read_int(4, self.endian, False) == USED_FLAG:
                self.column_headers.append(self._buffer.read_struct(column_headers_fmt))

        self._gdinfo = set()
        for n, head in enumerate(self.column_headers):
            self._gdinfo.add(
                Grid(
                    n,
                    head.GTM1[0],
                    head.GDT1 + head.GTM1[1],
                    head.GDT2 + head.GTM2[1] if head.GDT2 and head.GDTM2 else None,
                    head.GPM1 + head.GPM2 + head.GPM3,
                    head.GLV1,
                    head.GLV2,
                    head.GVCD,
                )
            )

        # Coordinates
        if self.navigation_block is not None:
            self._get_crs()
            self._set_coordinates()

    def gdinfo(self):
        """Return grid information."""
        return sorted(self._gdinfo)

    def project_point(self, lon, lat):
        """Project geographic corrdinates.

        Parameters
        ----------
        lon : float or array-like of float
            Longitude of point(s).

        lat : float or array-like of float
            Latitude of point(s).

        Returns
        -------
        tuple
            Tuple containing lists of x and y projected
            coordinate values.
        """
        return self._transform(lon, lat)

    def _get_crs(self):
        """Create CRS from GEMPAK navigation block."""
        gemproj = self.navigation_block.projection
        if gemproj not in GEMPROJ_TO_PROJ:
            raise NotImplementedError(f'{gemproj} projection not implemented.')
        proj, ptype = GEMPROJ_TO_PROJ[gemproj]
        ellps = 'sphere'  # Kept for posterity
        earth_radius = 6371200.0  # R takes precedence over ellps

        if ptype == 'azm':
            lat_0 = self.navigation_block.proj_angle1
            lon_0 = self.navigation_block.proj_angle2
            rot = self.navigation_block.proj_angle3
            if rot != 0:
                logger.warning('Rotated projections currently '
                               'not supported. Angle3 (%7.2f) ignored.', rot)
            self.crs = pyproj.CRS.from_dict({'proj': proj,
                                             'lat_0': lat_0,
                                             'lon_0': lon_0,
                                             'ellps': ellps,
                                             'R': earth_radius})
        elif ptype == 'cyl':
            if gemproj != 'MCD':
                lat_0 = self.navigation_block.proj_angle1
                lon_0 = self.navigation_block.proj_angle2
                rot = self.navigation_block.proj_angle3
                if rot != 0:
                    logger.warning('Rotated projections currently '
                                   'not supported. Angle3 (%7.2f) ignored.', rot)
                self.crs = pyproj.CRS.from_dict({'proj': proj,
                                                 'lat_0': lat_0,
                                                 'lon_0': lon_0,
                                                 'ellps': ellps,
                                                 'R': earth_radius})
            else:
                avglat = (self.navigation_block.upper_right_lat
                          + self.navigation_block.lower_left_lat) * 0.5
                k_0 = (1 / math.cos(avglat)
                       if self.navigation_block.proj_angle1 == 0
                       else self.navigation_block.proj_angle1
                       )
                lon_0 = self.navigation_block.proj_angle2
                self.crs = pyproj.CRS.from_dict({'proj': proj,
                                                 'lat_0': avglat,
                                                 'lon_0': lon_0,
                                                 'k_0': k_0,
                                                 'ellps': ellps,
                                                 'R': earth_radius})
        elif ptype == 'con':
            lat_1 = self.navigation_block.proj_angle1
            lon_0 = self.navigation_block.proj_angle2
            lat_2 = self.navigation_block.proj_angle3
            self.crs = pyproj.CRS.from_dict({'proj': proj,
                                             'lon_0': lon_0,
                                             'lat_1': lat_1,
                                             'lat_2': lat_2,
                                             'ellps': ellps,
                                             'R': earth_radius})

        elif ptype == 'obq':
            lon_0 = self.navigation_block.proj_angle1
            if gemproj == 'UTM':
                zone = np.digitize((lon_0 % 360) / 6 + 1, range(1, 61), right=True)
                self.crs = pyproj.CRS.from_dict({'proj': proj,
                                                 'zone': zone,
                                                 'ellps': ellps,
                                                 'R': earth_radius})
            else:
                self.crs = pyproj.CRS.from_dict({'proj': proj,
                                                 'lon_0': lon_0,
                                                 'ellps': ellps,
                                                 'R': earth_radius})

    def _set_coordinates(self):
        """Use GEMPAK navigation block to define coordinates.

        Defines geographic and projection coordinates for the object.
        """
        transform = pyproj.Proj(self.crs)
        self._transform = transform
        llx, lly = transform(self.navigation_block.lower_left_lon,
                             self.navigation_block.lower_left_lat)
        urx, ury = transform(self.navigation_block.upper_right_lon,
                             self.navigation_block.upper_right_lat)
        self.x = np.linspace(llx, urx, self.kx, dtype=np.float32)
        self.y = np.linspace(lly, ury, self.ky, dtype=np.float32)
        xx, yy = np.meshgrid(self.x, self.y, copy=False)
        self.lon, self.lat = transform(xx, yy, inverse=True)
        self.lon = self.lon.astype(np.float32)
        self.lat = self.lat.astype(np.float32)

    def _unpack_grid(self, packing_type, part):
        """Read raw GEMPAK grid integers and unpack into floats."""
        if packing_type == PackingType.none:
            lendat = self.data_header_length - part.header_length - 1

            if lendat > 1:
                buffer_fmt = f'{self.prefmt}{lendat}f'
                buffer = self._buffer.read_struct(struct.Struct(buffer_fmt))
                grid = np.zeros(self.ky * self.kx, dtype=np.float32)
                grid[...] = buffer
            else:
                grid = None

            return grid

        elif packing_type == PackingType.nmc:
            raise NotImplementedError('NMC unpacking not supported.')
            # integer_meta_fmt = [('bits', 'i'), ('missing_flag', 'i'), ('kxky', 'i')]
            # real_meta_fmt = [('reference', 'f'), ('scale', 'f')]
            # self.grid_meta_int = self._buffer.read_struct(NamedStruct(integer_meta_fmt,
            #                                                           self.prefmt,
            #                                                           'GridMetaInt'))
            # self.grid_meta_real = self._buffer.read_struct(NamedStruct(real_meta_fmt,
            #                                                            self.prefmt,
            #                                                            'GridMetaReal'))
            # grid_start = self._buffer.set_mark()
        elif packing_type == PackingType.diff:
            integer_meta_fmt = [('bits', 'i'), ('missing_flag', 'i'),
                                ('kxky', 'i'), ('kx', 'i')]
            real_meta_fmt = [('reference', 'f'), ('scale', 'f'), ('diffmin', 'f')]
            self.grid_meta_int = self._buffer.read_struct(NamedStruct(integer_meta_fmt,
                                                                      self.prefmt,
                                                                      'GridMetaInt'))
            self.grid_meta_real = self._buffer.read_struct(NamedStruct(real_meta_fmt,
                                                                       self.prefmt,
                                                                       'GridMetaReal'))
            # grid_start = self._buffer.set_mark()

            imiss = 2**self.grid_meta_int.bits - 1
            lendat = self.data_header_length - part.header_length - 8
            packed_buffer_fmt = f'{self.prefmt}{lendat}i'
            packed_buffer = self._buffer.read_struct(struct.Struct(packed_buffer_fmt))
            grid = np.zeros((self.ky, self.kx), dtype=np.float32)

            if lendat > 1:
                iword = 0
                ibit = 1
                first = True
                for j in range(self.ky):
                    line = False
                    for i in range(self.kx):
                        jshft = self.grid_meta_int.bits + ibit - 33
                        idat = self._fortran_ishift(packed_buffer[iword], jshft)
                        idat &= imiss

                        if jshft > 0:
                            jshft -= 32
                            idat2 = self._fortran_ishift(packed_buffer[iword + 1], jshft)
                            idat |= idat2

                        ibit += self.grid_meta_int.bits
                        if ibit > 32:
                            ibit -= 32
                            iword += 1

                        if (self.grid_meta_int.missing_flag and idat == imiss):
                            grid[j, i] = self.prod_desc.missing_float
                        else:
                            if first:
                                grid[j, i] = self.grid_meta_real.reference
                                psav = self.grid_meta_real.reference
                                plin = self.grid_meta_real.reference
                                line = True
                                first = False
                            else:
                                if not line:
                                    grid[j, i] = plin + (self.grid_meta_real.diffmin
                                                         + idat * self.grid_meta_real.scale)
                                    line = True
                                    plin = grid[j, i]
                                else:
                                    grid[j, i] = psav + (self.grid_meta_real.diffmin
                                                         + idat * self.grid_meta_real.scale)
                                psav = grid[j, i]
            else:
                grid = None

            return grid

        elif packing_type in [PackingType.grib, PackingType.dec]:
            integer_meta_fmt = [('bits', 'i'), ('missing_flag', 'i'), ('kxky', 'i')]
            real_meta_fmt = [('reference', 'f'), ('scale', 'f')]
            self.grid_meta_int = self._buffer.read_struct(NamedStruct(integer_meta_fmt,
                                                                      self.prefmt,
                                                                      'GridMetaInt'))
            self.grid_meta_real = self._buffer.read_struct(NamedStruct(real_meta_fmt,
                                                                       self.prefmt,
                                                                       'GridMetaReal'))
            # grid_start = self._buffer.set_mark()

            lendat = self.data_header_length - part.header_length - 6
            packed_buffer_fmt = f'{self.prefmt}{lendat}i'

            grid = np.zeros(self.grid_meta_int.kxky, dtype=np.float32)
            packed_buffer = self._buffer.read_struct(struct.Struct(packed_buffer_fmt))
            if lendat > 1:
                imax = 2**self.grid_meta_int.bits - 1
                ibit = 1
                iword = 0
                for cell in range(self.grid_meta_int.kxky):
                    jshft = self.grid_meta_int.bits + ibit - 33
                    idat = self._fortran_ishift(packed_buffer[iword], jshft)
                    idat &= imax

                    if jshft > 0:
                        jshft -= 32
                        idat2 = self._fortran_ishift(packed_buffer[iword + 1], jshft)
                        idat |= idat2

                    if (idat == imax) and self.grid_meta_int.missing_flag:
                        grid[cell] = self.prod_desc.missing_float
                    else:
                        grid[cell] = (self.grid_meta_real.reference
                                      + (idat * self.grid_meta_real.scale))

                    ibit += self.grid_meta_int.bits
                    if ibit > 32:
                        ibit -= 32
                        iword += 1
            else:
                grid = None

            return grid
        elif packing_type == PackingType.grib2:
            raise NotImplementedError('GRIB2 unpacking not supported.')
            # integer_meta_fmt = [('iuscal', 'i'), ('kx', 'i'),
            #                     ('ky', 'i'), ('iscan_mode', 'i')]
            # real_meta_fmt = [('rmsval', 'f')]
            # self.grid_meta_int = self._buffer.read_struct(NamedStruct(integer_meta_fmt,
            #                                                           self.prefmt,
            #                                                           'GridMetaInt'))
            # self.grid_meta_real = self._buffer.read_struct(NamedStruct(real_meta_fmt,
            #                                                            self.prefmt,
            #                                                            'GridMetaReal'))
            # grid_start = self._buffer.set_mark()
        else:
            raise NotImplementedError(
                f'No method for unknown grid packing {packing_type.name}'
            )

    def gdxarray(self, parameter=None, date_time=None, coordinate=None,
                 level=None, date_time2=None, level2=None):
        """Select grids and output as list of xarray DataArrays.

        Subset the data by parameter values. The default is to not
        subset and return the entire dataset.

        Parameters
        ----------
        parameter : str or array-like of str
            Name of GEMPAK parameter.

        date_time : datetime or array-like of datetime
            Datetime of the grid. Alternatively a string with
            the format YYYYmmddHHMM or first|FIRST or last|LAST
            which function to retrieve the latest and oldest
            time within the file, respectively.

        coordinate : str or array-like of str
            Vertical coordinate.

        level : float or array-like of float
            Vertical level.

        date_time2 : datetime or array-like of datetime
            Secondary valid datetime of the grid. Alternatively
            a string with the format YYYYmmddHHMM or first|FIRST
            or last|LAST which function to retrieve the latest
            and oldest time within the file, respectively.

        level2: float or array_like of float
            Secondary vertical level. Typically used for layers.

        Returns
        -------
        list
            List of xarray.DataArray objects for each grid.

        Notes
        -----
        When multiple filters are used, the order of what is returned
        in the list is determined by the order of the grids in the
        GEMPAK file. For example, if you request both U and V wind
        grids in that order, you still may get V winds in the returned
        list before U winds if that is how they were placed in file
        originally. If order becomes important, independent calls to
        gdxarray are more appropriate.
        """
        if parameter is not None:
            if (not isinstance(parameter, Iterable)
               or isinstance(parameter, str)):
                parameter = [parameter]
            parameter = [p.upper() for p in parameter]

        if date_time is not None:
            if (not isinstance(date_time, Iterable)
               or (isinstance(date_time, str)
               and date_time not in ['first', 'FIRST', 'last', 'LAST'])):
                date_time = [date_time]
            if date_time not in ['first', 'FIRST', 'last', 'LAST']:
                for i, dt in enumerate(date_time):
                    if isinstance(dt, str):
                        date_time[i] = datetime.strptime(dt, '%Y%m%d%H%M')

        if coordinate is not None:
            if (not isinstance(coordinate, Iterable)
               or isinstance(coordinate, str)):
                coordinate = [coordinate]
            coordinate = [c.upper() for c in coordinate]

        if level is not None and not isinstance(level, Iterable):
            level = [level]

        if date_time2 is not None:
            if (not isinstance(date_time2, Iterable)
               or (isinstance(date_time2, str)
               and date_time2 not in ['first', 'FIRST', 'last', 'LAST'])):
                date_time2 = [date_time2]
            if date_time2 not in ['first', 'FIRST', 'last', 'LAST']:
                for i, dt in enumerate(date_time2):
                    if isinstance(dt, str):
                        date_time2[i] = datetime.strptime(dt, '%Y%m%d%H%M')

        if level2 is not None and not isinstance(level2, Iterable):
            level2 = [level2]

        # Figure out which columns to extract from the file
        matched = sorted(self._gdinfo)

        # Do this now or the matched filter iterator will be consumed
        # prematurely.
        if date_time in ['last', 'LAST']:
            date_time = [max((d.DATTIM1 for d in matched))]
        elif date_time in ['first', 'FIRST']:
            date_time = [min((d.DATTIM1 for d in matched))]

        if date_time2 in ['last', 'LAST']:
            date_time2 = [max((d.DATTIM2 for d in matched))]
        elif date_time2 in ['first', 'FIRST']:
            date_time2 = [min((d.DATTIM2 for d in matched))]

        if parameter is not None:
            matched = filter(
                lambda grid: grid.PARM in parameter,
                matched
            )

        if date_time is not None:
            matched = filter(
                lambda grid: grid.DATTIM1 in date_time,
                matched
            )

        if coordinate is not None:
            matched = filter(
                lambda grid: grid.COORD in coordinate,
                matched
            )

        if level is not None:
            matched = filter(
                lambda grid: grid.LEVEL1 in level,
                matched
            )

        if date_time2 is not None:
            matched = filter(
                lambda grid: grid.DATTIM2 in date_time2,
                matched
            )

        if level2 is not None:
            matched = filter(
                lambda grid: grid.LEVEL2 in level2,
                matched
            )

        matched = list(matched)

        if len(matched) < 1:
            raise KeyError('No grids were matched with given parameters.')

        gridno = [g.GRIDNO for g in matched]

        grids = []
        irow = 0  # Only one row for grids
        for icol in gridno:
            col_head = self.column_headers[icol]
            for iprt, part in enumerate(self.parts):
                pointer = (self.prod_desc.data_block_ptr
                           + (irow * self.prod_desc.columns * self.prod_desc.parts)
                           + (icol * self.prod_desc.parts + iprt))
                self._buffer.jump_to(self._start, _word_to_position(pointer))
                self.data_ptr = self._buffer.read_int(4, self.endian, False)
                self._buffer.jump_to(self._start, _word_to_position(self.data_ptr))
                self.data_header_length = self._buffer.read_int(4, self.endian, False)
                data_header = self._buffer.set_mark()
                self._buffer.jump_to(data_header,
                                     _word_to_position(part.header_length + 1))
                packing_type = PackingType(self._buffer.read_int(4, self.endian, False))

                full_name = col_head.GPM1 + col_head.GPM2 + col_head.GPM3
                ftype, ftime = col_head.GTM1
                valid = col_head.GDT1 + ftime
                gvcord = col_head.GVCD.lower() if col_head.GVCD is not None else 'none'
                var = (GVCORD_TO_VAR[full_name]
                       if full_name in GVCORD_TO_VAR
                       else full_name.lower()
                       )
                data = self._unpack_grid(packing_type, part)
                if data is not None:
                    if data.ndim < 2:
                        data = np.ma.array(data.reshape((self.ky, self.kx)),
                                           mask=data == self.prod_desc.missing_float,
                                           dtype=np.float32)
                    else:
                        data = np.ma.array(data, mask=data == self.prod_desc.missing_float,
                                           dtype=np.float32)

                    xrda = xr.DataArray(
                        data=data[np.newaxis, np.newaxis, ...],
                        coords={
                            'time': [valid],
                            gvcord: [col_head.GLV1],
                            'x': self.x,
                            'y': self.y,
                            'lat': (['y', 'x'], self.lat),
                            'lon': (['y', 'x'], self.lon),
                        },
                        dims=['time', gvcord, 'y', 'x'],
                        name=var,
                        attrs={
                            **self.crs.to_cf(),
                            'grid_type': ftype,
                        }
                    )
                    grids.append(xrda)

                else:
                    logger.warning('Unable to read grid for %s', col_head.GPM1)
        return grids


class GempakSounding(GempakFile):
    """Subclass of GempakFile specific to GEMPAK sounding data."""

    def __init__(self, file, *args, **kwargs):
        """Instantiate GempakSounding object from file."""
        super().__init__(file)

        # Row Headers
        self._buffer.jump_to(self._start, _word_to_position(self.prod_desc.row_headers_ptr))
        self.row_headers = []
        row_headers_info = [(key, 'i', self._make_date) if key == 'DATE'
                            else (key, 'i', self._make_time) if key == 'TIME'
                            else (key, 'i')
                            for key in self.row_keys]
        row_headers_info.extend([(None, None)])
        row_headers_fmt = NamedStruct(row_headers_info, self.prefmt, 'RowHeaders')
        for _ in range(1, self.prod_desc.rows + 1):
            if self._buffer.read_int(4, self.endian, False) == USED_FLAG:
                self.row_headers.append(self._buffer.read_struct(row_headers_fmt))

        # Column Headers
        self._buffer.jump_to(self._start, _word_to_position(self.prod_desc.column_headers_ptr))
        self.column_headers = []
        column_headers_info = [(key, '4s', self._decode_strip) if key == 'STID'
                               else (key, 'i') if key == 'STNM'
                               else (key, 'i', lambda x: x / 100) if key == 'SLAT'
                               else (key, 'i', lambda x: x / 100) if key == 'SLON'
                               else (key, 'i') if key == 'SELV'
                               else (key, '4s', self._decode_strip) if key == 'STAT'
                               else (key, '4s', self._decode_strip) if key == 'COUN'
                               else (key, '4s', self._decode_strip) if key == 'STD2'
                               else (key, 'i')
                               for key in self.column_keys]
        column_headers_info.extend([(None, None)])
        column_headers_fmt = NamedStruct(column_headers_info, self.prefmt, 'ColumnHeaders')
        for _ in range(1, self.prod_desc.columns + 1):
            if self._buffer.read_int(4, self.endian, False) == USED_FLAG:
                self.column_headers.append(self._buffer.read_struct(column_headers_fmt))

        self.merged = 'SNDT' in (part.name for part in self.parts)

        self._sninfo = set()
        for irow, row_head in enumerate(self.row_headers):
            for icol, col_head in enumerate(self.column_headers):
                pointer = (self.prod_desc.data_block_ptr
                           + (irow * self.prod_desc.columns * self.prod_desc.parts)
                           + (icol * self.prod_desc.parts))

                self._buffer.jump_to(self._start, _word_to_position(pointer))
                data_ptr = self._buffer.read_int(4, self.endian, False)

                if data_ptr:
                    self._sninfo.add(
                        Sounding(
                            irow,
                            icol,
                            datetime.combine(row_head.DATE, row_head.TIME),
                            col_head.STID,
                            col_head.STNM,
                            col_head.SLAT,
                            col_head.SLON,
                            col_head.SELV,
                            col_head.STAT,
                            col_head.COUN,
                        )
                    )

    def sninfo(self):
        """Return sounding information."""
        return sorted(self._sninfo)

    def _unpack_merged(self, sndno):
        """Unpack merged sounding data."""
        soundings = []
        for irow, icol in sndno:
            row_head = self.row_headers[irow]
            col_head = self.column_headers[icol]
            sounding = {
                'STID': col_head.STID,
                'STNM': col_head.STNM,
                'SLAT': col_head.SLAT,
                'SLON': col_head.SLON,
                'SELV': col_head.SELV,
                'STAT': col_head.STAT,
                'COUN': col_head.COUN,
                'DATE': row_head.DATE,
                'TIME': row_head.TIME,
            }
            for iprt, part in enumerate(self.parts):
                pointer = (self.prod_desc.data_block_ptr
                           + (irow * self.prod_desc.columns * self.prod_desc.parts)
                           + (icol * self.prod_desc.parts + iprt))
                self._buffer.jump_to(self._start, _word_to_position(pointer))
                self.data_ptr = self._buffer.read_int(4, self.endian, False)
                if not self.data_ptr:
                    continue
                self._buffer.jump_to(self._start, _word_to_position(self.data_ptr))
                self.data_header_length = self._buffer.read_int(4, self.endian, False)
                data_header = self._buffer.set_mark()
                self._buffer.jump_to(data_header,
                                     _word_to_position(part.header_length + 1))
                lendat = self.data_header_length - part.header_length

                fmt_code = {
                    DataTypes.real: 'f',
                    DataTypes.realpack: 'i',
                    DataTypes.character: 's',
                }.get(part.data_type)

                if fmt_code is None:
                    raise NotImplementedError(
                        f'No methods for data type {part.data_type}'
                    )
                if fmt_code == 's':
                    lendat *= BYTES_PER_WORD

                packed_buffer = (
                    self._buffer.read_struct(
                        struct.Struct(f'{self.prefmt}{lendat}{fmt_code}')
                    )
                )

                parameters = self.parameters[iprt]
                nparms = len(parameters['name'])

                if part.data_type == DataTypes.realpack:
                    unpacked = self._unpack_real(packed_buffer, parameters, lendat)
                    for iprm, param in enumerate(parameters['name']):
                        sounding[param] = unpacked[iprm::nparms]
                else:
                    for iprm, param in enumerate(parameters['name']):
                        sounding[param] = np.array(
                            packed_buffer[iprm::nparms], dtype=np.float32
                        )

            soundings.append(sounding)
        return soundings

    def _unpack_unmerged(self, sndno):
        """Unpack unmerged sounding data."""
        soundings = []
        for irow, icol in sndno:
            row_head = self.row_headers[irow]
            col_head = self.column_headers[icol]
            sounding = {
                'STID': col_head.STID,
                'STNM': col_head.STNM,
                'SLAT': col_head.SLAT,
                'SLON': col_head.SLON,
                'SELV': col_head.SELV,
                'STAT': col_head.STAT,
                'COUN': col_head.COUN,
                'DATE': row_head.DATE,
                'TIME': row_head.TIME,
            }
            for iprt, part in enumerate(self.parts):
                pointer = (self.prod_desc.data_block_ptr
                           + (irow * self.prod_desc.columns * self.prod_desc.parts)
                           + (icol * self.prod_desc.parts + iprt))
                self._buffer.jump_to(self._start, _word_to_position(pointer))
                self.data_ptr = self._buffer.read_int(4, self.endian, False)
                if not self.data_ptr:
                    continue
                self._buffer.jump_to(self._start, _word_to_position(self.data_ptr))
                self.data_header_length = self._buffer.read_int(4, self.endian, False)
                data_header = self._buffer.set_mark()
                self._buffer.jump_to(data_header,
                                     _word_to_position(part.header_length + 1))
                lendat = self.data_header_length - part.header_length

                fmt_code = {
                    DataTypes.real: 'f',
                    DataTypes.realpack: 'i',
                    DataTypes.character: 's',
                }.get(part.data_type)

                if fmt_code is None:
                    raise NotImplementedError(
                        f'No methods for data type {part.data_type}'
                    )
                if fmt_code == 's':
                    lendat *= BYTES_PER_WORD

                packed_buffer = (
                    self._buffer.read_struct(
                        struct.Struct(f'{self.prefmt}{lendat}{fmt_code}')
                    )
                )

                parameters = self.parameters[iprt]
                nparms = len(parameters['name'])
                sounding[part.name] = {}

                if part.data_type == DataTypes.realpack:
                    unpacked = self._unpack_real(packed_buffer, parameters, lendat)
                    for iprm, param in enumerate(parameters['name']):
                        sounding[part.name][param] = unpacked[iprm::nparms]
                elif part.data_type == DataTypes.character:
                    for iprm, param in enumerate(parameters['name']):
                        sounding[part.name][param] = (
                            self._decode_strip(packed_buffer[iprm])
                        )
                else:
                    for iprm, param in enumerate(parameters['name']):
                        sounding[part.name][param] = (
                            np.array(packed_buffer[iprm::nparms], dtype=np.float32)
                        )

            soundings.append(self._merge_sounding(sounding))
        return soundings

    def _merge_sounding(self, parts):
        """Merge unmerged sounding data."""
        merged = {
            'STID': parts['STID'],
            'STNM': parts['STNM'],
            'SLAT': parts['SLAT'],
            'SLON': parts['SLON'],
            'SELV': parts['SELV'],
            'STAT': parts['STAT'],
            'COUN': parts['COUN'],
            'DATE': parts['DATE'],
            'TIME': parts['TIME'],
            'PRES': [],
            'HGHT': [],
            'TEMP': [],
            'DWPT': [],
            'DRCT': [],
            'SPED': [],
        }

        # Number of parameter levels
        num_man_levels = len(parts['TTAA']['PRES']) if 'TTAA' in parts else 0
        num_man_wind_levels = len(parts['PPAA']['PRES']) if 'PPAA' in parts else 0
        num_trop_levels = len(parts['TRPA']['PRES']) if 'TRPA' in parts else 0
        num_max_wind_levels = len(parts['MXWA']['PRES']) if 'MXWA' in parts else 0
        num_sigt_levels = len(parts['TTBB']['PRES']) if 'TTBB' in parts else 0
        num_sigw_levels = len(parts['PPBB']['SPED']) if 'PPBB' in parts else 0
        num_above_man_levels = len(parts['TTCC']['PRES']) if 'TTCC' in parts else 0
        num_above_trop_levels = len(parts['TRPC']['PRES']) if 'TRPC' in parts else 0
        num_above_max_wind_levels = len(parts['MXWC']['SPED']) if 'MXWC' in parts else 0
        num_above_sigt_levels = len(parts['TTDD']['PRES']) if 'TTDD' in parts else 0
        num_above_sigw_levels = len(parts['PPDD']['SPED']) if 'PPDD' in parts else 0
        num_above_man_wind_levels = len(parts['PPCC']['SPED']) if 'PPCC' in parts else 0

        total_data = (num_man_levels
                      + num_man_wind_levels
                      + num_trop_levels
                      + num_max_wind_levels
                      + num_sigt_levels
                      + num_sigw_levels
                      + num_above_man_levels
                      + num_above_trop_levels
                      + num_above_max_wind_levels
                      + num_above_sigt_levels
                      + num_above_sigw_levels
                      + num_above_man_wind_levels
                      )
        if total_data == 0:
            return None

        # Check SIG wind vertical coordinate
        # For some reason, the pressure data can get put into the
        # height array. Perhaps this is just a artifact of Python,
        # as GEMPAK itself just uses array indices without any
        # names involved. Since the first valid pressure of the
        # array will be negative in the case of pressure coordinates,
        # we can check for it and place data in the appropriate array.
        ppbb_is_z = True
        if num_sigw_levels:
            if 'PRES' in parts['PPBB']:
                ppbb_is_z = False
            else:
                for z in parts['PPBB']['HGHT']:
                    if z != self.prod_desc.missing_float and z < 0:
                        ppbb_is_z = False
                        parts['PPBB']['PRES'] = parts['PPBB']['HGHT']
                        break

        ppdd_is_z = True
        if num_above_sigw_levels:
            if 'PRES' in parts['PPDD']:
                ppdd_is_z = False
            else:
                for z in parts['PPDD']['HGHT']:
                    if z != self.prod_desc.missing_float and z < 0:
                        ppdd_is_z = False
                        parts['PPDD']['PRES'] = parts['PPDD']['HGHT']
                        break

        # Process surface data
        if num_man_levels < 1:
            merged['PRES'].append(self.prod_desc.missing_float)
            merged['HGHT'].append(self.prod_desc.missing_float)
            merged['TEMP'].append(self.prod_desc.missing_float)
            merged['DWPT'].append(self.prod_desc.missing_float)
            merged['DRCT'].append(self.prod_desc.missing_float)
            merged['SPED'].append(self.prod_desc.missing_float)
        else:
            merged['PRES'].append(parts['TTAA']['PRES'][0])
            merged['HGHT'].append(parts['TTAA']['HGHT'][0])
            merged['TEMP'].append(parts['TTAA']['TEMP'][0])
            merged['DWPT'].append(parts['TTAA']['DWPT'][0])
            merged['DRCT'].append(parts['TTAA']['DRCT'][0])
            merged['SPED'].append(parts['TTAA']['SPED'][0])

        merged['HGHT'][0] = merged['SELV']

        first_man_p = self.prod_desc.missing_float
        if num_man_levels >= 1:
            for mp, mt, mz in zip(parts['TTAA']['PRES'],
                                  parts['TTAA']['TEMP'],
                                  parts['TTAA']['HGHT']):
                if (mp != self.prod_desc.missing_float
                   and mt != self.prod_desc.missing_float
                   and mz != self.prod_desc.missing_float):
                    first_man_p = mp
                    break

        surface_p = merged['PRES'][0]
        if surface_p > 1060:
            surface_p = self.prod_desc.missing_float

        if (surface_p == self.prod_desc.missing_float
           or (surface_p < first_man_p
               and surface_p != self.prod_desc.missing_float)):
            merged['PRES'][0] = self.prod_desc.missing_float
            merged['HGHT'][0] = self.prod_desc.missing_float
            merged['TEMP'][0] = self.prod_desc.missing_float
            merged['DWPT'][0] = self.prod_desc.missing_float
            merged['DRCT'][0] = self.prod_desc.missing_float
            merged['SPED'][0] = self.prod_desc.missing_float

        if (num_sigt_levels >= 1
           and parts['TTBB']['PRES'][0] != self.prod_desc.missing_float
           and parts['TTBB']['TEMP'][0] != self.prod_desc.missing_float):
            first_man_p = merged['PRES'][0]
            first_sig_p = parts['TTBB']['PRES'][0]
            if (first_man_p == self.prod_desc.missing_float
               or np.isclose(first_man_p, first_sig_p)):
                merged['PRES'][0] = parts['TTBB']['PRES'][0]
                merged['DWPT'][0] = parts['TTBB']['DWPT'][0]
                merged['TEMP'][0] = parts['TTBB']['TEMP'][0]

        if num_sigw_levels >= 1:
            if ppbb_is_z:
                if (parts['PPBB']['HGHT'][0] == 0
                   and parts['PPBB']['DRCT'][0] != self.prod_desc.missing_float):
                    merged['DRCT'][0] = parts['PPBB']['DRCT'][0]
                    merged['SPED'][0] = parts['PPBB']['SPED'][0]
            else:
                if (parts['PPBB']['PRES'][0] != self.prod_desc.missing_float
                   and parts['PPBB']['DRCT'][0] != self.prod_desc.missing_float):
                    first_man_p = merged['PRES'][0]
                    first_sig_p = abs(parts['PPBB']['PRES'][0])
                    if (first_man_p == self.prod_desc.missing_float
                       or np.isclose(first_man_p, first_sig_p)):
                        merged['PRES'][0] = abs(parts['PPBB']['PRES'][0])
                        merged['DRCT'][0] = parts['PPBB']['DRCT'][0]
                        merged['SPED'][0] = parts['PPBB']['SPED'][0]

        # Merge MAN temperature
        bgl = 0
        qcman = []
        if num_man_levels >= 2 or num_above_man_levels >= 1:
            if merged['PRES'][0] == self.prod_desc.missing_float:
                plast = 2000
            else:
                plast = merged['PRES'][0]

        if num_man_levels >= 2:
            for i in range(1, num_man_levels):
                if (parts['TTAA']['PRES'][i] < plast
                   and parts['TTAA']['PRES'][i] != self.prod_desc.missing_float
                   and parts['TTAA']['TEMP'][i] != self.prod_desc.missing_float
                   and parts['TTAA']['HGHT'][i] != self.prod_desc.missing_float):
                    for pname, pval in parts['TTAA'].items():
                        merged[pname].append(pval[i])
                    plast = merged['PRES'][-1]
                else:
                    if parts['TTAA']['PRES'][i] > merged['PRES'][0]:
                        bgl += 1
                    else:
                        # GEMPAK ignores MAN data with missing TEMP/HGHT and does not
                        # interpolate for them.
                        if parts['TTAA']['PRES'][i] != self.prod_desc.missing_float:
                            qcman.append(parts['TTAA']['PRES'][i])

        if num_above_man_levels >= 1:
            for i in range(num_above_man_levels):
                if (parts['TTCC']['PRES'][i] < plast
                   and parts['TTCC']['PRES'][i] != self.prod_desc.missing_float
                   and parts['TTCC']['TEMP'][i] != self.prod_desc.missing_float
                   and parts['TTCC']['HGHT'][i] != self.prod_desc.missing_float):
                    for pname, pval in parts['TTCC'].items():
                        merged[pname].append(pval[i])
                    plast = merged['PRES'][-1]

        # Merge MAN wind
        if num_man_wind_levels >= 1 and num_man_levels >= 1 and len(merged['PRES']) >= 2:
            for iwind, pres in enumerate(parts['PPAA']['PRES']):
                if pres in merged['PRES'][1:]:
                    loc = merged['PRES'].index(pres)
                    if merged['DRCT'][loc] == self.prod_desc.missing_float:
                        merged['DRCT'][loc] = parts['PPAA']['DRCT'][iwind]
                        merged['SPED'][loc] = parts['PPAA']['SPED'][iwind]
                else:
                    if pres not in qcman:
                        size = len(merged['PRES'])
                        loc = size - bisect.bisect_left(merged['PRES'][1:][::-1], pres)
                        if loc >= size + 1:
                            loc = -1
                        merged['PRES'].insert(loc, pres)
                        merged['TEMP'].insert(loc, self.prod_desc.missing_float)
                        merged['DWPT'].insert(loc, self.prod_desc.missing_float)
                        merged['DRCT'].insert(loc, parts['PPAA']['DRCT'][iwind])
                        merged['SPED'].insert(loc, parts['PPAA']['SPED'][iwind])
                        merged['HGHT'].insert(loc, self.prod_desc.missing_float)

        if num_above_man_wind_levels >= 1 and num_man_levels >= 1 and len(merged['PRES']) >= 2:
            for iwind, pres in enumerate(parts['PPCC']['PRES']):
                if pres in merged['PRES'][1:]:
                    loc = merged['PRES'].index(pres)
                    if merged['DRCT'][loc] == self.prod_desc.missing_float:
                        merged['DRCT'][loc] = parts['PPCC']['DRCT'][iwind]
                        merged['SPED'][loc] = parts['PPCC']['SPED'][iwind]
                else:
                    if pres not in qcman:
                        size = len(merged['PRES'])
                        loc = size - bisect.bisect_left(merged['PRES'][1:][::-1], pres)
                        if loc >= size + 1:
                            loc = -1
                        merged['PRES'].insert(loc, pres)
                        merged['TEMP'].insert(loc, self.prod_desc.missing_float)
                        merged['DWPT'].insert(loc, self.prod_desc.missing_float)
                        merged['DRCT'].insert(loc, parts['PPCC']['DRCT'][iwind])
                        merged['SPED'].insert(loc, parts['PPCC']['SPED'][iwind])
                        merged['HGHT'].insert(loc, self.prod_desc.missing_float)

        # Merge TROP
        if num_trop_levels >= 1 or num_above_trop_levels >= 1:
            if merged['PRES'][0] != self.prod_desc.missing_float:
                pbot = merged['PRES'][0]
            elif len(merged['PRES']) > 1:
                pbot = merged['PRES'][1]
                if pbot < parts['TRPA']['PRES'][1]:
                    pbot = 1050
            else:
                pbot = 1050

        if num_trop_levels >= 1:
            for itrp, pres in enumerate(parts['TRPA']['PRES']):
                pres = abs(pres)
                if (pres != self.prod_desc.missing_float
                   and parts['TRPA']['TEMP'][itrp] != self.prod_desc.missing_float
                   and pres != 0):
                    if pres > pbot:
                        continue
                    elif pres in merged['PRES']:
                        ploc = merged['PRES'].index(pres)
                        if merged['TEMP'][ploc] == self.prod_desc.missing_float:
                            merged['TEMP'][ploc] = parts['TRPA']['TEMP'][itrp]
                            merged['DWPT'][ploc] = parts['TRPA']['DWPT'][itrp]
                        if merged['DRCT'][ploc] == self.prod_desc.missing_float:
                            merged['DRCT'][ploc] = parts['TRPA']['DRCT'][itrp]
                            merged['SPED'][ploc] = parts['TRPA']['SPED'][itrp]
                        merged['HGHT'][ploc] = self.prod_desc.missing_float
                    else:
                        size = len(merged['PRES'])
                        loc = size - bisect.bisect_left(merged['PRES'][::-1], pres)
                        merged['PRES'].insert(loc, pres)
                        merged['TEMP'].insert(loc, parts['TRPA']['TEMP'][itrp])
                        merged['DWPT'].insert(loc, parts['TRPA']['DWPT'][itrp])
                        merged['DRCT'].insert(loc, parts['TRPA']['DRCT'][itrp])
                        merged['SPED'].insert(loc, parts['TRPA']['SPED'][itrp])
                        merged['HGHT'].insert(loc, self.prod_desc.missing_float)
                pbot = pres

        if num_above_trop_levels >= 1:
            for itrp, pres in enumerate(parts['TRPC']['PRES']):
                pres = abs(pres)
                if (pres != self.prod_desc.missing_float
                   and parts['TRPC']['TEMP'][itrp] != self.prod_desc.missing_float
                   and pres != 0):
                    if pres > pbot:
                        continue
                    elif pres in merged['PRES']:
                        ploc = merged['PRES'].index(pres)
                        if merged['TEMP'][ploc] == self.prod_desc.missing_float:
                            merged['TEMP'][ploc] = parts['TRPC']['TEMP'][itrp]
                            merged['DWPT'][ploc] = parts['TRPC']['DWPT'][itrp]
                        if merged['DRCT'][ploc] == self.prod_desc.missing_float:
                            merged['DRCT'][ploc] = parts['TRPC']['DRCT'][itrp]
                            merged['SPED'][ploc] = parts['TRPC']['SPED'][itrp]
                        merged['HGHT'][ploc] = self.prod_desc.missing_float
                    else:
                        size = len(merged['PRES'])
                        loc = size - bisect.bisect_left(merged['PRES'][::-1], pres)
                        merged['PRES'].insert(loc, pres)
                        merged['TEMP'].insert(loc, parts['TRPC']['TEMP'][itrp])
                        merged['DWPT'].insert(loc, parts['TRPC']['DWPT'][itrp])
                        merged['DRCT'].insert(loc, parts['TRPC']['DRCT'][itrp])
                        merged['SPED'].insert(loc, parts['TRPC']['SPED'][itrp])
                        merged['HGHT'].insert(loc, self.prod_desc.missing_float)
                pbot = pres

        # Merge SIG temperature
        if num_sigt_levels >= 1 or num_above_sigt_levels >= 1:
            if merged['PRES'][0] != self.prod_desc.missing_float:
                pbot = merged['PRES'][0]
            elif len(merged['PRES']) > 1:
                pbot = merged['PRES'][1]
                if pbot < parts['TTBB']['PRES'][1]:
                    pbot = 1050
            else:
                pbot = 1050

        if num_sigt_levels >= 1:
            for isigt, pres in enumerate(parts['TTBB']['PRES']):
                pres = abs(pres)
                if (pres != self.prod_desc.missing_float
                   and parts['TTBB']['TEMP'][isigt] != self.prod_desc.missing_float
                   and pres != 0):
                    if pres > pbot:
                        continue
                    elif pres in merged['PRES']:
                        ploc = merged['PRES'].index(pres)
                        if merged['TEMP'][ploc] == self.prod_desc.missing_float:
                            merged['TEMP'][ploc] = parts['TTBB']['TEMP'][isigt]
                            merged['DWPT'][ploc] = parts['TTBB']['DWPT'][isigt]
                    else:
                        size = len(merged['PRES'])
                        loc = size - bisect.bisect_left(merged['PRES'][::-1], pres)
                        merged['PRES'].insert(loc, pres)
                        merged['TEMP'].insert(loc, parts['TTBB']['TEMP'][isigt])
                        merged['DWPT'].insert(loc, parts['TTBB']['DWPT'][isigt])
                        merged['DRCT'].insert(loc, self.prod_desc.missing_float)
                        merged['SPED'].insert(loc, self.prod_desc.missing_float)
                        merged['HGHT'].insert(loc, self.prod_desc.missing_float)
                pbot = pres

        if num_above_sigt_levels >= 1:
            for isigt, pres in enumerate(parts['TTDD']['PRES']):
                pres = abs(pres)
                if (pres != self.prod_desc.missing_float
                   and parts['TTDD']['TEMP'][isigt] != self.prod_desc.missing_float
                   and pres != 0):
                    if pres > pbot:
                        continue
                    elif pres in merged['PRES']:
                        ploc = merged['PRES'].index(pres)
                        if merged['TEMP'][ploc] == self.prod_desc.missing_float:
                            merged['TEMP'][ploc] = parts['TTDD']['TEMP'][isigt]
                            merged['DWPT'][ploc] = parts['TTDD']['DWPT'][isigt]
                    else:
                        size = len(merged['PRES'])
                        loc = size - bisect.bisect_left(merged['PRES'][::-1], pres)
                        merged['PRES'].insert(loc, pres)
                        merged['TEMP'].insert(loc, parts['TTDD']['TEMP'][isigt])
                        merged['DWPT'].insert(loc, parts['TTDD']['DWPT'][isigt])
                        merged['DRCT'].insert(loc, self.prod_desc.missing_float)
                        merged['SPED'].insert(loc, self.prod_desc.missing_float)
                        merged['HGHT'].insert(loc, self.prod_desc.missing_float)
                pbot = pres

        # Interpolate heights
        interp_moist_height(merged, self.prod_desc.missing_float)

        # Merge SIG winds on pressure surfaces
        if not ppbb_is_z or not ppdd_is_z:
            if num_sigw_levels >= 1 or num_above_sigw_levels >= 1:
                if merged['PRES'][0] != self.prod_desc.missing_float:
                    pbot = merged['PRES'][0]
                elif len(merged['PRES']) > 1:
                    pbot = merged['PRES'][1]
                else:
                    pbot = 0

            if num_sigw_levels >= 1 and not ppbb_is_z:
                for isigw, pres in enumerate(parts['PPBB']['PRES']):
                    pres = abs(pres)
                    if (pres != self.prod_desc.missing_float
                       and parts['PPBB']['DRCT'][isigw] != self.prod_desc.missing_float
                       and parts['PPBB']['SPED'][isigw] != self.prod_desc.missing_float
                       and pres != 0):
                        if pres > pbot:
                            continue
                        elif pres in merged['PRES']:
                            ploc = merged['PRES'].index(pres)
                            if (merged['DRCT'][ploc] == self.prod_desc.missing_float
                               or merged['SPED'][ploc] == self.prod_desc.missing_float):
                                merged['DRCT'][ploc] = parts['PPBB']['DRCT'][isigw]
                                merged['SPED'][ploc] = parts['PPBB']['SPED'][isigw]
                        else:
                            size = len(merged['PRES'])
                            loc = size - bisect.bisect_left(merged['PRES'][::-1], pres)
                            merged['PRES'].insert(loc, pres)
                            merged['DRCT'].insert(loc, parts['PPBB']['DRCT'][isigw])
                            merged['SPED'].insert(loc, parts['PPBB']['SPED'][isigw])
                            merged['TEMP'].insert(loc, self.prod_desc.missing_float)
                            merged['DWPT'].insert(loc, self.prod_desc.missing_float)
                            merged['HGHT'].insert(loc, self.prod_desc.missing_float)
                    pbot = pres

            if num_above_sigw_levels >= 1 and not ppdd_is_z:
                for isigw, pres in enumerate(parts['PPDD']['PRES']):
                    pres = abs(pres)
                    if (pres != self.prod_desc.missing_float
                       and parts['PPDD']['DRCT'][isigw] != self.prod_desc.missing_float
                       and parts['PPDD']['SPED'][isigw] != self.prod_desc.missing_float
                       and pres != 0):
                        if pres > pbot:
                            continue
                        elif pres in merged['PRES']:
                            ploc = merged['PRES'].index(pres)
                            if (merged['DRCT'][ploc] == self.prod_desc.missing_float
                               or merged['SPED'][ploc] == self.prod_desc.missing_float):
                                merged['DRCT'][ploc] = parts['PPDD']['DRCT'][isigw]
                                merged['SPED'][ploc] = parts['PPDD']['SPED'][isigw]
                        else:
                            size = len(merged['PRES'])
                            loc = size - bisect.bisect_left(merged['PRES'][::-1], pres)
                            merged['PRES'].insert(loc, pres)
                            merged['DRCT'].insert(loc, parts['PPDD']['DRCT'][isigw])
                            merged['SPED'].insert(loc, parts['PPDD']['SPED'][isigw])
                            merged['TEMP'].insert(loc, self.prod_desc.missing_float)
                            merged['DWPT'].insert(loc, self.prod_desc.missing_float)
                            merged['HGHT'].insert(loc, self.prod_desc.missing_float)
                    pbot = pres

        # Merge max winds on pressure surfaces
        if num_max_wind_levels >= 1 or num_above_max_wind_levels >= 1:
            if merged['PRES'][0] != self.prod_desc.missing_float:
                pbot = merged['PRES'][0]
            elif len(merged['PRES']) > 1:
                pbot = merged['PRES'][1]
            else:
                pbot = 0

        if num_max_wind_levels >= 1:
            for imxw, pres in enumerate(parts['MXWA']['PRES']):
                pres = abs(pres)
                if (pres != self.prod_desc.missing_float
                   and parts['MXWA']['DRCT'][imxw] != self.prod_desc.missing_float
                   and parts['MXWA']['SPED'][imxw] != self.prod_desc.missing_float
                   and pres != 0):
                    if pres > pbot:
                        continue
                    elif pres in merged['PRES']:
                        ploc = merged['PRES'].index(pres)
                        if (merged['DRCT'][ploc] == self.prod_desc.missing_float
                           or merged['SPED'][ploc] == self.prod_desc.missing_float):
                            merged['DRCT'][ploc] = parts['MXWA']['DRCT'][imxw]
                            merged['SPED'][ploc] = parts['MXWA']['SPED'][imxw]
                    else:
                        size = len(merged['PRES'])
                        loc = size - bisect.bisect_left(merged['PRES'][::-1], pres)
                        merged['PRES'].insert(loc, pres)
                        merged['DRCT'].insert(loc, parts['MXWA']['DRCT'][imxw])
                        merged['SPED'].insert(loc, parts['MXWA']['SPED'][imxw])
                        merged['TEMP'].insert(loc, self.prod_desc.missing_float)
                        merged['DWPT'].insert(loc, self.prod_desc.missing_float)
                        merged['HGHT'].insert(loc, self.prod_desc.missing_float)
                pbot = pres

        if num_above_max_wind_levels >= 1:
            for imxw, pres in enumerate(parts['MXWC']['PRES']):
                pres = abs(pres)
                if (pres != self.prod_desc.missing_float
                   and parts['MXWC']['DRCT'][imxw] != self.prod_desc.missing_float
                   and parts['MXWC']['SPED'][imxw] != self.prod_desc.missing_float
                   and pres != 0):
                    if pres > pbot:
                        continue
                    elif pres in merged['PRES']:
                        ploc = merged['PRES'].index(pres)
                        if (merged['DRCT'][ploc] == self.prod_desc.missing_float
                           or merged['SPED'][ploc] == self.prod_desc.missing_float):
                            merged['DRCT'][ploc] = parts['MXWC']['DRCT'][imxw]
                            merged['SPED'][ploc] = parts['MXWC']['SPED'][imxw]
                    else:
                        size = len(merged['PRES'])
                        loc = size - bisect.bisect_left(merged['PRES'][::-1], pres)
                        merged['PRES'].insert(loc, pres)
                        merged['DRCT'].insert(loc, parts['MXWC']['DRCT'][imxw])
                        merged['SPED'].insert(loc, parts['MXWC']['SPED'][imxw])
                        merged['TEMP'].insert(loc, self.prod_desc.missing_float)
                        merged['DWPT'].insert(loc, self.prod_desc.missing_float)
                        merged['HGHT'].insert(loc, self.prod_desc.missing_float)
                pbot = pres

        # Interpolate height for SIG/MAX winds
        interp_logp_height(merged, self.prod_desc.missing_float)

        # Merge SIG winds on height surfaces
        if ppbb_is_z or ppdd_is_z:
            nsgw = num_sigw_levels if ppbb_is_z else 0
            nasw = num_above_sigw_levels if ppdd_is_z else 0
            if (nsgw >= 1 and (parts['PPBB']['HGHT'][0] == 0
               or parts['PPBB']['HGHT'][0] == merged['HGHT'][0])):
                istart = 1
            else:
                istart = 0

            size = len(merged['HGHT'])
            psfc = merged['PRES'][0]
            zsfc = merged['HGHT'][0]

            if (size >= 2 and psfc != self.prod_desc.missing_float
               and zsfc != self.prod_desc.missing_float):
                more = True
                zold = merged['HGHT'][0]
                znxt = merged['HGHT'][1]
                ilev = 1
            elif size >= 3:
                more = True
                zold = merged['HGHT'][1]
                znxt = merged['HGHT'][2]
                ilev = 2
            else:
                zold = self.prod_desc.missing_float
                znxt = self.prod_desc.missing_float

            if (zold == self.prod_desc.missing_float
               or znxt == self.prod_desc.missing_float):
                more = False

            if istart <= nsgw:
                above = False
                i = istart
                iend = nsgw
            else:
                above = True
                i = 0
                iend = nasw

            while more and i < iend:
                if not above:
                    hght = parts['PPBB']['HGHT'][i]
                    drct = parts['PPBB']['DRCT'][i]
                    sped = parts['PPBB']['SPED'][i]
                else:
                    hght = parts['PPDD']['HGHT'][i]
                    drct = parts['PPDD']['DRCT'][i]
                    sped = parts['PPDD']['SPED'][i]
                skip = False

                if ((hght == self.prod_desc.missing_float
                   and drct == self.prod_desc.missing_float
                   and sped == self.prod_desc.missing_float)
                   or hght <= zold):
                    skip = True
                elif abs(zold - hght) < 1:
                    skip = True
                    if (merged['DRCT'][ilev - 1] == self.prod_desc.missing_float
                       or merged['SPED'][ilev - 1] == self.prod_desc.missing_float):
                        merged['DRCT'][ilev - 1] = drct
                        merged['SPED'][ilev - 1] = sped
                elif hght >= znxt:
                    while more and hght > znxt:
                        zold = znxt
                        ilev += 1
                        if ilev >= size:
                            more = False
                        else:
                            znxt = merged['HGHT'][ilev]
                            if znxt == self.prod_desc.missing_float:
                                more = False

                if more and not skip:
                    if abs(znxt - hght) < 1:
                        if (merged['DRCT'][ilev - 1] == self.prod_desc.missing_float
                           or merged['SPED'][ilev - 1] == self.prod_desc.missing_float):
                            merged['DRCT'][ilev] = drct
                            merged['SPED'][ilev] = sped
                    else:
                        loc = bisect.bisect_left(merged['HGHT'], hght)
                        merged['HGHT'].insert(loc, hght)
                        merged['DRCT'].insert(loc, drct)
                        merged['SPED'].insert(loc, sped)
                        merged['PRES'].insert(loc, self.prod_desc.missing_float)
                        merged['TEMP'].insert(loc, self.prod_desc.missing_float)
                        merged['DWPT'].insert(loc, self.prod_desc.missing_float)
                        size += 1
                        ilev += 1
                        zold = hght

                if not above and i == nsgw - 1:
                    above = True
                    i = 0
                    iend = nasw
                else:
                    i += 1

            # Interpolate misssing pressure with height
            interp_logp_pressure(merged, self.prod_desc.missing_float)

        # Interpolate missing data
        interp_missing_data(merged, self.prod_desc.missing_float)

        # Add below ground MAN data
        if merged['PRES'][0] != self.prod_desc.missing_float and bgl > 0:
            for ibgl in range(1, num_man_levels):
                pres = parts['TTAA']['PRES'][ibgl]
                if pres > merged['PRES'][0]:
                    loc = size - bisect.bisect_left(merged['PRES'][1:][::-1], pres)
                    merged['PRES'].insert(loc, pres)
                    merged['TEMP'].insert(loc, parts['TTAA']['TEMP'][ibgl])
                    merged['DWPT'].insert(loc, parts['TTAA']['DWPT'][ibgl])
                    merged['DRCT'].insert(loc, parts['TTAA']['DRCT'][ibgl])
                    merged['SPED'].insert(loc, parts['TTAA']['SPED'][ibgl])
                    merged['HGHT'].insert(loc, parts['TTAA']['HGHT'][ibgl])
                    size += 1

        # Add text data, if it is included
        if 'TXTA' in parts:
            merged['TXTA'] = parts['TXTA']['TEXT']
        if 'TXTB' in parts:
            merged['TXTB'] = parts['TXTB']['TEXT']
        if 'TXTC' in parts:
            merged['TXTC'] = parts['TXTC']['TEXT']
        if 'TXPB' in parts:
            merged['TXPB'] = parts['TXPB']['TEXT']

        return merged

    def snxarray(self, station_id=None, station_number=None,
                 date_time=None, state=None, country=None, bbox=None):
        """Select soundings and output as list of xarray Datasets.

        Subset the data by parameter values. The default is to not
        subset and return the entire dataset.

        Parameters
        ----------
        station_id : str or array-like of str
            Station ID of sounding site.

        station_number : int or array-like of int
            Station number of sounding site.

        date_time : datetime or array-like of datetime
            Datetime of the sounding. Alternatively a string with
            the format YYYYmmddHHMM or first|FIRST or last|LAST
            which function to retrieve the latest and oldest
            time within the file, respectively.

        state : str or array-like of str
            State where sounding site is located.

        country : str or array-like of str
            Country where sounding site is located.

        bbox: floats (left, right, bottom, top)
            Bounding area where sounding sites are located.

        Returns
        -------
        list
            List of xarray.Dataset objects for each sounding.
        """
        if station_id is not None:
            if (not isinstance(station_id, Iterable)
               or isinstance(station_id, str)):
                station_id = [station_id]
            station_id = [c.upper() for c in station_id]

        if station_number is not None:
            if not isinstance(station_number, Iterable):
                station_number = [station_number]
            station_number = [int(sn) for sn in station_number]

        if date_time is not None:
            if (not isinstance(date_time, Iterable)
               or (isinstance(date_time, str)
               and date_time not in ['first', 'FIRST', 'last', 'LAST'])):
                date_time = [date_time]
            if date_time not in ['first', 'FIRST', 'last', 'LAST']:
                for i, dt in enumerate(date_time):
                    if isinstance(dt, str):
                        date_time[i] = datetime.strptime(dt, '%Y%m%d%H%M')

        if (state is not None
           and (not isinstance(state, Iterable)
                or isinstance(state, str))):
            state = [state]
            state = [s.upper() for s in state]

        if (country is not None
           and (not isinstance(country, Iterable)
                or isinstance(country, str))):
            country = [country]
            country = [c.upper() for c in country]

        # Figure out which columns to extract from the file
        matched = sorted(self._sninfo)

        # Do this now or the matched filter iterator will be consumed
        # prematurely.
        if date_time in ['last', 'LAST']:
            date_time = [max((d.DATTIM for d in matched))]
        elif date_time in ['first', 'FIRST']:
            date_time = [min((d.DATTIM for d in matched))]

        if country is not None:
            matched = filter(
                lambda snd: snd.COUNTRY in country,
                matched
            )

        if state is not None:
            matched = filter(
                lambda snd: snd.STATE in state,
                matched
            )

        if bbox is not None:
            matched = filter(
                lambda snd: _bbox_filter(snd.LAT, snd.LON, bbox),
                matched
            )

        if date_time is not None:
            matched = filter(
                lambda snd: snd.DATTIM in date_time,
                matched
            )

        if station_id is not None:
            matched = filter(
                lambda snd: snd.ID in station_id,
                matched
            )

        if station_number is not None:
            matched = filter(
                lambda snd: snd.NUMBER in station_number,
                matched
            )

        matched = list(matched)

        if len(matched) < 1:
            raise KeyError('No stations were matched with given parameters.')

        sndno = [(s.DTNO, s.SNDNO) for s in matched]

        if self.merged:
            data = self._unpack_merged(sndno)
        else:
            data = self._unpack_unmerged(sndno)

        soundings = []
        for snd in data:
            if snd is None:
                continue
            wmo_text = {}
            attrs = {
                'station_id': snd.pop('STID'),
                'station_number': snd.pop('STNM'),
                'lat': snd.pop('SLAT'),
                'lon': snd.pop('SLON'),
                'elevation': snd.pop('SELV'),
                'state': snd.pop('STAT'),
                'country': snd.pop('COUN'),
            }

            if 'PRES' in snd:
                vcoord = 'pres'
                attrs['station_pressure'] = snd['PRES'][0]
            elif 'HGHT' in snd:
                vcoord = 'hght'
            else:
                raise ValueError('Unknown vertical coordinate in sounding.')

            if 'TXTA' in snd:
                wmo_text['txta'] = snd.pop('TXTA')
            if 'TXTB' in snd:
                wmo_text['txtb'] = snd.pop('TXTB')
            if 'TXTC' in snd:
                wmo_text['txtc'] = snd.pop('TXTC')
            if 'TXPB' in snd:
                wmo_text['txpb'] = snd.pop('TXPB')
            if wmo_text:
                attrs['wmo_codes'] = wmo_text

            dt = datetime.combine(snd.pop('DATE'), snd.pop('TIME'))
            if vcoord == 'pres':
                vcdata = np.array(snd.pop('PRES'))
            elif vcoord == 'hght':
                vcdata = np.array(snd.pop('HGHT'))

            var = {}
            for param, values in snd.items():
                values = np.array(values)[np.newaxis, ...]
                maskval = np.ma.array(values, mask=values == self.prod_desc.missing_float,
                                      dtype=np.float32)
                var[param.lower()] = (['time', 'pres'], maskval)

            xrds = xr.Dataset(var,
                              coords={'time': np.atleast_1d(dt), vcoord: vcdata},
                              attrs=attrs)

            # Sort to fix GEMPAK surface data at first level
            if vcoord == 'pres':
                xrds = xrds.sortby('pres', ascending=False)
            elif vcoord == 'hght':
                xrds = xrds.sortby('hght', ascending=True)

            soundings.append(xrds)
        return soundings


class GempakSurface(GempakFile):
    """Subclass of GempakFile specific to GEMPAK surface data."""

    def __init__(self, file, *args, **kwargs):
        """Instantiate GempakSurface object from file."""
        super().__init__(file)

        # Row Headers
        self._buffer.jump_to(self._start, _word_to_position(self.prod_desc.row_headers_ptr))
        self.row_headers = []
        row_headers_info = self._key_types(self.row_keys)
        row_headers_info.extend([(None, None)])
        row_headers_fmt = NamedStruct(row_headers_info, self.prefmt, 'RowHeaders')
        for _ in range(1, self.prod_desc.rows + 1):
            if self._buffer.read_int(4, self.endian, False) == USED_FLAG:
                self.row_headers.append(self._buffer.read_struct(row_headers_fmt))

        # Column Headers
        self._buffer.jump_to(self._start, _word_to_position(self.prod_desc.column_headers_ptr))
        self.column_headers = []
        column_headers_info = self._key_types(self.column_keys)
        column_headers_info.extend([(None, None)])
        column_headers_fmt = NamedStruct(column_headers_info, self.prefmt, 'ColumnHeaders')
        for _ in range(1, self.prod_desc.columns + 1):
            if self._buffer.read_int(4, self.endian, False) == USED_FLAG:
                self.column_headers.append(self._buffer.read_struct(column_headers_fmt))

        self._get_surface_type()

        self._sfinfo = set()
        if self.surface_type == 'standard':
            for irow, row_head in enumerate(self.row_headers):
                for icol, col_head in enumerate(self.column_headers):
                    for iprt in range(len(self.parts)):
                        pointer = (self.prod_desc.data_block_ptr
                                   + (irow * self.prod_desc.columns * self.prod_desc.parts)
                                   + (icol * self.prod_desc.parts + iprt))

                        self._buffer.jump_to(self._start, _word_to_position(pointer))
                        data_ptr = self._buffer.read_int(4, self.endian, False)

                        if data_ptr:
                            self._sfinfo.add(
                                Surface(
                                    irow,
                                    icol,
                                    datetime.combine(row_head.DATE, row_head.TIME),
                                    col_head.STID + col_head.STD2,
                                    col_head.STNM,
                                    col_head.SLAT,
                                    col_head.SLON,
                                    col_head.SELV,
                                    col_head.STAT,
                                    col_head.COUN,
                                )
                            )
        elif self.surface_type == 'ship':
            irow = 0
            for icol, col_head in enumerate(self.column_headers):
                for iprt in range(len(self.parts)):
                    pointer = (self.prod_desc.data_block_ptr
                               + (irow * self.prod_desc.columns * self.prod_desc.parts)
                               + (icol * self.prod_desc.parts + iprt))

                    self._buffer.jump_to(self._start, _word_to_position(pointer))
                    data_ptr = self._buffer.read_int(4, self.endian, False)

                    if data_ptr:
                        self._sfinfo.add(
                            Surface(
                                irow,
                                icol,
                                datetime.combine(col_head.DATE, col_head.TIME),
                                col_head.STID + col_head.STD2,
                                col_head.STNM,
                                col_head.SLAT,
                                col_head.SLON,
                                col_head.SELV,
                                col_head.STAT,
                                col_head.COUN,
                            )
                        )
        elif self.surface_type == 'climate':
            for icol, col_head in enumerate(self.column_headers):
                for irow, row_head in enumerate(self.row_headers):
                    for iprt in range(len(self.parts)):
                        pointer = (self.prod_desc.data_block_ptr
                                   + (irow * self.prod_desc.columns * self.prod_desc.parts)
                                   + (icol * self.prod_desc.parts + iprt))

                        self._buffer.jump_to(self._start, _word_to_position(pointer))
                        data_ptr = self._buffer.read_int(4, self.endian, False)

                        if data_ptr:
                            self._sfinfo.add(
                                Surface(
                                    irow,
                                    icol,
                                    datetime.combine(col_head.DATE, col_head.TIME),
                                    row_head.STID + row_head.STD2,
                                    row_head.STNM,
                                    row_head.SLAT,
                                    row_head.SLON,
                                    row_head.SELV,
                                    row_head.STAT,
                                    row_head.COUN,
                                )
                            )
        else:
            raise TypeError(f'Unknown surface type {self.surface_type}')

    def sfinfo(self):
        """Return station information."""
        return sorted(self._sfinfo)

    def _get_surface_type(self):
        """Determine type of surface file."""
        if len(self.row_headers) == 1:
            self.surface_type = 'ship'
        elif 'DATE' in self.row_keys:
            self.surface_type = 'standard'
        elif 'DATE' in self.column_keys:
            self.surface_type = 'climate'
        else:
            raise TypeError('Unknown surface data type')

    def _key_types(self, keys):
        """Determine header information from a set of keys."""
        return [(key, '4s', self._decode_strip) if key == 'STID'
                else (key, 'i') if key == 'STNM'
                else (key, 'i', lambda x: x / 100) if key == 'SLAT'
                else (key, 'i', lambda x: x / 100) if key == 'SLON'
                else (key, 'i') if key == 'SELV'
                else (key, '4s', self._decode_strip) if key == 'STAT'
                else (key, '4s', self._decode_strip) if key == 'COUN'
                else (key, '4s', self._decode_strip) if key == 'STD2'
                else (key, 'i', self._make_date) if key == 'DATE'
                else (key, 'i', self._make_time) if key == 'TIME'
                else (key, 'i')
                for key in keys]

    def _unpack_climate(self, sfcno):
        """Unpack a climate surface data file."""
        stations = []
        for irow, icol in sfcno:
            col_head = self.column_headers[icol]
            row_head = self.row_headers[irow]
            station = {
                'STID': row_head.STID,
                'STNM': row_head.STNM,
                'SLAT': row_head.SLAT,
                'SLON': row_head.SLON,
                'SELV': row_head.SELV,
                'STAT': row_head.STAT,
                'COUN': row_head.COUN,
                'STD2': row_head.STD2,
                'SPRI': row_head.SPRI,
                'DATE': col_head.DATE,
                'TIME': col_head.TIME,
            }
            for iprt, part in enumerate(self.parts):
                pointer = (self.prod_desc.data_block_ptr
                           + (irow * self.prod_desc.columns * self.prod_desc.parts)
                           + (icol * self.prod_desc.parts + iprt))
                self._buffer.jump_to(self._start, _word_to_position(pointer))
                self.data_ptr = self._buffer.read_int(4, self.endian, False)
                if not self.data_ptr:
                    continue
                self._buffer.jump_to(self._start, _word_to_position(self.data_ptr))
                self.data_header_length = self._buffer.read_int(4, self.endian, False)
                data_header = self._buffer.set_mark()
                self._buffer.jump_to(data_header,
                                     _word_to_position(part.header_length + 1))
                lendat = self.data_header_length - part.header_length

                fmt_code = {
                    DataTypes.real: 'f',
                    DataTypes.realpack: 'i',
                    DataTypes.character: 's',
                }.get(part.data_type)

                if fmt_code is None:
                    raise NotImplementedError(
                        f'No methods for data type {part.data_type}'
                    )
                if fmt_code == 's':
                    lendat *= BYTES_PER_WORD

                packed_buffer = (
                    self._buffer.read_struct(
                        struct.Struct(f'{self.prefmt}{lendat}{fmt_code}')
                    )
                )

                parameters = self.parameters[iprt]

                if part.data_type == DataTypes.realpack:
                    unpacked = self._unpack_real(packed_buffer, parameters, lendat)
                    for iprm, param in enumerate(parameters['name']):
                        station[param] = unpacked[iprm]
                elif part.data_type == DataTypes.character:
                    for iprm, param in enumerate(parameters['name']):
                        station[param] = self._decode_strip(packed_buffer[iprm])
                else:
                    for iprm, param in enumerate(parameters['name']):
                        station[param] = np.array(
                            packed_buffer[iprm], dtype=np.float32
                        )

            stations.append(station)
        return stations

    def _unpack_ship(self, sfcno):
        """Unpack ship (moving observation) surface data file."""
        stations = []
        for irow, icol in sfcno:  # irow should always be zero
            col_head = self.column_headers[icol]
            station = {
                'STID': col_head.STID,
                'STNM': col_head.STNM,
                'SLAT': col_head.SLAT,
                'SLON': col_head.SLON,
                'SELV': col_head.SELV,
                'STAT': col_head.STAT,
                'COUN': col_head.COUN,
                'STD2': col_head.STD2,
                'SPRI': col_head.SPRI,
                'DATE': col_head.DATE,
                'TIME': col_head.TIME,
            }
            for iprt, part in enumerate(self.parts):
                pointer = (self.prod_desc.data_block_ptr
                           + (irow * self.prod_desc.columns * self.prod_desc.parts)
                           + (icol * self.prod_desc.parts + iprt))
                self._buffer.jump_to(self._start, _word_to_position(pointer))
                self.data_ptr = self._buffer.read_int(4, self.endian, False)
                if not self.data_ptr:
                    continue
                self._buffer.jump_to(self._start, _word_to_position(self.data_ptr))
                self.data_header_length = self._buffer.read_int(4, self.endian, False)
                data_header = self._buffer.set_mark()
                self._buffer.jump_to(data_header,
                                     _word_to_position(part.header_length + 1))
                lendat = self.data_header_length - part.header_length

                fmt_code = {
                    DataTypes.real: 'f',
                    DataTypes.realpack: 'i',
                    DataTypes.character: 's',
                }.get(part.data_type)

                if fmt_code is None:
                    raise NotImplementedError(
                        f'No methods for data type {part.data_type}'
                    )
                if fmt_code == 's':
                    lendat *= BYTES_PER_WORD

                packed_buffer = (
                    self._buffer.read_struct(
                        struct.Struct(f'{self.prefmt}{lendat}{fmt_code}')
                    )
                )

                parameters = self.parameters[iprt]

                if part.data_type == DataTypes.realpack:
                    unpacked = self._unpack_real(packed_buffer, parameters, lendat)
                    for iprm, param in enumerate(parameters['name']):
                        station[param] = unpacked[iprm]
                elif part.data_type == DataTypes.character:
                    for iprm, param in enumerate(parameters['name']):
                        station[param] = self._decode_strip(packed_buffer[iprm])
                else:
                    for iprm, param in enumerate(parameters['name']):
                        station[param] = np.array(
                            packed_buffer[iprm], dtype=np.float32
                        )

            stations.append(station)
        return stations

    def _unpack_standard(self, sfcno):
        """Unpack a standard surface data file."""
        reports = []
        for irow, icol in sfcno:
            row_head = self.row_headers[irow]
            col_head = self.column_headers[icol]
            report = {
                'STID': col_head.STID,
                'STNM': col_head.STNM,
                'SLAT': col_head.SLAT,
                'SLON': col_head.SLON,
                'SELV': col_head.SELV,
                'STAT': col_head.STAT,
                'COUN': col_head.COUN,
                'STD2': col_head.STD2,
                'SPRI': col_head.SPRI,
                'DATE': row_head.DATE,
                'TIME': row_head.TIME,
            }
            values = {}
            for iprt, part in enumerate(self.parts):
                pointer = (self.prod_desc.data_block_ptr
                           + (irow * self.prod_desc.columns * self.prod_desc.parts)
                           + (icol * self.prod_desc.parts + iprt))
                self._buffer.jump_to(self._start, _word_to_position(pointer))
                self.data_ptr = self._buffer.read_int(4, self.endian, False)
                if not self.data_ptr:
                    continue
                self._buffer.jump_to(self._start, _word_to_position(self.data_ptr))
                self.data_header_length = self._buffer.read_int(4, self.endian, False)
                data_header = self._buffer.set_mark()
                # if part.header_length == 1:
                #     ihhmm = self._buffer.read_int(4, self.endian, False)
                # if part.header_length == 2:
                #     nreps = self._buffer.read_int(4, self.endian, False)
                #     ihhmm = self._buffer.read_int(4, self.endian, False)
                self._buffer.jump_to(data_header,
                                     _word_to_position(part.header_length + 1))
                lendat = self.data_header_length - part.header_length

                fmt_code = {
                    DataTypes.real: 'f',
                    DataTypes.realpack: 'i',
                    DataTypes.character: 's',
                }.get(part.data_type)

                if fmt_code is None:
                    raise NotImplementedError(f'No methods for data type {part.data_type}')
                if fmt_code == 's':
                    lendat *= BYTES_PER_WORD

                packed_buffer = (
                    self._buffer.read_struct(
                        struct.Struct(f'{self.prefmt}{lendat}{fmt_code}')
                    )
                )

                parameters = self.parameters[iprt]

                if part.data_type == DataTypes.realpack:
                    unpacked = self._unpack_real(packed_buffer, parameters, lendat)
                    for iprm, param in enumerate(parameters['name']):
                        values[param] = unpacked[iprm]
                elif part.data_type == DataTypes.character:
                    for iprm, param in enumerate(parameters['name']):
                        values[param] = self._decode_strip(packed_buffer[iprm])
                else:
                    for iprm, param in enumerate(parameters['name']):
                        values[param] = packed_buffer[iprm]

            processed = self._process_report_text(report, values)

            for rpt in processed:
                reports.append(rpt)
        return reports

    @staticmethod
    def _process_report_text(report, values):
        """Process METAR and SPECI text.

        This method will parse the METAR/SPECI text to ensure that all reports
        have been processed and given the proper timestamp.
        """
        processed = []
        year = report['DATE'].year
        month = report['DATE'].month

        # GEMPAK bins reports in this timestamp, not by actual observed time.
        gem_day = report['DATE'].day
        gem_hour = report['TIME'].hour
        gem_minute = report['TIME'].minute

        # Each surface report can have one or both of METAR/SPECI text.
        text = values.pop('TEXT', None)
        spcl = values.pop('SPCL', None)

        for param, txt in zip(['TEXT', 'SPCL'], [text, spcl]):
            if txt:
                station = METAR_STATION_RE.search(txt)
                if station is None:
                    # If no station can be parsed, at least ensure the text
                    # will be kept with the report we return.
                    stnstr = '----'
                else:
                    stnstr = station.groupdict()['station']

                if txt.count(stnstr) > 1:
                    # GEMPAK sometimes has more than one text report attached. We can
                    # recover all of them by splitting with the station ID.
                    reports = [
                        f'{stnstr}{s}'.strip() for s in filter(None, txt.split(stnstr))
                    ]
                else:
                    reports = [txt.strip()]

                for rpt in reports:
                    timestamp = METAR_TIME_RE.search(rpt)
                    if timestamp is None:
                        # If no timestamp can be parsed, we will use the GEMPAK
                        # stored timestamp.
                        time_group = {
                            'day': gem_day,
                            'hour': gem_hour,
                            'minute': gem_minute
                        }
                    else:
                        time_group = timestamp.groupdict()

                    new_report = deepcopy(report)
                    if param == 'SPCL':  # Do not update standard METAR time
                        dt = datetime(year, month, int(time_group['day']),
                                      int(time_group['hour']), int(time_group['minute']))
                        new_report['DATE'] = dt.date()
                        new_report['TIME'] = dt.time()

                    if ((param == 'TEXT' and values)
                       or (param == 'SPCL' and not text and values)):
                        # The surface values more than likely associated only with
                        # standard METAR GEMPAK has decoded. However, this should
                        # attach values to SPECI reports if they are present and
                        # no METAR text was found.
                        new_report.update(**values)
                    new_report[param] = rpt

                    processed.append(new_report)

        return processed

    def nearest_time(self, date_time, station_id=None, station_number=None, state=None,
                     country=None, bbox=None, include_special=False):
        """Get nearest observation to given time for selected stations.

        Parameters
        ----------
        date_time : datetime or array-like of datetime
            Valid/observed datetime of the surface station.
            Alternatively a string with the format YYYYmmddHHMM.

        station_id : str or array-like of str
            Station ID of the surface station.

        station_number : int or array-like of int or str
            Station number of the surface station.

        state : str or array-like of str
            State where surface station is located.

        country : str or array-like of str
            Country where surface station is located.

        bbox: floats (left, right, bottom, top)
            Bounding area where surface stations are located.

        include_special : bool
            If True, parse special observations that are stored
            as raw METAR text. Default is False.

        Returns
        -------
        list
            List of dicts/JSONs for each surface station.

        Notes
        -----
        Only one filter option can be used at one time. Larger spatial
        extents will yield slower performance.
        """
        if isinstance(date_time, str):
            date_time = datetime.strptime(date_time, '%Y%m%d%H%M')

        nargs = sum(map(bool, [station_id, station_number, state, country, bbox]))

        if nargs == 0:
            raise ValueError('Must have one filter.')

        if nargs > 1:
            raise NotImplementedError('Multiple filters are not supported.')

        if station_id is not None:
            if (not isinstance(station_id, Iterable)
               or isinstance(station_id, str)):
                station_id = [station_id]
            station_id = [c.upper() for c in station_id]

        if station_number is not None:
            if not isinstance(station_number, Iterable):
                station_number = [station_number]
            station_number = [int(sn) for sn in station_number]

        if country is not None:
            if (not isinstance(country, Iterable)
               or isinstance(country, str)):
                country = [country]
            country = [c.upper() for c in country]

        if state is not None:
            if (not isinstance(state, Iterable)
               or isinstance(state, str)):
                state = [state]
            state = [s.upper() for s in state]

        if bbox is not None:
            station_id = {
                stn.ID for stn in self._sfinfo if _bbox_filter(stn.LAT, stn.LON, bbox)
            }

        if country:
            station_id = set()
            for cty in country:
                station_id.update(stn.ID for stn in self._sfinfo if stn.COUNTRY == cty)

        if state:
            station_id = set()
            for st in state:
                station_id.update(stn.ID for stn in self._sfinfo if stn.STATE == st)

        time_matched = []
        if station_id:
            for stn in station_id:
                matched = self.sfjson(station_id=stn, include_special=include_special)

                nearest = min(
                    matched,
                    key=lambda d: abs(d['properties']['date_time'] - date_time)
                )

                time_matched.append(nearest)

        if station_number:
            for stn in station_number:
                matched = self.sfjson(station_number=stn, include_special=include_special)

                nearest = min(
                    matched,
                    key=lambda d: abs(d['properties']['date_time'] - date_time)
                )

                time_matched.append(nearest)

        return time_matched

    def sfjson(self, station_id=None, station_number=None, date_time=None, state=None,
               country=None, bbox=None, include_special=False):
        """Select surface stations and output as list of JSON objects.

        Subset the data by parameter values. The default is to not
        subset and return the entire dataset.

        Parameters
        ----------
        station_id : str or array-like of str
            Station ID of the surface station.

        station_number : int or array-like of int
            Station number of the surface station.

        date_time : datetime or array-like of datetime
            Datetime of the surface observation. Alternatively
            a string with the format YYYYmmddHHMM or first|FIRST
            or last|LAST which function to retrieve the latest
            and oldest time within the file, respectively.

        state : str or array-like of str
            State where surface station is located.

        country : str or array-like of str
            Country where surface station is located.

        bbox: floats (left, right, bottom, top)
            Bounding area where surface stations are located.

        include_special : bool
            If True, parse special observations that are stored
            as raw METAR text. Default is False.

        Returns
        -------
        list
            List of dicts/JSONs for each surface station.
        """
        if station_id is not None:
            if (not isinstance(station_id, Iterable)
               or isinstance(station_id, str)):
                station_id = [station_id]
            station_id = [c.upper() for c in station_id]

        if station_number is not None:
            if not isinstance(station_number, Iterable):
                station_number = [station_number]
            station_number = [int(sn) for sn in station_number]

        if date_time is not None:
            if (not isinstance(date_time, Iterable)
               or (isinstance(date_time, str)
               and date_time not in ['first', 'FIRST', 'last', 'LAST'])):
                date_time = [date_time]
            if date_time not in ['first', 'FIRST', 'last', 'LAST']:
                for i, dt in enumerate(date_time):
                    if isinstance(dt, str):
                        date_time[i] = datetime.strptime(dt, '%Y%m%d%H%M')

        if state is not None:
            if (not isinstance(state, Iterable)
               or isinstance(state, str)):
                state = [state]
            state = [s.upper() for s in state]

        if country is not None:
            if (not isinstance(country, Iterable)
               or isinstance(country, str)):
                country = [country]
            country = [c.upper() for c in country]

        # Figure out which columns to extract from the file
        # matched = self._sfinfo.copy()
        matched = sorted(self._sfinfo)

        # Do this now or the matched filter iterator will be consumed
        # prematurely.
        if date_time in ['last', 'LAST']:
            date_time = [max((d.DATTIM for d in matched))]
        elif date_time in ['first', 'FIRST']:
            date_time = [min((d.DATTIM for d in matched))]

        if country is not None:
            matched = filter(
                lambda sfc: sfc.COUNTRY in country,
                matched
            )

        if state is not None:
            matched = filter(
                lambda sfc: sfc.STATE in state,
                matched
            )

        if bbox is not None:
            matched = filter(
                lambda sfc: _bbox_filter(sfc.LAT, sfc.LON, bbox),
                matched
            )

        if date_time is not None:
            matched = filter(
                lambda sfc: sfc.DATTIM in date_time,
                matched
            )

        if station_id is not None:
            matched = filter(
                lambda sfc: sfc.ID in station_id,
                matched
            )

        if station_number is not None:
            matched = filter(
                lambda sfc: sfc.NUMBER in station_number,
                matched
            )

        matched = list(matched)

        if len(matched) < 1:
            raise KeyError('No stations were matched with given parameters.')

        sfcno = [(s.ROW, s.COL) for s in matched]

        if self.surface_type == 'standard':
            data = self._unpack_standard(sfcno)
        elif self.surface_type == 'ship':
            data = self._unpack_ship(sfcno)
        elif self.surface_type == 'climate':
            data = self._unpack_climate(sfcno)

        stnarr = []
        for stn in data:
            if stn:
                if not include_special and 'SPCL' in stn:
                    continue
                stnobj = {
                    'properties': {
                        'date_time': datetime.combine(stn.pop('DATE'),
                                                      stn.pop('TIME')),
                        'station_id': stn.pop('STID') + stn.pop('STD2'),
                        'station_number': stn.pop('STNM'),
                        'longitude': stn.pop('SLON'),
                        'latitude': stn.pop('SLAT'),
                        'elevation': stn.pop('SELV'),
                        'state': stn.pop('STAT'),
                        'country': stn.pop('COUN'),
                        'priority': stn.pop('SPRI'),
                    },
                    'values': {name.lower(): ob for name, ob in stn.items()}
                }
                stnarr.append(stnobj)

        return stnarr
