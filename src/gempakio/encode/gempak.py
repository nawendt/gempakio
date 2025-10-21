# Copyright (c) 2025 Nathan Wendt.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Classes for encoding various GEMPAK file formats."""

from collections import namedtuple
import ctypes
from datetime import datetime, timedelta
from io import BytesIO
import re
import struct
from math import ceil

import numpy as np
import pyproj

from gempakio.common import (
    _position_to_word,
    _word_to_position,
    ANLB_SIZE,
    DataSource,
    DataTypes,
    FileTypes,
    GEMPAK_HEADER,
    HEADER_DTYPE,
    MAX_LEVELS,
    MBLKSZ,
    MISSING_FLOAT,
    MISSING_INT,
    MMFREE,
    MMHDRS,
    MMPARM,
    NAVB_SIZE,
    PackingType,
    VerticalCoordinates,
)
from gempakio.tools import NamedStruct, OrderedSet


def pack_grib(grid, missing_float, nbits=16):
    """Pack a grid of floats into integers."""
    kxky = np.multiply(*grid.shape)

    if nbits < 1 or nbits > 31:
        raise ValueError('Precision requested is invalid.')

    lendat = (nbits * kxky) // 32
    if lendat * 32 != nbits * kxky:
        lendat += 1

    out = np.zeros(lendat, dtype=np.int32)

    has_missing = bool((grid == missing_float).any())

    all_missing = bool((grid == missing_float).all())

    if all_missing:
        qmin = missing_float
        qmax = missing_float
    else:
        qmin = grid.min()
        qmax = grid.max()

    qdiff = qmax - qmin
    idat = round(qdiff)
    if qdiff < 0 or idat > 2147483647:  # Max 32-bit integer
        raise ValueError('Problem packing grid.')

    if qdiff == 0 and not has_missing:
        # This is a constant grid
        scale = 1
    else:
        imax = 2**nbits - 1
        nnnn = 0
        if abs(qdiff) > (2**-126 * imax):  # Smallest number for 32-bit float
            if idat >= imax:
                while idat >= imax:
                    nnnn -= 1
                    idat = qdiff * 2**nnnn
            else:
                while round(qdiff * 2 ** (nnnn + 1)) < imax:
                    nnnn += 1

        scale = 2**nnnn

        rgrid = grid.ravel()
        rgrid_round = np.where(rgrid != missing_float, 
                                np.round(np.maximum(rgrid - qmin, 0) * scale).astype('int32'), 
                                imax)

        # Compute the amount of shifting for each input word and which input words contain the start of an output word
        word_shifts = ((33 - ((np.arange(kxky) + 1) * nbits)) % 32 - 1) % 32
        word_starts = (word_shifts >= (32 - nbits))
        word_start_idxs = np.where(word_starts)[0]

        # The maximum number of input words covered by each output word (maybe tack on one for the other part of any
        #   split words)
        any_split_words = (32 / nbits) != (32 // nbits)
        n_input_words = np.diff(word_start_idxs).max() + (1 if any_split_words else 0)

        # Now construct 2d arrays of shifts and input indexes, where each array is the output length by number of input
        #   words per output word
        jshfts = np.zeros((lendat, n_input_words), dtype=np.int8)
        iis = np.zeros((lendat, n_input_words), dtype=np.int64)

        # Fill the first input word with the relevant values from the word starts
        jshfts[:, 0] = word_shifts[word_starts]
        iis[:, 0] = word_start_idxs

        # For each output word, the max index is the start index for the next output word. (Except for the last output
        #   word, which has the input data length as the max index)
        iis_2d_max = np.roll(word_start_idxs, -1)
        iis_2d_max[-1] = kxky - 1

        for niw in range(1, n_input_words):
            # Fill the next input word (indexes get incremented by one and shifts subtract nbits)
            iis[:, niw] = iis[:, niw - 1] + 1
            jshfts[:, niw] = jshfts[:, niw - 1] - nbits

            # Check for unneeded words (indexes exceed the max for this word)
            unneeded_words = (iis[:, niw] > iis_2d_max) | (iis[:, niw - 1] == -1)

            iis[unneeded_words, niw] = -1
            jshfts[unneeded_words, niw] = 0

        # Shift and sum all the input words to get the correct output word
        rgrid_shifted = np.where(iis >= 0,
                                 np.where(jshfts > 0, rgrid_round[iis] << jshfts, rgrid_round[iis] >> np.abs(jshfts)),
                                 0)
        
        out = rgrid_shifted.sum(axis=1)

        scale **= -1

    return qmin, scale, out


class GempakStream(BytesIO):
    """In-memory bytes stream for GEMPAK data."""

    def __init__(self):
        super().__init__()

    def jump_to(self, word):
        """Jumpt to given word."""
        self.seek(_word_to_position(word))

    def word(self):
        """Get current word."""
        return _position_to_word(self.tell())

    def write_string(self, string):
        """Write string word."""
        self.write(struct.pack('<4s', bytes(f'{string:<4s}', 'utf-8')))

    def write_int(self, i):
        """Write integer word."""
        self.write(struct.pack('<i', i))

    def write_float(self, f):
        """Write float word."""
        self.write(struct.pack('<f', f))

    def write_struct(self, struct_class, **kwargs):
        """Write structure to file as bytes."""
        self.write(struct_class.pack(**kwargs))


class DataManagementFile:
    """Class to facilitate writing GEMPAK files to disk."""

    _label_struct = NamedStruct(
        [
            ('dm_head', '28s'),
            ('version', 'i'),
            ('file_headers', 'i'),
            ('file_keys_ptr', 'i'),
            ('rows', 'i'),
            ('row_keys', 'i'),
            ('row_keys_ptr', 'i'),
            ('row_headers_ptr', 'i'),
            ('columns', 'i'),
            ('column_keys', 'i'),
            ('column_keys_ptr', 'i'),
            ('column_headers_ptr', 'i'),
            ('parts', 'i'),
            ('parts_ptr', 'i'),
            ('data_mgmt_ptr', 'i'),
            ('data_mgmt_length', 'i'),
            ('data_block_ptr', 'i'),
            ('file_type', 'i'),
            ('data_source', 'i'),
            ('machine_type', 'i'),
            ('missing_int', 'i'),
            (None, '12x'),
            ('missing_float', 'f'),
        ],
        '<',
        'Label',
    )

    _data_mgmt_struct = NamedStruct(
        [
            ('next_free_word', 'i'),
            ('max_free_pairs', 'i'),
            ('actual_free_pairs', 'i'),
            ('last_word', 'i'),
            (None, '464x'),
        ],
        '<',
        'DataManagement',
    )

    _grid_nav_struct = NamedStruct(
        [
            ('grid_definition_type', 'f'),
            ('projection', '4s'),
            ('left_grid_number', 'f'),
            ('bottom_grid_number', 'f'),
            ('right_grid_number', 'f'),
            ('top_grid_number', 'f'),
            ('lower_left_lat', 'f'),
            ('lower_left_lon', 'f'),
            ('upper_right_lat', 'f'),
            ('upper_right_lon', 'f'),
            ('proj_angle1', 'f'),
            ('proj_angle2', 'f'),
            ('proj_angle3', 'f'),
            (None, '972x'),
        ],
        '<',
        'Navigation',
    )

    _analysis_struct = NamedStruct(
        [
            ('analysis_type', 'f'),
            ('delta_n', 'f'),
            ('grid_ext_left', 'f'),
            ('grid_ext_down', 'f'),
            ('grid_ext_right', 'f'),
            ('grid_ext_up', 'f'),
            ('garea_llcr_lat', 'f'),
            ('garea_llcr_lon', 'f'),
            ('garea_urcr_lat', 'f'),
            ('garea_urcr_lon', 'f'),
            ('extarea_llcr_lat', 'f'),
            ('extarea_llcr_lon', 'f'),
            ('extarea_urcr_lat', 'f'),
            ('extarea_urcr_lon', 'f'),
            ('datarea_llcr_lat', 'f'),
            ('datarea_llcr_lon', 'f'),
            ('datarea_urcr_lat', 'f'),
            ('datarea_urcrn_lon', 'f'),
            (None, '440x'),
        ],
        '<',
        'Analysis',
    )

    def __init__(self):
        self._version = 1
        self._machine_type = 11
        self._missing_int = MISSING_INT
        self._missing_float = MISSING_FLOAT
        self._data_mgmt_ptr = 129
        self._data_mgmt_length = MBLKSZ
        self._max_free_pairs = MMFREE
        self._actual_free_pairs = 0
        self._last_word = 0
        self.parameter_names = []
        self.row_names = []
        self.column_names = []
        self.rows = 0
        self.columns = 0
        self._file_headers = {}
        self._parts_dict = {}
        self.file_type = None
        self.data_source = None
        self._column_set = OrderedSet()
        self._row_set = OrderedSet()
        self.column_headers = []
        self.row_headers = []
        self.packing_type = None
        self.data = {}

    @staticmethod
    def _dmword(word):
        """Find the record and start word within record.

        Notes
        -----
        See GEMPAK function DM_WORD.
        """
        record = (word - 1) // MBLKSZ + 1
        start = word - (record - 1) * MBLKSZ

        return record, start

    @staticmethod
    def _encode_vertical_coordinate(coord):
        try:
            return VerticalCoordinates[coord.lower()].value
        except KeyError as err:
            raise KeyError(f'`{coord}` has no numeric value.') from err

    def _init_headers(self):
        self.make_column_header = namedtuple('ColumnHeader', self.column_names)
        self.make_row_header = namedtuple('RowHeader', self.row_names)

    def _replace_nan(self, array):
        """Replace nan values from an array with missing value."""
        nan_loc = np.isnan(array)
        array[nan_loc] = self._missing_float
        return array

    def _set_pointers(self):
        """Set pointers for the output file.

        Notes
        -----
        See GEMPAK function DM_CRET.
        """
        # Keys
        self.row_keys = len(self.row_names)
        self._row_keys_ptr = self._data_mgmt_ptr + self._data_mgmt_length

        self.column_keys = len(self.column_names)
        self.column_keys_ptr = self._row_keys_ptr + self.row_keys

        # Headers
        self.file_keys_ptr = self.column_keys_ptr + self.column_keys
        lenfil = 0
        for _fh, info in self._file_headers.items():
            lenfil += info['length'] + 1
        rec, word = self._dmword(self.file_keys_ptr + 3 * len(self._file_headers) + lenfil)
        if word != 1:
            self.row_headers_ptr = rec * MBLKSZ + 1
        else:
            self.row_headers_ptr = self.file_keys_ptr + 3 * len(self._file_headers) + lenfil
        self.column_headers_ptr = self.row_headers_ptr + self.rows * (self.row_keys + 1)

        # Parts
        lenpart = 0
        nparts = len(self._parts_dict)
        for _part, info in self._parts_dict.items():
            lenpart += len(info['parameters'])
        rec, word = self._dmword(
            self.column_headers_ptr + self.columns * (self.column_keys + 1)
        )
        if word != 1:
            self.parts_ptr = rec * MBLKSZ + 1
        else:
            self.parts_ptr = self.column_headers_ptr + self.columns * (self.column_keys + 1)

        # Data
        rec, word = self._dmword(self.parts_ptr + 4 * nparts + 4 * lenpart)
        if word != 1:
            self.data_block_ptr = rec * MBLKSZ + 1
        else:
            self.data_block_ptr = self.parts_ptr + 4 * nparts + 4 * lenpart

        # Data Management (initial next free word)
        rec, word = self._dmword(self.data_block_ptr + nparts * self.rows * self.columns)
        if word != 1:
            self.next_free_word = rec * MBLKSZ + 1
        else:
            self.next_free_word = self.data_block_ptr + nparts * self.rows * self.columns

    def _write_label(self, stream):
        """Write file label to a stream."""
        stream.write_struct(
            self._label_struct,
            dm_head=bytes(GEMPAK_HEADER, 'utf-8'),
            version=self._version,
            file_headers=len(self._file_headers),
            file_keys_ptr=self.file_keys_ptr,
            rows=self.rows,
            row_keys=self.row_keys,
            row_keys_ptr=self._row_keys_ptr,
            row_headers_ptr=self.row_headers_ptr,
            columns=self.columns,
            column_keys=self.column_keys,
            column_keys_ptr=self.column_keys_ptr,
            column_headers_ptr=self.column_headers_ptr,
            parts=len(self._parts_dict),
            parts_ptr=self.parts_ptr,
            data_mgmt_ptr=self._data_mgmt_ptr,
            data_mgmt_length=self._data_mgmt_length,
            data_block_ptr=self.data_block_ptr,
            file_type=self.file_type,
            data_source=self.data_source,
            machine_type=self._machine_type,
            missing_int=self._missing_int,
            missing_float=self._missing_float,
        )

    def _write_data_management(self, stream):
        """Write data management block to a stream."""
        stream.write_struct(
            self._data_mgmt_struct,
            next_free_word=self.next_free_word,
            max_free_pairs=self._max_free_pairs,
            actual_free_pairs=self._actual_free_pairs,
            last_word=self._last_word,
        )

    def _write_file_keys(self, stream):
        """Write file headers to a stream."""
        for name in self._file_headers:
            stream.write_string(name)

        for _name, info in self._file_headers.items():
            stream.write_int(info['length'])

        for _name, info in self._file_headers.items():
            stream.write_int(info['type'])

    def _write_row_keys(self, stream):
        """Write row keys to a stream."""
        for rn in self.row_names:
            stream.write_string(rn)

    def _write_column_keys(self, stream):
        """Write column keys to a stream."""
        for cn in self.column_names:
            stream.write_string(cn)

    def _write_parts(self, stream):
        """Write parts to a stream."""
        for name in self._parts_dict:
            stream.write_string(name)

        for _name, info in self._parts_dict.items():
            stream.write_int(info['header'])

        for _name, info in self._parts_dict.items():
            stream.write_int(info['type'])

        for _name, info in self._parts_dict.items():
            stream.write_int(len(info['parameters']))

        for _name, info in self._parts_dict.items():
            for param in info['parameters']:
                stream.write_string(param)

        for _name, info in self._parts_dict.items():
            for offset in info['offset']:
                stream.write_int(offset)

        for _name, info in self._parts_dict.items():
            for scale in info['scale']:
                stream.write_int(scale)

        for _name, info in self._parts_dict.items():
            for bits in info['bits']:
                stream.write_int(bits)

    def _write_file_headers(self):
        raise NotImplementedError('Must be defined within subclass.')

    def _write_row_headers(self):
        raise NotImplementedError('Must be defined within subclass.')

    def _write_column_headers(self):
        raise NotImplementedError('Must be defined within subclass.')

    def _write_data(self):
        raise NotImplementedError('Must be defined within subclass.')

    def to_gempak(self, file):
        """Write GEMPAK file to disk.

        Parameters
        ----------
        file : str or `pathlib.Path`
            Path of file to be created.
        """
        self._set_pointers()

        with GempakStream() as stream:
            # Write file label
            self._write_label(stream)

            # Write row key names
            stream.jump_to(self._row_keys_ptr)
            self._write_row_keys(stream)

            # Write column key names
            stream.jump_to(self.column_keys_ptr)
            self._write_column_keys(stream)

            # Write file keys/headers, if present
            if self._file_headers:
                stream.jump_to(self.file_keys_ptr)
                self._write_file_keys(stream)
                self._write_file_headers(stream)

            # Write parts and parameters
            stream.jump_to(self.parts_ptr)
            self._write_parts(stream)

            # Write row headers
            stream.jump_to(self.row_headers_ptr)
            self._write_row_headers(stream)

            # Write column headers
            stream.jump_to(self.column_headers_ptr)
            self._write_column_headers(stream)

            # Write data
            stream.jump_to(self.data_block_ptr)
            self._write_data(stream)

            # Write data management record
            stream.jump_to(self._data_mgmt_ptr)
            self._write_data_management(stream)

            with open(file, 'wb') as out:
                out.write(stream.getbuffer())


class GridFile(DataManagementFile):
    """GEMPAK grid file class.

    This class is used to build a collection of grids to write to disk
    as a GEMPAK grid file.
    """

    def __init__(self, lon, lat, projection, use_xy=False, rotation=0):
        """Instantiate GridFile.

        Parameters
        ----------
        lon : `numpy.ndarray`
            Longitude of the grid. Dimension order (y, x). If use_xy is
            True, this will be interpreted as x projected coordinate
            and will be one-dimensional (x,).

        lat : `numpy.ndarray`
            Latitude of the grid. Dimension order (y, x). If use_xy is
            True, this will be interpreted as y projected coordinate
            and will be one-dimensional (y,).

        projection : `pyproj.Proj`
            Data projection. Valid options are:
                Mercator
                Stereographic
                Lambert Conformal Conic
                Equidistant Cylindrical
                Orthographic
                Azimuthal Equidistant
                Lambert Azimuthal Equal Area
                Gnomonic
                Transverse Mercator
                Universal Transverse Mercator

        use_xy : bool
            Input coordinates are projected (x and y). Default is False.

        rotation : float
            Angle (in degrees) by which to rotate the projection. Default
            is 0.

        Notes
        -----
        Grids are written with an empty analysis block.
        """
        super().__init__()
        self.file_type = FileTypes.grid
        self.data_source = DataSource.grid
        self.packing_type = PackingType.grib
        self._data_type = DataTypes.grid
        self.rows = 1
        self.angle1 = 0
        self.angle2 = 0
        self.angle3 = 0
        self.rotation = rotation
        self.left_grid_number = 1
        self.bottom_grid_number = 1
        self.precision = 16

        self._file_headers = {
            'NAVB': {'length': NAVB_SIZE, 'type': 1},
            'ANLB': {'length': ANLB_SIZE, 'type': 1},
        }
        self.row_names = ['GRID']
        self.column_names = [
            'GDT1',
            'GTM1',
            'GDT2',
            'GTM2',
            'GLV1',
            'GLV2',
            'GVCD',
            'GPM1',
            'GPM2',
            'GPM3',
        ]
        self.parameter_names = ['GRID']

        self._init_headers()

        self._parts_dict = {
            'GRID': {
                'header': 2,  # minimum required for nx, ny
                'type': self._data_type,
                'parameters': self.parameter_names,
                'scale': [0] * len(self.parameter_names),
                'offset': [0] * len(self.parameter_names),
                'bits': [0] * len(self.parameter_names),
            }
        }

        if not use_xy and lat.shape != lon.shape:
            raise ValueError('Input coordinates must be same dimensions.')

        if not isinstance(projection, pyproj.Proj):
            raise TypeError('projection must be pyproj.Proj class.')

        self.projection = projection

        if use_xy:
            if len(lat.shape) != 1 or len(lon.shape) != 1:
                raise ValueError('Projected input coordinates must one-dimensional.')
            self.nx = len(lon)
            self.ny = len(lat)
            self.x = lon
            self.y = lat
            self.lon, self.lat = self.projection(
                *np.meshgrid(lon, lat, copy=False), inverse=True
            )
            self._is_xy = True
        else:
            if len(lat.shape) != 2 or len(lon.shape) != 2:
                raise ValueError('Geographic input coordinates must be two-dimensional.')
            self.ny, self.nx = lat.shape
            _x, _y = self.projection(lon, lat)
            self.x = _x[0, :]
            self.y = _y[:, 0]
            self.lon = lon
            self.lat = lat
            self._is_xy = False

        self._set_projection_params()
        self._set_bbox()

    @staticmethod
    def _to_dict(operation):
        param_dict = {}
        for param in operation.params:
            param_dict[param.name.lower().replace(' ', '_')] = param.value
        return param_dict

    def _set_bbox(self):
        """Set bounds of data."""
        if self._is_xy:
            self.lower_left_lon, self.lower_left_lat = self.projection(
                self.x[0], self.y[0], inverse=True
            )
            self.upper_right_lon, self.upper_right_lat = self.projection(
                self.x[-1], self.y[-1], inverse=True
            )
        else:
            self.lower_left_lon = self.lon[0, 0]
            self.lower_left_lat = self.lat[0, 0]
            self.upper_right_lon = self.lon[-1, -1]
            self.upper_right_lat = self.lat[-1, -1]

    def _set_projection_params(self):
        """Set projection parameters for GridFile."""
        params = self.projection.crs.to_cf()
        name = params.get('grid_mapping_name')

        if name is None:
            method = self.projection.crs.coordinate_operation.method_name
            if 'Equidistant Cylindrical' in method:
                name = 'equidistant_cylindrical'
            elif method == 'Gnomonic':
                name = 'gnomonic'
                params = params['crs_wkt']  # Not converted to CF by pyproj/PROJ
            elif 'Lambert Azimuthal Equal Area' in method:
                name = 'lambert_azimuthal_equal_area'
                params = params['crs_wkt']  # Not converted to CF by pyproj/PROJ
            else:
                name = 'unknown'

        if name == 'mercator':
            self.gemproj = 'MER'
            self.angle1 = params['standard_parallel']
            self.angle2 = params['longitude_of_projection_origin']
            self.angle3 = self.rotation
        elif name == 'polar_stereographic':
            self.angle1 = params['latitude_of_projection_origin']
            self.angle2 = params['straight_vertical_longitude_from_pole']
            self.gemproj = 'STR'
        elif name == 'stereographic':
            self.angle1 = params['latitude_of_projection_origin']
            self.angle2 = params['longitude_of_projection_origin']
            self.gemproj = 'STR'
            self.angle3 = self.rotation
        elif name == 'lambert_conformal_conic':
            self.angle1, self.angle3 = params['standard_parallel']
            self.angle2 = params['longitude_of_central_meridian']
            self.gemproj = 'LCC'
        elif name == 'equidistant_cylindrical':
            self.gemproj = 'CED'
            params = self._to_dict(self.projection.crs.coordinate_operation)
            self.angle1 = params['latitude_of_natural_origin']
            self.angle2 = params['longitude_of_natural_origin']
            self.angle3 = self.rotation
        elif name == 'orthographic':
            self.angle1 = params['latitude_of_projection_origin']
            self.angle2 = params['longitude_of_projection_origin']
            self.gemproj = 'ORT'
            self.angle3 = self.rotation
        elif name == 'azimuthal_equidistant':
            self.gemproj = 'AED'
            self.angle1 = params['latitude_of_projection_origin']
            self.angle2 = params['longitude_of_projection_origin']
            self.angle3 = self.rotation
        elif name == 'lambert_azimuthal_equal_area':
            self.gemproj = 'LEA'
            self.angle1 = float(
                re.search(
                    r'(?:\"Latitude of natural origin\",(?P<lat_0>-?\d{1,2}\.?\d*))', params
                ).groupdict()['lat_0']
            )
            self.angle2 = float(
                re.search(
                    r'(?:\"Longitude of natural origin\",(?P<lon_0>-?\d{1,3}\.?\d*))', params
                ).groupdict()['lon_0']
            )
            self.angle3 = self.rotation
        elif name == 'gnomonic':
            self.gemproj = 'GNO'
            self.angle1 = float(
                re.search(
                    r'(?:\"Latitude of natural origin\",(?P<lat_0>-?\d{1,2}\.?\d*))', params
                ).groupdict()['lat_0']
            )
            self.angle2 = float(
                re.search(
                    r'(?:\"Longitude of natural origin\",(?P<lon_0>-?\d{1,3}\.?\d*))', params
                ).groupdict()['lon_0']
            )
            self.angle3 = self.rotation
        else:
            raise NotImplementedError(f'`{name}` projection not implemented.')

    def add_grid(
        self,
        grid,
        parameter_name,
        vertical_coordinate,
        level,
        date_time,
        level2=None,
        date_time2=None,
    ):
        """Add grid to the file.

        Parameters
        ----------
        grid : `numpy.ndarray`
            Grid data to be added. Dimension order (y, x).

        parameter_name : str
            Name of parameter. Note, names in GEMPAK have a maximum
            of 12 characters.

        vertical_coordinate : str or None
            Vertical coordinate of grid. Names must 4 characters.

        level : int
            Vertical level of the grid.

        date_time : str or datetime
            Grid date and time. Valid string formats are YYYYmmddHHMM
            or YYYYmmddHHMMThhhmm, where T is the grid type and hhhmm is the forecast
            hour/minute from the preceding initialization date and time. mm can be omitted
            not used. If hhh is also missing, 0 will be assumed.

            Valid types (T):
                A : Analysis
                F : Forecast
                V : Valid
                G : Guess
                I : Initial

            An analysis (A) is assumed if no type is designated.

        level2 : int or None
            Secondary vertical level of the grid. This is typically used for
            layer data.

        date_time2 : str or datetime or None
            Secondary grid date and time. This is typically used for time-averaged
            data or other time window applications. String format follows that of
            date_time argument.
        """
        if not isinstance(grid, np.ndarray):
            raise TypeError('Grid must be a numpy.ndarray.')

        if grid.shape != (self.ny, self.nx):
            raise ValueError(
                f'Grid dimensions {grid.shape} must match dimensions '
                f'{(self.ny, self.nx)} defined in GridFile.'
            )

        if not isinstance(parameter_name, str):
            raise TypeError('Parameter name bust be a string.')

        if len(parameter_name) > 12:
            raise ValueError('Parameter names cannot be more than 12 characters.')

        parameter_name = parameter_name.upper()

        if not isinstance(vertical_coordinate, str | type(None)):
            raise TypeError('Vertical coordinate must be string or None.')

        if vertical_coordinate is None:
            vertical_coordinate = 'NONE'
        else:
            vertical_coordinate = vertical_coordinate.upper()

        if len(vertical_coordinate) > 4:
            raise ValueError('Vertical coordinate can only be 4 characters.')

        if not isinstance(level, int | float):
            raise TypeError('Level parameter must be an integer.')

        level = int(level)

        forecast_hour = 0
        forecast_minute = 0
        grid_type = 0
        if isinstance(date_time, str):
            if len(date_time) < 12:
                raise ValueError(f'{date_time} does not match minimum format of YYYYmmddHHMM.')
            date_time = date_time.upper()
            split_time = re.split('([AFVIG])', date_time)
            if len(split_time) == 3:
                init, gtype, fhr = split_time
                init_date = datetime.strptime(init, '%Y%m%d%H%M')
                if len(fhr) > 3:
                    fmin = fhr[3:]
                    fhr = fhr[:3]
                    forecast_hour = int(fhr)
                    forecast_minute = int(fmin)
                else:
                    forecast_hour = 0 if fhr == '' else int(fhr)
                grid_type = {'A': 0, 'F': 1, 'V': 1, 'G': 2, 'I': 3}.get(gtype)
            elif len(split_time) > 3:
                raise ValueError(f'Cannot parse malformed date_time input {date_time}.')
            else:
                init_date = datetime.strptime(date_time, '%Y%m%d%H%M')
                forecast_hour = int(init_date.strftime('%H'))
                forecast_minute = int(init_date.strftime('%M'))
        elif isinstance(date_time, datetime):
            init_date = date_time
            forecast_hour = int(init_date.strftime('%H'))
            forecast_minute = int(init_date.strftime('%M'))
        else:
            raise TypeError('date_time must be string or datetime.')

        if not isinstance(level2, int | float | type(None)):
            raise TypeError('Secondary level must be integer or None.')

        if level2 is not None:
            level2 = int(level2)

        forecast_hour2 = 0
        forecast_minute2 = 0
        grid_type2 = 0
        if date_time2 is not None:
            if isinstance(date_time2, str):
                if len(date_time) < 12:
                    raise ValueError(
                        f'{date_time2} does not match minimum format of YYYYmmddHHMM.'
                    )
                date_time2 = date_time2.upper()
                split_time = re.split('([AFVIG])', date_time2)
                if len(split_time) == 3:
                    init, gtype, fhr = split_time
                    init_date2 = datetime.strptime(init, '%Y%m%d%H%M')
                    if len(fhr) > 3:
                        fmin = fhr[3:]
                        fhr = fhr[:3]
                        forecast_hour2 = int(fhr)
                        forecast_minute2 = int(fmin)
                    else:
                        forecast_hour2 = 0 if fhr == '' else int(fhr)
                    grid_type2 = {'A': 0, 'F': 1, 'V': 1, 'G': 2, 'I': 3}.get(gtype)
                    if grid_type != grid_type2:
                        raise ValueError('Grid type mismatch in date_time and date_time2.')
                elif len(split_time) > 3:
                    raise ValueError(f'Cannot parse malformed date_time2 input {date_time2}.')
                else:
                    init_date2 = datetime.strptime(date_time2, '%Y%m%d%H%M')
                    forecast_hour2 = int(init_date2.strftime('%H'))
                    forecast_minute2 = int(init_date2.strftime('%M'))
            elif isinstance(date_time2, datetime):
                init_date2 = date_time2
                forecast_hour2 = int(init_date2.strftime('%H'))
                forecast_minute2 = int(init_date2.strftime('%M'))
            else:
                raise TypeError('date_time must be string or datetime or None.')

        pbuff = f'{parameter_name:<12s}'
        gpm1, gpm2, gpm3 = (pbuff[i : (i + 4)] for i in range(0, len(pbuff), 4))

        new_column = (
            (grid_type, init_date),
            (grid_type, forecast_hour, forecast_minute),
            (grid_type2, init_date2) if date_time2 is not None else 0,
            (grid_type2, forecast_hour2, forecast_minute2),
            level,
            level2 if level2 is not None else -1,
            (
                self._encode_vertical_coordinate(vertical_coordinate)
                if vertical_coordinate is not None
                else 0
            ),
            gpm1,
            gpm2,
            gpm3,
        )
        self._column_set.add(self.make_column_header(*new_column))

        self.data[new_column] = self._replace_nan(grid)

        self.columns = len(self._column_set)

        if self.rows + self.columns > MMHDRS:
            raise ValueError('Exceeded maximum number data entries.')

        self.column_headers = sorted(self._column_set)
        self.row_headers = sorted(self._row_set)

    def _pack_grib(self, grid, nbits=16):
        """Pack a grid of floats into integers."""
        return pack_grib(grid, self._missing_float, nbits=nbits)

    def _write_row_headers(self, stream):
        """Write row headers to a GridFile stream."""
        # Grid row headers can be simplified as there is only
        # one: GRID
        stream.write_int(-self._missing_int)
        stream.write_int(1)

    def _write_column_headers(self, stream):
        """Write column headers to a GridFile stream."""
        start_word = stream.word()
        # Initialize all headers to unused state
        for _c in range(self.columns):
            for _k in range(self.column_keys + 1):
                stream.write_int(self._missing_int)

        stream.jump_to(start_word)
        for ch in self.column_headers:
            stream.write_int(-self._missing_int)
            for key in ch._fields:
                dtype = HEADER_DTYPE[key]
                if 's' in dtype:
                    stream.write_string(getattr(ch, key))
                elif 'i' in dtype:
                    if key in ['GDT1', 'GDT2']:
                        value = getattr(ch, key)
                        if value != 0:
                            itype, dattim = value
                            if itype == 0:
                                idate = int(dattim.strftime('%y%m%d'))
                            else:
                                idate = int(dattim.strftime('%m%d%y%H%M'))
                        else:
                            idate = 0
                        stream.write_int(idate)
                    elif key in ['GTM1', 'GTM2']:
                        itype, ihr, imin = getattr(ch, key)
                        if itype == 0:
                            ftime = int(f'{ihr:02d}{imin:02d}')
                        else:
                            ftime = itype * 100000 + int(f'{ihr:03d}{imin:02d}')
                        stream.write_int(ftime)
                    else:
                        stream.write_int(getattr(ch, key))

    def _write_file_headers(self, stream):
        """Write file headers to a GridFile stream."""
        # Write navigation block
        stream.write_int(NAVB_SIZE)
        stream.write_struct(
            self._grid_nav_struct,
            grid_definition_type=2,  # always full map projection
            projection=bytes(f'{self.gemproj:4s}', 'utf-8'),  # Must be 4 char w/ space
            left_grid_number=self.left_grid_number,
            bottom_grid_number=self.bottom_grid_number,
            right_grid_number=self.nx,
            top_grid_number=self.ny,
            lower_left_lat=self.lower_left_lat,
            lower_left_lon=self.lower_left_lon,
            upper_right_lat=self.upper_right_lat,
            upper_right_lon=self.upper_right_lon,
            proj_angle1=self.angle1,
            proj_angle2=self.angle2,
            proj_angle3=self.angle3,
        )

        # Write a basic analysis block (type 2)
        rx1 = self.nx // 2
        rx2 = rx1 + 1
        ry1 = self.ny // 2
        ry2 = ry1 + 1
        dellon = self.lon[ry2, rx2] - self.lon[ry1, rx1]
        dellat = self.lat[ry2, rx2] - self.lat[ry1, rx1]
        deltan = np.sqrt(0.5 * (dellon**2 + dellat**2)) * 2

        stream.write_int(ANLB_SIZE)
        stream.write_struct(
            self._analysis_struct,
            analysis_type=2,
            delta_n=deltan,
            grid_ext_left=0,
            grid_ext_down=0,
            grid_ext_right=0,
            grid_ext_up=0,
            garea_llcr_lat=self.lower_left_lat,
            garea_llcr_lon=self.lower_left_lon,
            garea_urcr_lat=self.upper_right_lat,
            garea_urcr_lon=self.upper_right_lon,
            extarea_llcr_lat=self.lower_left_lat,
            extarea_llcr_lon=self.lower_left_lon,
            extarea_urcr_lat=self.upper_right_lat,
            extarea_urcr_lon=self.upper_right_lon,
            datarea_llcr_lat=self.lower_left_lat,
            datarea_llcr_lon=self.lower_left_lon,
            datarea_urcr_lat=self.upper_right_lat,
            datarea_urcrn_lon=self.upper_right_lon,
        )

    def _write_data(self, stream):
        """Write grid to a GridFile stream."""
        # Only one row and part for grids, so math can be simplified
        for j, col in enumerate(self.column_headers):
            pointer = self.data_block_ptr + j
            stream.jump_to(pointer)
            if col in self.data:
                stream.write_int(self.next_free_word)
                stream.jump_to(self.next_free_word)
                if self.packing_type == PackingType.grib:
                    ref, scale, packed_grid = self._pack_grib(self.data[col], self.precision)
                    lendat = len(packed_grid)
                    stream.write_int(lendat + self._parts_dict['GRID']['header'] + 6)
                    stream.write_int(self.nx)
                    stream.write_int(self.ny)
                    stream.write_int(self.packing_type)
                    stream.write_int(self.precision)
                    stream.write_int(-1)
                    stream.write_int(self.nx * self.ny)
                    stream.write_float(ref)
                    stream.write_float(scale)
                    for cell in packed_grid:
                        stream.write_int(cell)
                else:
                    raise NotImplementedError(
                        f'Packing method {self.packing_type} not currently implemented.'
                    )
                self.next_free_word = stream.word()
            else:
                stream.write_int(0)

        _rec, word = self._dmword(stream.word())
        # Fill out the remainder of the block, if needed
        if word != 1:
            for _n in range(MBLKSZ - word + 1):
                stream.write_int(0)


class SoundingFile(DataManagementFile):
    """GEMPAK sounding file class.

    This class is used to build a collection of soundings to write to disk
    as a GEMPAK sounding file.
    """

    def __init__(self, parameters, pack_data=False):
        """Instantiate SoundingFile.

        Parameters
        ----------
        parameters : array_like
            Set the parameters that each sounding in the file will include.
            GEMPAK files are structured such that parameters cannot change
            between individual soundings within a single file.

            A typical observed sounding in GEMPAK will have PRES, HGHT, TEMP,
            DWPT, DRCT (wind direction), and SPED (wind speed). Model soundings
            will often have several other parameters. A minimal sounding would
            contain a vertical coordinate variable and at least one data parameter.
            A more comprehensive list of sounding parameters can be found in the
            GEMPAK SNPARM documentation.

        pack_data : bool
            Toggle data packing (i.e., real numbers packed as integers).
            Currently not implemented.
        """
        super().__init__()
        self.file_type = FileTypes.sounding
        self.data_source = DataSource.raob_buoy

        if pack_data:
            self._data_type = DataTypes.realpack
        else:
            self._data_type = DataTypes.real

        self.row_names = ['DATE', 'TIME']
        self.column_names = ['STID', 'STNM', 'SLAT', 'SLON', 'SELV', 'STAT', 'COUN', 'STD2']
        self._init_headers()

        self._add_parameters(parameters)

        self._parts_dict = {
            'SNDT': {
                'header': 1,
                'type': self._data_type,
                'parameters': self.parameter_names,
                'scale': [0] * len(self.parameter_names),
                'offset': [0] * len(self.parameter_names),
                'bits': [0] * len(self.parameter_names),
            }
        }

        if pack_data:
            raise NotImplementedError('Data packing not implemented.')

    def _add_parameters(self, parameters):
        """Add parameters to sounding file."""
        if not isinstance(parameters, list | tuple | np.ndarray):
            raise TypeError('parameters should be array-like.')

        if len(parameters) + len(self.parameter_names) > MMPARM:
            raise ValueError('Reached maximum parameter limit.')

        for param in parameters:
            if param not in self.parameter_names:
                self.parameter_names.append(param.upper())

        self._param_args = namedtuple('Parameters', self.parameter_names)

    @staticmethod
    def _validate_length(data_dict):
        """Validate sounding parameters are of same length."""
        sz = len(data_dict[next(iter(data_dict))])
        return all(len(x) == sz for x in data_dict.values()) and sz < MAX_LEVELS

    def add_sounding(self, data, slat, slon, date_time, station_info=None):
        """Add sounding to the file.

        Parameters
        ----------
        data : dict
            Dictionary where keys are parameter names and values are the
            associated data. Keys must match those that the sounding
            file is created with. Should data contain `nan` values, they
            will be replaced with the GEMPAK missing value -9999.

        slat : float
            Site latitude.

        slon : float
            Site longitude.

        date_time : str or datetime
            Sounding date and time. Valid string formats are YYYYmmddHHMM
            or YYYYmmddHHMMFx, where x is the forecast hour from the preceding
            date and time.

        station_info : dict or None
            A dictionary that contains station metadata. Valid keys are
            station_id (e.g., KMSN), station_number (typically WMO ID),
            elevation (m), state, and country.

        Notes
        -----
        If a sounding with duplicate metadata (date, time, site, etc.) is added,
        it will replace the previous entry in the file.
        """
        if not isinstance(data, dict):
            raise TypeError('data must be a dict.')

        data = {k.upper(): self._replace_nan(np.asarray(v.copy())) for k, v in data.items()}
        params = self._param_args(**data)

        if not self._validate_length(data):
            raise ValueError('All input data must be same length.')

        if not isinstance(slat, int | float):
            raise TypeError('Coordinates must be int/float.')

        if not isinstance(slon, int | float):
            raise TypeError('Coordinates must be int/float.')

        if isinstance(date_time, str):
            date_time = date_time.upper()
            if 'F' in date_time:
                init, fhr = date_time.split('F')
                self._datetime = datetime.strptime(init, '%Y%m%d%H%M') + timedelta(
                    hours=int(fhr)
                )
            else:
                self._datetime = datetime.strptime(date_time, '%Y%m%d%H%M')
        elif isinstance(date_time, datetime):
            self._datetime = date_time
        else:
            raise TypeError('date_time must be string or datetime.')

        if station_info is None:
            stid = ''
            stnm = self._missing_int
            selv = self._missing_int
            stat = ''
            coun = ''
            std2 = ''
        else:
            stid = str(station_info.get('station_id', '')).upper()
            stnm = int(station_info.get('station_number', self._missing_int))
            selv = int(station_info.get('elevation', self._missing_int))
            stat = str(station_info.get('state', '')).upper()[:4]
            coun = str(station_info.get('country', '')).upper()[:4]
            std2 = ''

            if len(stid) > 4:
                std2 = stid[4:8]
                stid = stid[:4]

        new_row = (self._datetime.date(), self._datetime.time())
        self._row_set.add(self.make_row_header(*new_row))

        new_column = (stid, stnm, slat, slon, selv, stat, coun, std2)
        self._column_set.add(self.make_column_header(*new_column))

        self.data[(new_row, new_column)] = params._asdict()

        self.rows = len(self._row_set)
        self.columns = len(self._column_set)

        if self.rows + self.columns > MMHDRS:
            raise ValueError('Exceeded maximum number data entries.')

        self.column_headers = sorted(self._column_set)
        self.row_headers = sorted(self._row_set)

    def _write_row_headers(self, stream):
        """Write row headers to a SoundingFile stream."""
        start_word = stream.word()
        # Initialize all headers to unused state
        for _r in range(self.rows):
            for _k in range(self.row_keys + 1):
                stream.write_int(self._missing_int)

        stream.jump_to(start_word)
        for rh in self.row_headers:
            stream.write_int(-self._missing_int)
            for key in rh._fields:
                dtype = HEADER_DTYPE[key]
                if 's' in dtype:
                    stream.write_string(getattr(rh, key))
                elif 'i' in dtype:
                    if key == 'DATE':
                        idate = int(rh.DATE.strftime('%y%m%d'))
                        stream.write_int(idate)
                    elif key == 'TIME':
                        itime = int(rh.TIME.strftime('%H%M'))
                        stream.write_int(itime)
                    else:
                        stream.write_int(getattr(rh, key))

    def _write_column_headers(self, stream):
        """Write column headers to a SoundingFile stream."""
        start_word = stream.word()
        # Initialize all headers to unused state
        for _c in range(self.columns):
            for _k in range(self.column_keys + 1):
                stream.write_int(self._missing_int)

        stream.jump_to(start_word)
        for ch in self.column_headers:
            stream.write_int(-self._missing_int)
            for key in ch._fields:
                dtype = HEADER_DTYPE[key]
                if 's' in dtype:
                    stream.write_string(getattr(ch, key))
                elif 'i' in dtype:
                    if key in ['SLAT', 'SLON']:
                        coord = int(getattr(ch, key) * 100)
                        stream.write_int(coord)
                    else:
                        stream.write_int(getattr(ch, key))

    def _write_data(self, stream):
        """Write sounding to a SoundingFile stream."""
        for j, col in enumerate(self.column_headers):
            for i, row in enumerate(self.row_headers):
                # Only 1 part for merged sounding, so math can be simplified
                pointer = self.data_block_ptr + i * self.columns + j
                stream.jump_to(pointer)
                if (row, col) in self.data:
                    stream.write_int(self.next_free_word)
                    params = self.data[(row, col)]
                    # Need data length, so just grab first parameter
                    lendat = self._parts_dict['SNDT']['header'] + len(
                        params[next(iter(params))]
                    ) * len(params)
                    itime = int(row.TIME.strftime('%H%M'))
                    stream.jump_to(self.next_free_word)
                    stream.write_int(lendat)
                    stream.write_int(itime)

                    for rec in zip(*params.values(), strict=True):
                        if self._data_type == DataTypes.realpack:
                            for _param, _pval in zip(self.parameter_names, rec, strict=True):
                                # pack values
                                pass
                        else:
                            for pval in rec:
                                stream.write_float(pval)
                    # Update the next free word after writing data.
                    self.next_free_word = stream.word()

                else:
                    stream.write_int(0)

        _rec, word = self._dmword(stream.word())
        # Fill out the remainder of the block, if needed
        if word != 1:
            for _n in range(MBLKSZ - word + 1):
                stream.write_int(0)


class SurfaceFile(DataManagementFile):
    """GEMPAK standard surface file class.

    This class is used to build a collection of surface stations to write to disk
    as a GEMPAK surface file.
    """

    def __init__(self, parameters, pack_data=False):
        """Instantiate SurfaceFile.

        Parameters
        ----------
        parameters : array_like
            Set the parameters that each surface station in the file will include.
            GEMPAK files are structured such that parameters cannot change
            between individual stations within a single file.

            Common parameters for surface files include PMSL, ALTI, WNUM, CHC[1-3],
            TMPC, DWPC, SKNT, DRCT, and VSBY. See GEMPAK SFPARM documentation for
            a more comprehensive list of parameters.

        pack_data : bool
            Toggle data packing (i.e., real numbers packed as integers).
            Currently not implemented.

        Notes
        -----
        Creates a standard surface file that does not contain any text
        (e.g., TEXT/SPCL parameters).
        """
        super().__init__()
        self.file_type = FileTypes.surface
        self.data_source = DataSource.metar

        if pack_data:
            self._data_type = DataTypes.realpack
        else:
            self._data_type = DataTypes.real

        self.row_names = ['DATE', 'TIME']
        self.column_names = [
            'STID',
            'STNM',
            'SLAT',
            'SLON',
            'SELV',
            'STAT',
            'COUN',
            'STD2',
            'SPRI',
        ]
        self._init_headers()

        self._add_parameters(parameters)

        self._parts_dict = {
            'SFDT': {
                'header': 1,
                'type': self._data_type,
                'parameters': self.parameter_names,
                'scale': [0] * len(self.parameter_names),
                'offset': [0] * len(self.parameter_names),
                'bits': [0] * len(self.parameter_names),
            }
        }

        if pack_data:
            raise NotImplementedError('Data packing not implemented.')

    def _add_parameters(self, parameters):
        """Add parameters to surface file."""
        if not isinstance(parameters, list | tuple | np.ndarray):
            raise TypeError('parameters should be array-like.')

        if len(parameters) + len(self.parameter_names) > MMPARM:
            raise ValueError('Reached maximum parameter limit.')

        for param in parameters:
            if param not in self.parameter_names:
                self.parameter_names.append(param.upper())

        self._param_args = namedtuple('Parameters', self.parameter_names)

    @staticmethod
    def _validate_length(data_dict):
        """Validate surface parameters are of same length."""
        return all(np.isscalar(x) for x in data_dict.values())

    def add_station(self, data, slat, slon, date_time, station_info=None):
        """Add station to the file.

        Parameters
        ----------
        data : dict
            Dictionary where keys are parameter names and values are the
            associated (scalar) data. Keys must match those that the surface
            file is created with. Should data contain `nan` values, they
            will be replaced with the GEMPAK missing value -9999.

        slat : float
            Station latitude.

        slon : float
            Station longitude.

        date_time : str or datetime
            Observation date and time. Valid string formats are YYYYmmddHHMM.

        station_info : dict or None
            A dictionary that contains station metadata. Valid keys are
            station_id (e.g., KMSN), station_number (typically WMO ID),
            elevation (m), state, country, and priority.

        Notes
        -----
        If a station with duplicate metadata (date, time, site, etc.) is added,
        it will replace the previous entry in the file.
        """
        if not isinstance(data, dict):
            raise TypeError('data must be a dict.')

        data = {k.upper(): self._missing_float if np.isnan(v) else v for k, v in data.items()}
        params = self._param_args(**data)

        if not self._validate_length(data):
            raise ValueError('All input data must be same length.')

        if not isinstance(slat, int | float):
            raise TypeError('Coordinates must be int/float.')

        if not isinstance(slon, int | float):
            raise TypeError('Coordinates must be int/float.')

        if isinstance(date_time, str):
            date_time = date_time.upper()
            self._datetime = datetime.strptime(date_time, '%Y%m%d%H%M')
        elif isinstance(date_time, datetime):
            self._datetime = date_time
        else:
            raise TypeError('date_time must be string or datetime.')

        if station_info is None:
            stid = ''
            stnm = self._missing_int
            selv = self._missing_int
            stat = ''
            coun = ''
            std2 = ''
            spri = 0
        else:
            stid = str(station_info.get('station_id', '')).upper()
            stnm = int(station_info.get('station_number', self._missing_int))
            selv = int(station_info.get('elevation', self._missing_int))
            stat = str(station_info.get('state', '')).upper()[:4]
            coun = str(station_info.get('country', '')).upper()[:4]
            std2 = ''
            spri = station_info.get('priority', 0)

            if len(stid) > 4:
                std2 = stid[4:8]
                stid = stid[:4]

        new_row = (date_time.date(), date_time.time())
        self._row_set.add(self.make_row_header(*new_row))

        new_column = (stid, stnm, slat, slon, selv, stat, coun, std2, spri)
        self._column_set.add(self.make_column_header(*new_column))

        self.data[(new_row, new_column)] = params._asdict()

        self.rows = len(self._row_set)
        self.columns = len(self._column_set)

        if self.rows + self.columns > MMHDRS:
            raise ValueError('Exceeded maximum number data entries.')

        self.column_headers = sorted(self._column_set)
        self.row_headers = sorted(self._row_set)

    def _write_row_headers(self, stream):
        """Write row headers to a SurfaceFile stream."""
        start_word = stream.word()
        # Initialize all headers to unused state
        for _r in range(self.rows):
            for _k in range(self.row_keys + 1):
                stream.write_int(self._missing_int)

        stream.jump_to(start_word)
        for rh in self.row_headers:
            stream.write_int(-self._missing_int)
            for key in rh._fields:
                dtype = HEADER_DTYPE[key]
                if 's' in dtype:
                    stream.write_string(getattr(rh, key))
                elif 'i' in dtype:
                    if key == 'DATE':
                        idate = int(rh.DATE.strftime('%y%m%d'))
                        stream.write_int(idate)
                    elif key == 'TIME':
                        itime = int(rh.TIME.strftime('%H%M'))
                        stream.write_int(itime)
                    else:
                        stream.write_int(getattr(rh, key))

    def _write_column_headers(self, stream):
        """Write column headers to a SurfaceFile stream."""
        start_word = stream.word()
        # Initialize all headers to unused state
        for _c in range(self.columns):
            for _k in range(self.column_keys + 1):
                stream.write_int(self._missing_int)

        stream.jump_to(start_word)
        for ch in self.column_headers:
            stream.write_int(-self._missing_int)
            for key in ch._fields:
                dtype = HEADER_DTYPE[key]
                if 's' in dtype:
                    stream.write_string(getattr(ch, key))
                elif 'i' in dtype:
                    if key in ['SLAT', 'SLON']:
                        coord = int(getattr(ch, key) * 100)
                        stream.write_int(coord)
                    else:
                        stream.write_int(getattr(ch, key))

    def _write_data(self, stream):
        """Write sounding to a SurfaceFile stream."""
        for i, row in enumerate(self.row_headers):
            for j, col in enumerate(self.column_headers):
                # Only 1 part for merged sounding, so math can be simplified
                pointer = self.data_block_ptr + i * self.columns + j
                stream.jump_to(pointer)
                if (row, col) in self.data:
                    stream.write_int(self.next_free_word)
                    params = self.data[(row, col)]
                    # Need data length, so just grab first parameter
                    lendat = self._parts_dict['SFDT']['header'] + 1 * len(params)
                    itime = int(row.TIME.strftime('%H%M'))
                    stream.jump_to(self.next_free_word)
                    stream.write_int(lendat)
                    stream.write_int(itime)

                    for _param, pval in params.items():
                        if self._data_type == DataTypes.realpack:
                            pass
                        else:
                            stream.write_float(pval)
                    # Update the next free word after writing data.
                    self.next_free_word = stream.word()

                else:
                    stream.write_int(0)

        _rec, word = self._dmword(stream.word())
        # Fill out the remainder of the block, if needed
        if word != 1:
            for _n in range(MBLKSZ - word + 1):
                stream.write_int(0)
