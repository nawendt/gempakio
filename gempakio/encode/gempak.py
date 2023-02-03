# Copyright (c) 2023 Nathan Wendt.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Classes for encoding various GEMPAK file formats."""

from collections import namedtuple
from datetime import datetime, timedelta
from io import BytesIO
import struct

import numpy as np

from gempakio.common import (_position_to_word, _word_to_position, DataSource, DataTypes,
                             FileTypes, GEMPAK_HEADER, HEADER_DTYPE, MBLKSZ, MISSING_FLOAT,
                             MISSING_INT, MMFREE, MMHDRS, MMPARM)
from gempakio.tools import NamedStruct


class GempakStream(BytesIO):
    """In-memory bytes stream for GEMPAK data."""

    def __init__(self):
        super().__init__()

    def jump_to(self, word):
        """Jumpt to given word."""
        # word - 1 to get word start, not end
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

    label_struct = NamedStruct(
        [('dm_head', '28s'), ('version', 'i'), ('file_headers', 'i'), ('file_keys_ptr', 'i'),
         ('rows', 'i'), ('row_keys', 'i'), ('row_keys_ptr', 'i'), ('row_headers_ptr', 'i'),
         ('columns', 'i'), ('column_keys', 'i'), ('column_keys_ptr', 'i'),
         ('column_headers_ptr', 'i'), ('parts', 'i'), ('parts_ptr', 'i'),
         ('data_mgmt_ptr', 'i'), ('data_mgmt_length', 'i'), ('data_block_ptr', 'i'),
         ('file_type', 'i'), ('data_source', 'i'), ('machine_type', 'i'), ('missing_int', 'i'),
         (None, '12x'), ('missing_float', 'f')], '<', 'Label'
    )

    data_mgmt_struct = NamedStruct(
        [('next_free_word', 'i'), ('max_free_pairs', 'i'), ('actual_free_pairs', 'i'),
         ('last_word', 'i'), (None, '464x')], '<', 'DataManagement'
    )

    def __init__(self):
        self.version = 1
        self.machine_type = 11
        self.missing_int = MISSING_INT
        self.missing_float = MISSING_FLOAT
        self.data_mgmt_ptr = 129
        self.data_mgmt_length = MBLKSZ
        self.max_free_pairs = MMFREE
        self.actual_free_pairs = 0
        self.last_word = 0
        self.parameter_names = []
        self.row_names = []
        self.column_names = []
        self.rows = 0
        self.columns = 0
        self.file_headers = []
        self.parts_dict = {}
        self.file_type = None
        self.data_source = None
        self.column_headers = set()
        self.row_headers = set()
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

    def _init_headers(self):
        self.make_column_header = namedtuple('ColumnHeader', self.column_names)
        self.make_row_header = namedtuple('RowHeader', self.row_names)

    def _set_pointers(self):
        """Set pointers for the output file.

        Notes
        -----
        See GEMPAK function DM_CRET.
        """
        # Keys
        self.row_keys = len(self.row_names)
        self.row_keys_ptr = self.data_mgmt_ptr + self.data_mgmt_length

        self.column_keys = len(self.column_names)
        self.column_keys_ptr = self.row_keys_ptr + self.row_keys

        # Headers
        self.file_keys_ptr = self.column_keys_ptr + self.column_keys
        lenfil = 0
        for _fh, info in self.file_headers:
            lenfil += (info['length'] + 1)
        rec, word = self._dmword(self.file_keys_ptr + 3 * len(self.file_headers) + lenfil)
        if word != 1:
            self.row_headers_ptr = rec * MBLKSZ + 1
        else:
            self.row_headers_ptr = self.file_keys_ptr + 3 * len(self.file_headers) + lenfil
        self.column_headers_ptr = self.row_headers_ptr + self.rows * (self.row_keys + 1)

        # Parts
        lenpart = 0
        nparts = len(self.parts_dict)
        for _part, info in self.parts_dict.items():
            lenpart += len(info['parameters'])
        rec, word = self._dmword(
            self.column_headers_ptr + self.columns * (self.column_keys + 1)
        )
        if word != 1:
            self.parts_ptr = rec * MBLKSZ + 1
        else:
            self.parts_ptr = self.column_headers_ptr + self.columns * (self.column_keys + 1)

        # Data
        rec, word = self._dmword(
            self.parts_ptr + 4 * nparts + 4 * lenpart
        )
        if word != 1:
            self.data_block_ptr = rec * MBLKSZ + 1
        else:
            self.data_block_ptr = self.parts_ptr + 4 * nparts + 4 * lenpart

        # Data Management (initial next free word)
        rec, word = self._dmword(
            self.data_block_ptr + nparts * self.rows * self.columns
        )
        if word != 1:
            self.next_free_word = rec * MBLKSZ + 1
        else:
            self.next_free_word = self.data_block_ptr + nparts * self.rows * self.columns

    def _write_label(self, stream):
        """Write file label to a stream."""
        stream.write_struct(
            self.label_struct,
            dm_head=bytes(GEMPAK_HEADER, 'utf-8'),
            version=self.version,
            file_headers=len(self.file_headers),
            file_keys_ptr=self.file_keys_ptr,
            rows=self.rows,
            row_keys=self.row_keys,
            row_keys_ptr=self.row_keys_ptr,
            row_headers_ptr=self.row_headers_ptr,
            columns=self.columns,
            column_keys=self.column_keys,
            column_keys_ptr=self.column_keys_ptr,
            column_headers_ptr=self.column_headers_ptr,
            parts=len(self.parts_dict),
            parts_ptr=self.parts_ptr,
            data_mgmt_ptr=self.data_mgmt_ptr,
            data_mgmt_length=self.data_mgmt_length,
            data_block_ptr=self.data_block_ptr,
            file_type=self.file_type,
            data_source=self.data_source,
            machine_type=self.machine_type,
            missing_int=self.missing_int,
            missing_float=self.missing_float
        )

    def _write_data_management(self, stream):
        """Write data management block to a stream."""
        stream.write_struct(
            self.data_mgmt_struct,
            next_free_word=self.next_free_word,
            max_free_pairs=self.max_free_pairs,
            actual_free_pairs=self.actual_free_pairs,
            last_word=self.last_word
        )

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
        for name in self.parts_dict:
            stream.write_string(name)

        for _name, info in self.parts_dict.items():
            stream.write_int(info['header'])

        for _name, info in self.parts_dict.items():
            stream.write_int(info['type'])

        for _name, info in self.parts_dict.items():
            stream.write_int(len(info['parameters']))

        for _name, info in self.parts_dict.items():
            for param in info['parameters']:
                stream.write_string(param)

        for _name, info in self.parts_dict.items():
            for offset in info['offset']:
                stream.write_int(offset)

        for _name, info in self.parts_dict.items():
            for scale in info['scale']:
                stream.write_int(scale)

        for _name, info in self.parts_dict.items():
            for bits in info['bits']:
                stream.write_int(bits)

    def _write_row_headers(self):
        raise NotImplementedError('Must be defined within subclass.')

    def _write_column_headers(self):
        raise NotImplementedError('Must be defined within subclass.')

    def _write_data(self):
        raise NotImplementedError('Must be defined within subclass.')


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

        pack_data : bool
            Toggle data packing (i.e., real numbers packed as integers.).
            Currently not implemented.
        """
        super().__init__()
        self.file_type = FileTypes.sounding.value
        self.data_source = DataSource.raob_buoy.value

        if pack_data:
            self._data_type = DataTypes.realpack.value
        else:
            self._data_type = DataTypes.real.value

        self.row_names = ['DATE', 'TIME']
        self.column_names = ['STID', 'STNM', 'SLAT', 'SLON', 'SELV',
                             'STAT', 'COUN', 'STD2']
        self._init_headers()

        self._add_parameters(parameters)

        self.parts_dict = {
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
        if not isinstance(parameters, (list, tuple, np.ndarray)):
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
        return all(len(x) == sz for x in data_dict.values())

    def _replace_nan(self, array):
        """Replace nan values from an array with missing value."""
        nan_loc = np.isnan(array)
        array = array[nan_loc] = self.missing_float

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

        data = {k.upper(): self._replace_nan(np.asarray(v)) for k, v in data.items()}
        params = self._param_args(**data)

        if not self._validate_length(data):
            raise ValueError('All input data must be same length.')            

        if not isinstance(slat, (int, float)):
            raise TypeError('Coordinates must be int/float.')

        if not isinstance(slon, (int, float)):
            raise TypeError('Coordinates must be int/float.')

        if isinstance(date_time, str):
            date_time = date_time.upper()
            if 'F' in date_time:
                init, fhr = date_time.split('F')
                self._datetime = (datetime.strptime(init, '%Y%m%d%H%M')
                                  + timedelta(hours=int(fhr)))
            else:
                self._datetime = datetime.strptime(date_time, '%Y%m%d%H%M')
        elif isinstance(date_time, datetime):
            self._datetime = date_time
        else:
            raise TypeError('date_time must be string or datetime.')

        if station_info is None:
            stid = ''
            stnm = self.missing_int
            selv = self.missing_int
            stat = ''
            coun = ''
            std2 = ''
        else:
            stid = str(station_info.get('station_id', '')).upper()
            stnm = int(station_info.get('station_number', self.missing_int))
            selv = int(station_info.get('elevation', self.missing_int))
            stat = str(station_info.get('state', '')).upper()[:4]
            coun = str(station_info.get('country', '')).upper()[:4]
            std2 = ''

            if len(stid) > 4:
                std2 = stid[4:8]
                stid = stid[:4]

        new_row = (date_time.date(), date_time.time())
        self.row_headers.add(self.make_row_header(*new_row))

        new_column = (stid, stnm, slat, slon, selv, stat, coun, std2)
        self.column_headers.add(self.make_column_header(*new_column))

        self.data[(new_row, new_column)] = params._asdict()

        self.rows = len(self.row_headers)
        self.columns = len(self.column_headers)

        if self.rows + self.columns > MMHDRS:
            raise ValueError('Exceeded maximum number data entries.')

    def _write_row_headers(self, stream):
        """Write row headers to a stream."""
        start_word = stream.word()
        # Initialize all headers to unused state
        for _r in range(self.rows):
            for _k in range(self.row_keys + 1):
                stream.write_int(self.missing_int)

        stream.jump_to(start_word)
        for rh in sorted(self.row_headers):
            stream.write_int(-self.missing_int)
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
        """Write column headers to a stream."""
        start_word = stream.word()
        # Initialize all headers to unused state
        for _c in range(self.columns):
            for _k in range(self.column_keys + 1):
                stream.write_int(self.missing_int)

        stream.jump_to(start_word)
        for ch in self.column_headers:
            stream.write_int(-self.missing_int)
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
        """Write sounding to a stream."""
        for i, row in enumerate(self.row_headers):
            for j, col in enumerate(self.column_headers):
                pointer = self.data_block_ptr + i * self.columns + j
                stream.jump_to(pointer)
                if (row, col) in self.data:
                    stream.write_int(self.next_free_word)
                    params = self.data[(row, col)]
                    # Need data length, so just grab first parameter
                    lendat = (self.parts_dict['SNDT']['header']
                              + len(params[next(iter(params))]) * len(params))
                    itime = int(row.TIME.strftime('%H%M'))
                    stream.jump_to(self.next_free_word)
                    stream.write_int(lendat)
                    stream.write_int(itime)

                    for rec in zip(*params.values()):
                        if self._data_type == DataTypes.realpack.value:
                            for _param, _pval in zip(self.parameter_names, rec):
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

    def write(self, file):
        """Write sounding file to disk.

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
            stream.jump_to(self.row_keys_ptr)
            self._write_row_keys(stream)

            # Write column key names
            stream.jump_to(self.column_keys_ptr)
            self._write_column_keys(stream)

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
            stream.jump_to(self.data_mgmt_ptr)
            self._write_data_management(stream)

            with open(file, 'wb') as out:
                out.write(stream.getbuffer())
