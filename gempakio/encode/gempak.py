# Copyright (c) 2023 Nathan Wendt.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Classes for encoding various GEMPAK file formats."""

from datetime import datetime
from io import BytesIO
import struct

from gempakio.common import DataSource, DataTypes, FileTypes, GEMPAK_HEADER, MBLKSZ, MMFREE
from gempakio.tools import NamedStruct


class GempakStream(BytesIO):
    """In-memory bytes stream for GEMPAK data."""

    def __init__(self):
        super().__init__()


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

    def __init__(self, file):
        self._file = file
        self._fptr = None

    def __enter__(self):
        """Enter context."""
        self._fptr = open(self._file, 'wb')  # noqa: SIM115
        self._tell = 0
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit context."""
        self._fptr.close()
        self._fptr = None
        self._tell = 0

    @property
    def tell(self):
        """Get file pointer location."""
        return self._tell

    @property
    def word(self):
        """Get current word."""
        return self._tell // 4

    def jump_to(self, word):
        """Jumpt to given word."""
        self._fptr.seek(word * 4)
        self._tell = self._fptr.tell()

    def write_string(self, string):
        """Write string word."""
        if len(string) != 4:
            raise ValueError('String must be size 4.')
        self._fptr.write(struct.pack('4s', bytes(string, 'utf-8')))
        self._tell += 4

    def write_int(self, i):
        """Write integer word."""
        self._fptr.write(struct.pack('<i', i))
        self._tell += 4

    def write_float(self, f):
        """Write float word."""
        self._fptr.write(struct.pack('<f', f))
        self._tell += 4

    def write_struct(self, struct_class, **kwargs):
        """Write structure to file as bytes."""
        self._fptr.write(struct_class.pack(**kwargs))
        self._tell += struct_class.size


class GempakData:
    """Base class for encoding user data."""

    def __init__(self):
        self.version = 1
        self.machine_type = 11
        self.missing_int = -9999
        self.missing_float = -9999.0

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


class SoundingFile(GempakData):
    """GEMPAK sounding file class.

    This class is used to build a collection of soundings to write to disk
    as a GEMPAK sounding file.
    """

    def __init__(self, pack_data=False):
        super().__init__()
        self.file_type = FileTypes.sounding.value
        self.data_source = DataSource.raob_buoy.value
        self._soundings = []

        if pack_data:
            self._data_type = DataTypes.realpack.value
        else:
            self._data_type = DataTypes.real.value

    def _set_structure(self):
        """Set pointers for the output file.

        Notes
        -----
        See GEMPAK functions SN_CREF and DM_CRET.
        """
        # Data Management
        self.data_mgmt_ptr = 129
        self.data_mgmt_length = 128
        self.max_free_pairs = MMFREE
        self.actual_free_pairs = 0
        self.last_word = 0

        # Keys
        self.row_names = ['DATE', 'TIME']
        self.rows = 1
        self.row_keys = len(self.row_names)
        self.row_keys_ptr = self.data_mgmt_ptr + self.data_mgmt_length
        self.column_names = ['STID', 'STNM', 'SLAT', 'SLON', 'SELV',
                             'STAT', 'COUN', 'STD2']
        self.columns = 1
        self.column_keys = len(self.column_names)
        self.column_keys_ptr = self.row_keys_ptr + self.row_keys
        self.file_keys_ptr = self.column_keys_ptr + self.column_keys

        # Headers
        self.file_headers = 0
        rec, word = self._dmword(self.file_keys_ptr)
        if word != 1:
            self.row_headers_ptr = rec * MBLKSZ + 1
        else:
            self.row_headers_ptr = self.file_keys_ptr
        self.column_headers_ptr = self.row_headers_ptr + self.rows * (self.row_keys + 1)

        # Parts
        self.parts = 1
        self.part_name = 'SNDT'
        self.part_header = 1
        rec, word = self._dmword(
            self.column_headers_ptr + self.columns * (self.column_keys + 1)
        )
        if word != 1:
            self.parts_ptr = rec * MBLKSZ + 1
        else:
            self.parts_ptr = self.column_headers_ptr + self.columns * (self.column_keys + 1)

        # Data
        rec, word = self._dmword(
            self.parts_ptr + 4 * self.parts + 4 * self._params  # Always 1 part for merged
        )
        if word != 1:
            self.data_block_ptr = rec * MBLKSZ + 1
        else:
            self.data_block_ptr = self.parts_ptr + 4 * self.parts + 4 * self._params

        # Data Management (next free word)
        rec, word = self._dmword(
            self.data_block_ptr + self.parts * self.rows * self.columns
        )
        if word != 1:
            self.next_free_word = rec * MBLKSZ + 1
        else:
            self.next_free_word = self.data_block_ptr + self.parts * self.rows * self.columns

    def add_sounding(self, sounding):
        """Add sounding to the file."""
        if not isinstance(sounding, Sounding):
            raise TypeError('Input sounding must be `Sounding` class.')

    def write(self, file):
        """Write sounding file to disk."""
        self._set_structure()

        with DataManagementFile(file) as dm:
            # Write file label
            dm.write_struct(
                dm.label_struct,
                dm_head=bytes(GEMPAK_HEADER, 'utf-8'),
                version=self.version,
                file_headers=self.file_headers,
                file_keys_ptr=self.file_keys_ptr,
                rows=self.rows,
                row_keys=self.row_keys,
                row_keys_ptr=self.row_headers_ptr,
                row_headers_ptr=self.row_headers_ptr,
                columns=self.columns,
                column_keys=self.column_keys,
                column_keys_ptr=self.column_keys_ptr,
                column_headers_ptr=self.column_headers_ptr,
                parts=self.parts,
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

            # Write data management record
            dm.jump_to(self.data_mgmt_ptr)
            dm.write_struct(
                dm.data_mgmt_struct,
                next_free_word=self.next_free_word,
                max_free_pairs=self.max_free_pairs,
                actual_free_pairs=self.actual_free_pairs,
                last_word=self.last_word
            )

            # Write row key names
            dm.jump_to(self.row_keys_ptr)
            for rn in self.row_names:
                dm.write_string(rn)

            # Write column key names
            dm.jump_to(self.column_keys_ptr)
            for cn in self.column_names:
                dm.write_string(cn)

            # Write parts and parameters
            dm.jump_to(self.parts_ptr)
            dm.write_string(self.part_name)
            dm.write_int(self.part_header)
            dm.write_int(self._data_type)
            # dm.write_int()

            # Write row headers
            dm.jump_to(self.row_headers_ptr)


class Sounding:
    """User sounding data class."""

    def __init__(self, pres, temp, dwpt, hght, drct, sped, slat, slon,
                 date_time, station_info=None):
        self._params = 0
        self._data = {}
        self._lat = slat
        self._lon = slon

        if station_info is None:
            self._stid = ''
            self._stnm = self.missing_int
            self._selv = self.missing_int
            self._stat = ''
            self._coun = ''
            self._std2 = ''
        else:
            self._stid = str(station_info.pop('station_id', '')).upper()
            self._stnm = int(station_info.pop('station_number', self.missing_int))
            self._selv = int(station_info.pop('elevation', self.missing_int))
            self._stat = str(station_info.pop('state', '')).upper()[:4]
            self._coun = str(station_info.pop('country', '')).upper()[:4]
            self._std2 = ''

            if len(self._stid) > 4:
                self._std2 = self._stid[4:8]
                self._stid = self._stid[:4]

        if isinstance(date_time, str):
            self._datetime = datetime.strftime('%Y%m%d%H%M')
        elif isinstance(date_time, datetime):
            self._datetime = date_time
        else:
            raise TypeError('date_time must be string or datetime.')

        # Check input data
        input_len = [len(x) for x in [pres, temp, dwpt, hght, drct, sped]]
        if input_len.count(len(pres)) != 6:
            raise ValueError('All input data must be same length.')

        # QC input data

        # Add input data parameters
        self.add_parameters(
            {
                'PRES': pres,
                'HGHT': hght,
                'TEMP': temp,
                'DWPT': dwpt,
                'DRCT': drct,
                'SPED': sped
            }
        )

    def _qc_input(self):
        """QC the required input data."""

    def add_parameters(self, parameters):
        """Add parameters to sounding."""
        for key, values in parameters.items():
            if key not in self._data:
                self._params += 1
            self._data[key.upper()] = values

    def remove_parameters(self, parameters):
        """Remove parameters from sounding."""
        if not isinstance(parameters, list):
            parameters = [parameters]

        for param in parameters:
            if param.upper() in self._data:
                self._params -= 1
                self._data.pop(param.upper())
