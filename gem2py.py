"""Tools to process GEMPAK-formatted products."""

import contextlib
from datetime import datetime
from enum import Enum
from io import BytesIO
from itertools import cycle, product, repeat
import logging
import re

import numpy as np
from xarray import Variable
from xarray.backends.common import AbstractDataStore
try:
    from xarray.core.utils import FrozenDict
except ImportError:
    from xarray.core.utils import FrozenOrderedDict as FrozenDict

from metpy.io._tools import Bits, DictStruct, IOBuffer, NamedStruct, open_as_needed

ANLB_SIZE = 128
BYTES_PER_WORD = 4
NAVB_SIZE = 256

def _word_to_position(word, bytes_per_word=BYTES_PER_WORD):
    r"""Return beginning position of a word in bytes"""
    return (word * bytes_per_word) - bytes_per_word


class GempakDM(AbstractDataStore):
    """
    """

    gempak_header = 'GEMPAK DATA MANAGEMENT FILE '
    
    prod_desc_fmt = NamedStruct([('version', 'i'), ('file_headers', 'i'),
                                ('file_keys_ptr', 'i'), ('rows', 'i'),
                                ('row_keys', 'i'), ('row_keys_ptr','i'),
                                ('row_headers_ptr', 'i'), ('columns', 'i'),
                                ('column_keys', 'i'), ('column_keys_ptr','i'),
                                ('column_headers_ptr', 'i'), ('parts', 'i'),
                                ('parts_ptr', 'i'), ('data_mgmt_ptr', 'i'),
                                ('data_mgmt_length', 'i'), ('data_block_ptr', 'i'),
                                ('file_type', 'i'), ('data_source', 'i'),
                                ('machine_type', 'i'), ('missing_int', 'i'),
                                (None, '12x'), ('missing_float' ,'f')])
    
    grid_nav_fmt = NamedStruct([('grid_definition_type', 'f'), ('projection', '3sx'),
                                ('left_grid_number', 'f'), ('bottom_grid_number', 'f'),
                                ('right_grid_number', 'f'), ('top_grid_number', 'f'),
                                ('lower_left_lat', 'f'), ('lower_left_lon', 'f'),
                                ('upper_right_lat', 'f'), ('upper_right_lon', 'f'),
                                ('proj_angle1', 'f'), ('proj_angle2', 'f'),
                                ('proj_angle3', 'f'), (None, '972x')])

    grid_anl_fmt1 = NamedStruct([('analysis_type', 'f'), ('delta_n', 'f'),
                                 ('delta_x', 'f'), ('delta_y', 'f'),
                                 (None, '4x'), ('garea_llcr_lat', 'f'),
                                 ('garea_llcr_lon', 'f'), ('garea_urcr_lat', 'f'),
                                 ('garea_urcr_lon', 'f'), ('extarea_llcr_lat', 'f'),
                                 ('extarea_llcr_lon', 'f'), ('extarea_urcr_lat', 'f'),
                                 ('extarea_urcr_lon', 'f'), ('datarea_llcr_lat', 'f'),
                                 ('datarea_llcr_lon', 'f'), ('datarea_urcr_lat', 'f'),
                                 ('datarea_urcrn_lon', 'f'), (None, '444x')])

    grid_anl_fmt2 = NamedStruct([('analysis_type', 'f'), ('delta_n', 'f'),
                                 ('grid_ext_left', 'f'), ('grid_ext_down', 'f'),
                                 ('grid_ext_right', 'f'), ('grid_ext_up', 'f'),
                                 ('garea_llcr_lat', 'f'), ('garea_llcr_lon', 'f'),
                                 ('garea_urcr_lat', 'f'), ('garea_urcr_lon', 'f'),
                                 ('extarea_llcr_lat', 'f'), ('extarea_llcr_lon', 'f'),
                                 ('extarea_urcr_lat', 'f'), ('extarea_urcr_lon', 'f'),
                                 ('datarea_llcr_lat', 'f'), ('datarea_llcr_lon', 'f'),
                                 ('datarea_urcr_lat', 'f'), ('datarea_urcrn_lon', 'f'),
                                 (None, '440x')])

    def __init__(self, filename):
        """
        """
        fobj = open_as_needed(filename)

        with contextlib.closing(fobj):
            self._buffer = IOBuffer.fromfile(fobj)

        # Save file start position as pointers use this as reference
        start = self._buffer.set_mark()

        # Process the main GEMPAK header to verify file format
        self._process_gempak_header()

        # Process main metadata header
        self.prod_desc = self._buffer.read_struct(self.prod_desc_fmt)
        meta = self._buffer.set_mark()

        # FILE KEYS
        # Surface and upper-air files will not have the file headers, so we need to check for that.
        if self.prod_desc.file_headers > 0:
            # Should this be so general or can we rely on there only ever being NAVB and ANLB?
            # This will grab all headers, but we will forego processing all but NAVB and ANLB for now.
            fkey_prod = product(['header_name', 'header_length', 'header_type'], range(1, self.prod_desc.file_headers + 1))
            fkey_names = [ x for x in [ '{}{}'.format(*x) for x in fkey_prod ] ]
            fkey_info = list(zip(fkey_names, np.repeat(('4s', 'i', 'i'), self.prod_desc.file_headers)))
            self.file_keys_format = NamedStruct(fkey_info)

            self._buffer.jump_to(start, _word_to_position(self.prod_desc.file_keys_ptr))
            self.file_keys = self._buffer.read_struct(self.file_keys_format)

            file_key_blocks = self._buffer.set_mark()
            # We are now at the NAVB/ANLB portion of the file.
            # NAVB
            navb_size = self._buffer.read_int('i')
            if navb_size != NAVB_SIZE:
                raise ValueError('Navigation block size does not match GEMPAK specification')
            else:
                self.navigation_block = self._buffer.read_struct(self.grid_nav_fmt)
            
            # ANLB
            anlb_size = self._buffer.read_int('i')
            anlb_start = self._buffer.set_mark()
            if anlb_size != ANLB_SIZE:
                raise ValueError('Analysis block size does not match GEMPAK specification')
            else:
                anlb_type = self._buffer.read_int('f')
                self._buffer.jump_to(anlb_start)
                if anlb_type == 1:
                    self.analysis_block = self._buffer.read_struct(self.grid_anl_fmt1)
                elif anlb_type == 2:
                    self.analysis_block = self._buffer.read_struct(self.grid_anl_fmt2)

            # We are neglecting other file keys at this time
        else:
            self.analysis_block = None
            self.navigation_block = None

        # ROW KEYS
        self._buffer.jump_to(start, _word_to_position(self.prod_desc.row_keys_ptr))
        row_key_info = [ ('row_key{:d}'.format(n), '4s') for n in range(1, self.prod_desc.row_keys + 1) ]
        row_keys_fmt = NamedStruct(row_key_info)
        self.row_keys = self._buffer.read_struct(row_keys_fmt)

        # COL KEYS
        self._buffer.jump_to(start, _word_to_position(self.prod_desc.column_keys_ptr))
        column_key_info = [ ('column_key{:d}'.format(n), '4s') for n in range(1, self.prod_desc.column_keys + 1) ]
        column_keys_fmt = NamedStruct(column_key_info)
        self.column_keys = self._buffer.read_struct(column_keys_fmt)

        # Based on GEMPAK source, row/col headers have a 0th element in their Fortran arrays.
        # This appears to be a flag value to say a header is used or not. Will account for that
        # here. Do we need it?
        # ROW HEADERS
        self._buffer.jump_to(start, _word_to_position(self.prod_desc.row_headers_ptr))
        # Just going to use a list of NamedStructs to store this now. Probably is a better way to do this.
        self.row_headers = []
        for n in range(1, self.prod_desc.rows + 1):
            row_headers_info = [ ('row_header{:d}'.format(n), 'i') for n in range(0, self.prod_desc.row_keys + 1) ]
            row_headers_fmt = NamedStruct(row_headers_info)
            self.row_headers.append(self._buffer.read_struct(row_headers_fmt))

        # COL HEADERS
        self._buffer.jump_to(start, _word_to_position(self.prod_desc.column_headers_ptr))
        self.column_headers = []
        for n in range(1, self.prod_desc.columns + 1):
            column_headers_info = [ ('column_header{:d}'.format(n), 'i') for n in range(0, self.prod_desc.column_keys + 1) ]
            column_headers_fmt = NamedStruct(column_headers_info)
            self.column_headers.append(self._buffer.read_struct(column_headers_fmt))

        # PARTS
        self._buffer.jump_to(start, _word_to_position(self.prod_desc.parts_ptr))
        parts_info = [ ('parts_key{:d}'.format(n), '4s') for n in range(1, self.prod_desc.parts + 1) ]
        parts_fmt = NamedStruct(parts_info)
        self.parts = self._buffer.read_struct(parts_fmt)

    def _process_gempak_header(self):
        """Read off the GEMPAK header from the file, if necessary."""
        data = self._buffer.get_next(len(self.gempak_header)).decode('utf-8', 'ignore')
        if data == self.gempak_header:
            self._buffer.skip(len(self.gempak_header))
        else:
            raise TypeError('Unknown file format or invalid GEMPAK file')