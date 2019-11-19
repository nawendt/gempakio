"""Tools to process GEMPAK-formatted products."""

import contextlib
from datetime import datetime
from enum import Enum
from io import BytesIO
from itertools import repeat
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


class GempakDM(AbstractDataStore):
    """
    """

    gempak_header = 'GEMPAK DATA MANAGEMENT FILE '
    
    prod_desc_fmt = DictStruct([('version', 'i'), ('file_headers', 'i'),
                                ('file_keys_ptr', 'i'), ('rows', 'i'),
                                ('row_keys', 'i'), ('row_keys_ptr','i'),
                                ('row_headers_ptr', 'i'), ('columns', 'i'),
                                ('column_keys', 'i'), ('column_keys_ptr','i'),
                                ('column_headers_ptr', 'i'), ('parts', 'i'),
                                ('parts_ptr', 'i'), ('data_mgmt_ptr', 'i'),
                                ('data_mgmt_length', 'i'), ('data_block_ptr', 'i'),
                                ('file_type', 'i'), ('data_source', 'i'),
                                ('machine_type', 'i'), ('missing_int', 'i'),
                                ('blank', 'i'), ('blank', 'i'),
                                ('blank', 'i'),  ('missing_float' ,'f')])

    data_management_fmt = DictStruct([('next_free_word', 'i'), ('max_free_pairs', 'i'),
                                      ('actual_free_pairs', 'i'), ('last_word', 'i')])
    

    def __init__(self, filename):
        """
        """
        fobj = open_as_needed(filename)

        with contextlib.closing(fobj):
            self._buffer = IOBuffer.fromfile(fobj)

        start = self._buffer.set_mark()

        self._process_gempak_header()

        gpkhd = self._buffer.set_mark()

        self.prod_desc = self._buffer.read_struct(self.prod_desc_fmt)

        self._buffer.jump_to(start, (self.prod_desc['data_mgmt_ptr'] - 1) * 4)

        self.dm_header = self._buffer.read_struct(self.data_management_fmt)


        # row metadata
        # row key 256 (-1)
        # col key 257 (-1)
        # part key 267 (-1)
        

    def _process_gempak_header(self):
        """Read off the GEMPAK header from the file, if necessary."""
        data = self._buffer.get_next(len(self.gempak_header)).decode('utf-8', 'ignore')
        if data == self.gempak_header:
            self._buffer.skip(len(self.gempak_header))
        else:
            raise TypeError('Unknown file format or invalid GEMPAK file')

    def _get_dm_metadata(self):
        """
        """
        self.row_metadata = self._buffer.read_struct()

