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

def _word_to_position(word, bytes_per_word=4):
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

        # Process file keys
        # Should this be so general or can we rely on there only ever being NAVB and ANLB?
        fkey_prod = product(['header_name', 'header_length', 'header_type'], range(1, self.prod_desc.file_headers + 1))
        fkey_names = [ x for x in [ '{}{}'.format(*x) for x in fkey_prod ] ]
        fkey_meta = list(zip(fkey_names, np.repeat(('4s', 'i', 'i'), self.prod_desc.file_headers)))
        self.file_keys_format = NamedStruct(fkey_meta)

        self._buffer.jump_to(start, _word_to_position(self.prod_desc.file_keys_ptr))
        self.file_keys = self._buffer.read_struct(self.file_keys_format)

        file_key_blocks = self._buffer.set_mark()
        # We are now at the NAVB/ANLB portion of the file.
        for block in range(self.prod_desc.file_headers):
            bsize = self._buffer.read_int('i')
            
    def _process_gempak_header(self):
        """Read off the GEMPAK header from the file, if necessary."""
        data = self._buffer.get_next(len(self.gempak_header)).decode('utf-8', 'ignore')
        if data == self.gempak_header:
            self._buffer.skip(len(self.gempak_header))
        else:
            raise TypeError('Unknown file format or invalid GEMPAK file')