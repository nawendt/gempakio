"""Tools to process GEMPAK-formatted products."""

import contextlib
from datetime import datetime, timedelta
from enum import Enum
from io import BytesIO
from itertools import product
import re
import struct
import sys
import warnings

import numpy as np
import pyproj
import xarray as xr

from metpy.io._tools import IOBuffer, NamedStruct, open_as_needed

def formatwarning(message, category, filename, lineno, line):
    """
    """
    return '{}: {}\n'.format(category.__name__, message)
warnings.formatwarning = formatwarning

ANLB_SIZE = 128
BYTES_PER_WORD = 4
NAVB_SIZE = 256
PARAM_ATTR = [('name', (4, 's')), ('scale', (1,'i')), ('offset', (1,'i')), ('bits', (1, 'i'))]
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
}

GVCORD_TO_VAR = {
    'PRES': 'p',
    'HGHT': 'z',
    'THTA': 'theta',
}

class FileTypes(Enum):
    surface = 1
    sounding = 2
    grid = 3

class DataTypes(Enum):
    real = 1
    integer = 2
    character = 3
    realpack = 4
    grid = 5

class VerticalCoordinates(Enum):
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
    none = 0
    grib = 1
    nmc = 2
    diff = 3
    dec = 4
    grib2 = 5

class ForecastType(Enum):
    analysis = 0
    forecast = 1
    guess = 2
    initial = 3

class DataSource(Enum):
    airway_surface = 1
    metar = 2
    ship = 3
    buoy = 4
    synoptic = 5
    sounding = 4
    vas = 5
    grid = 6
    watch_by_county = 7
    unknown = 99
    text = 100

GEMPAK_HEADER = 'GEMPAK DATA MANAGEMENT FILE '
ENDIAN = sys.byteorder

def _word_to_position(word, bytes_per_word=BYTES_PER_WORD):
    r"""Return beginning position of a word in bytes"""
    return (word * bytes_per_word) - bytes_per_word

class GempakFile():
    r"""Base class for GEMPAK files. Reads ubiquitous GEMPAK file headers
    (i.e., the data managment portion of each file)."""

    prod_desc_fmt = [('version', 'i'), ('file_headers', 'i'),
                     ('file_keys_ptr', 'i'), ('rows', 'i'),
                     ('row_keys', 'i'), ('row_keys_ptr','i'),
                     ('row_headers_ptr', 'i'), ('columns', 'i'),
                     ('column_keys', 'i'), ('column_keys_ptr','i'),
                     ('column_headers_ptr', 'i'), ('parts', 'i'),
                     ('parts_ptr', 'i'), ('data_mgmt_ptr', 'i'),
                     ('data_mgmt_length', 'i'), ('data_block_ptr', 'i'),
                     ('file_type', 'i', FileTypes), ('data_source', 'i', DataSource),
                     ('machine_type', 'i'), ('missing_int', 'i'),
                     (None, '12x'), ('missing_float' ,'f')]

    grid_nav_fmt = [('grid_definition_type', 'f'), ('projection', '3sx', bytes.decode),
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

    data_management_fmt = ([('next_free_word', 'i'),('max_free_pairs', 'i'),
                            ('actual_free_pairs', 'i'), ('last_word', 'i')] + 
                        [ ('free_word{:d}'.format(n), 'i') for n in range(1, 29) ])
    
    def __init__(self, file):
        """Instantiate GempakFile object from file."""

        fobj = open_as_needed(file)

        with contextlib.closing(fobj):
            self._buffer = IOBuffer.fromfile(fobj)

        # Save file start position as pointers use this as reference
        start = self._buffer.set_mark()

        # Process the main GEMPAK header to verify file format
        self._process_gempak_header()
        meta = self._buffer.set_mark()

        # # Check for byte swapping
        self._swap_bytes(bytes(self._buffer.read_binary(4)))
        self._buffer.jump_to(meta)

        # Process main metadata header
        self.prod_desc = self._buffer.read_struct(NamedStruct(self.prod_desc_fmt, self.prefmt, 'ProductDescription'))

        # File Keys
        # Surface and upper-air files will not have the file headers, so we need to check for that.
        if self.prod_desc.file_headers > 0:
            # This would grab any file headers, but it seems NAVB and ANLB are the only ones used
            fkey_prod = product(['header_name', 'header_length', 'header_type'], range(1, self.prod_desc.file_headers + 1))
            fkey_names = [ x for x in [ '{}{}'.format(*x) for x in fkey_prod ] ]
            fkey_info = list(zip(fkey_names, np.repeat(('4s', 'i', 'i'), self.prod_desc.file_headers)))
            self.file_keys_format = NamedStruct(fkey_info, self.prefmt, 'FileKeys')

            self._buffer.jump_to(start, _word_to_position(self.prod_desc.file_keys_ptr))
            self.file_keys = self._buffer.read_struct(self.file_keys_format)

            file_key_blocks = self._buffer.set_mark()
            # Navigation Block
            navb_size = self._buffer.read_int('i')
            if navb_size != NAVB_SIZE:
                raise ValueError('Navigation block size does not match GEMPAK specification')
            else:
                self.navigation_block = self._buffer.read_struct(NamedStruct(self.grid_nav_fmt, self.prefmt, 'NavigationBlock'))
            self.kx = int(self.navigation_block.right_grid_number)
            self.ky = int(self.navigation_block.top_grid_number)
            
            # Analysis Block
            anlb_size = self._buffer.read_int(self.prefmt + 'i')
            anlb_start = self._buffer.set_mark()
            if anlb_size != ANLB_SIZE:
                raise ValueError('Analysis block size does not match GEMPAK specification')
            else:
                anlb_type = self._buffer.read_int(self.prefmt + 'f')
                self._buffer.jump_to(anlb_start)
                if anlb_type == 1:
                    self.analysis_block = self._buffer.read_struct(NamedStruct(self.grid_anl_fmt1, self.prefmt, 'AnalysisBlock'))
                elif anlb_type == 2:
                    self.analysis_block = self._buffer.read_struct(NamedStruct(self.grid_anl_fmt2, self.prefmt, 'AnalysisBlock'))
                else:
                    self.analysis_block = None
        else:
            self.analysis_block = None
            self.navigation_block = None

        # Data Management
        self._buffer.jump_to(start, _word_to_position(self.prod_desc.data_mgmt_ptr))
        self.data_management = self._buffer.read_struct(NamedStruct(self.data_management_fmt, self.prefmt, 'DataManagement'))

    def _swap_bytes(self, bytes):
        self.swaped_bytes = not (struct.pack('@i', 1) == bytes)

        if self.swaped_bytes:
            if sys.byteorder == 'little':
                self.prefmt = '>'
            elif sys.byteorder == 'big':
                self.prefmt = '<'
        else:
            self.prefmt = ''

    def _process_gempak_header(self):
        """Read the GEMPAK header from the file, if necessary."""

        fmt = [('text', '28s', bytes.decode), (None, None)]

        header = self._buffer.read_struct(NamedStruct(fmt, '', 'GempakHeader'))
        if header.text != GEMPAK_HEADER:
            raise TypeError('Unknown file format or invalid GEMPAK file')

    @staticmethod
    def _convert_dattim(dattim):
        if dattim:
            if dattim < 100000000:
                dt = datetime.strptime(str(dattim), '%y%m%d')
            else:
                dt = datetime.strptime('{:010d}'.format(dattim), '%m%d%y%H%M')
        else:
            dt = None
        return dt

    @staticmethod
    def _convert_ftime(ftime):
        if ftime:
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
        if level and level >= 0:
            return level
        else:
            return None

    @staticmethod
    def _convert_vertical_coord(coord):
        if coord:
            if coord <= 8:
                return VerticalCoordinates(coord).name.upper()
            else:
                return struct.pack('i',coord).decode()
        else:
            return None

    @staticmethod
    def _convert_parms(parm):
        dparm = parm.decode()
        return dparm.strip() if dparm.strip() else None

    @staticmethod
    def fortran_ishift(i, shift):
        mask = 0xffffffff
        if shift > 0:
            if i < 0:
                shifted = (i & mask) << shift
            else:    
                shifted = i << shift
        elif shift < 0:
            if i < 0:
                shifted = (i & mask) >> abs(shift)
            else:
                shifted = i >> abs(shift)
        elif shift == 0:
            shifted = i
        else:
            raise ValueError('Bad shift value {}.'.format(shift))
        return shifted

    @staticmethod
    def decode_strip(b):
        return b.decode().strip()

class GempakGrid(GempakFile):
    r"""Subclass of GempakFile specific to GEMPAK gridded data.

    """

    def __init__(self, file, *args, **kwargs):
        super().__init__(file)
        self.packing_type_from_user = kwargs.get('packing_type', None)

        DATETIME_NAMES = ['GDT1', 'GDT2']
        LEVEL_NAMES = ['GLV1', 'GLV2']
        FTIME_NAMES = ['GTM1', 'GTM2']
        STRING_NAMES = ['GPM1', 'GPM2', 'GPM3']

        start = self._buffer._bookmarks[0]

        # Row Keys
        self._buffer.jump_to(start, _word_to_position(self.prod_desc.row_keys_ptr))
        row_key_info = [ ('row_key{:d}'.format(n), '4s', self.decode_strip) for n in range(1, self.prod_desc.row_keys + 1) ]
        row_key_info.extend([(None, None)])
        row_keys_fmt = NamedStruct(row_key_info, self.prefmt, 'RowKeys')
        self.row_keys = self._buffer.read_struct(row_keys_fmt)

        # Column Keys
        self._buffer.jump_to(start, _word_to_position(self.prod_desc.column_keys_ptr))
        column_key_info = [ ('column_key{:d}'.format(n), '4s', self.decode_strip) for n in range(1, self.prod_desc.column_keys + 1) ]
        column_key_info.extend([(None, None)])
        column_keys_fmt = NamedStruct(column_key_info, self.prefmt, 'ColumnKeys')
        self.column_keys = self._buffer.read_struct(column_keys_fmt)

        # Row Headers
        # Based on GEMPAK source, row/col headers have a 0th element in their Fortran arrays.
        # This appears to be a flag value to say a header is used or not. 9999
        # means its in use, otherwise -9999. GEMPAK allows empty grids, etc., but
        # not a real need to keep track of that in Python.
        self._buffer.jump_to(start, _word_to_position(self.prod_desc.row_headers_ptr))
        self.row_headers = []
        row_headers_info = [ (key, 'i') for key in self.row_keys ]
        row_headers_info.extend([(None, None)])
        row_headers_fmt = NamedStruct(row_headers_info, self.prefmt, 'RowHeaders')
        for n in range(1, self.prod_desc.rows + 1):
            if self._buffer.read_int(self.prefmt + 'i') == USED_FLAG:
                self.row_headers.append(self._buffer.read_struct(row_headers_fmt))

        # Column Headers
        self._buffer.jump_to(start, _word_to_position(self.prod_desc.column_headers_ptr))
        self.column_headers = []
        column_headers_info = [ (key, 'i', self._convert_level) if key in LEVEL_NAMES
                                else (key, 'i', self._convert_vertical_coord) if key == 'GVCD'
                                else (key, 'i', self._convert_dattim) if key in DATETIME_NAMES
                                else (key, 'i', self._convert_ftime) if key in FTIME_NAMES
                                else (key, '4s', self._convert_parms) if key in STRING_NAMES
                                else (key, 'i')
                                for key in self.column_keys ]
        column_headers_info.extend([(None, None)])
        column_headers_fmt = NamedStruct(column_headers_info, self.prefmt, 'ColumnHeaders')
        for n in range(1, self.prod_desc.columns + 1):
            if self._buffer.read_int(self.prefmt + 'i') == USED_FLAG:
                self.column_headers.append(self._buffer.read_struct(column_headers_fmt))

        # Parts
        self._buffer.jump_to(start, _word_to_position(self.prod_desc.parts_ptr))
        parts = self._buffer.set_mark()
        self.parts = []
        parts_info = [ ('name', '4s', self.decode_strip),
                        (None, '{:d}x'.format((self.prod_desc.parts - 1) * BYTES_PER_WORD)),
                        ('header_length', 'i'),
                        (None, '{:d}x'.format((self.prod_desc.parts - 1) * BYTES_PER_WORD)),
                        ('data_type', 'i', DataTypes),
                        (None, '{:d}x'.format((self.prod_desc.parts - 1) * BYTES_PER_WORD)),
                        ('parameter_count', 'i') ]
        parts_info.extend([(None, None)])
        parts_fmt = NamedStruct(parts_info, self.prefmt, 'Parts')
        for n in range(1, self.prod_desc.parts + 1):
            self.parts.append(self._buffer.read_struct(parts_fmt))
            self._buffer.jump_to(start, _word_to_position(self.prod_desc.parts_ptr + n))

        # Parameters
        # No need to jump to any position as this follows parts information
        self._buffer.jump_to(start, _word_to_position(self.prod_desc.parts_ptr + self.prod_desc.parts * 4))
        self.parameters = [ { key: [] for key, _ in PARAM_ATTR } for n in range(self.prod_desc.parts) ]
        for attr, fmt in PARAM_ATTR:
            fmt = (fmt[0], self.prefmt + fmt[1])
            for n, part in enumerate(self.parts):
                for _ in range(part.parameter_count):
                    if fmt[1] == 's':
                        self.parameters[n][attr] += [self._buffer.read_binary(*fmt)[0].decode()]
                    else:
                        self.parameters[n][attr] += self._buffer.read_binary(*fmt)

        # Coordinates
        if self.navigation_block is not None:
            self._get_crs()
            self._get_coordinates()

    def _get_crs(self):
        gemproj = self.navigation_block.projection
        proj, ptype = GEMPROJ_TO_PROJ[gemproj]

        if ptype == 'azm':
            lat_0 = self.navigation_block.proj_angle1
            lon_0 = self.navigation_block.proj_angle2
            lat_ts = self.navigation_block.proj_angle3
            self.crs = pyproj.CRS.from_dict({'proj': proj, 'lat_0': lat_0, 'lon_0': lon_0, 'lat_ts': lat_ts})
        elif ptype == 'cyl':
            if gemproj != 'mcd':
                lat_0 = self.navigation_block.proj_angle1
                lon_0 = self.navigation_block.proj_angle2
                lat_ts = self.navigation_block.proj_angle3
                self.crs = pyproj.CRS.from_dict({'proj': proj, 'lat_0': lat_0, 'lon_0': lon_0, 'lat_ts': lat_ts})
            else:
                avglat = (self.navigation_block.upper_right_lat + 
                          self.navigation_block.lower_left_lat) * 0.5
                k_0 = 1/cos(avglat) if self.navigation_block.proj_angle1 == 0 else self.navigation_block.proj_angle1
                lon_0 = self.navigation_block.proj_angle2
                self.crs = pyproj.CRS.from_dict({'proj': proj, 'lat_0': lat_0, 'k_0': k_0})
        elif ptype == 'con':
            lat_1 = self.navigation_block.proj_angle1
            lon_0 = self.navigation_block.proj_angle2
            lat_2 = self.navigation_block.proj_angle3
            self.crs = pyproj.CRS.from_dict({'proj': proj, 'lon_0': lon_0, 'lat_1': lat_1, 'lat_2': lat_2})

    def _get_coordinates(self):
        transform = pyproj.Proj(self.crs)
        llx,lly = transform(self.navigation_block.lower_left_lon,self.navigation_block.lower_left_lat)
        urx,ury = transform(self.navigation_block.upper_right_lon,self.navigation_block.upper_right_lat)
        self.x = np.linspace(llx, urx, self.kx)
        self.y = np.linspace(lly, ury, self.ky)
        xx, yy = np.meshgrid(self.x, self.y)
        self.lon, self.lat = transform(xx, yy, inverse=True)

    def _unpack_grid(self, packing_type, part):
        print(packing_type)
        if packing_type == PackingType.none:
            # raise NotImplementedError('Upacked data not supported.')
            lendat = self.data_header_length - part.header_length - 1
            buffer_fmt = '{}{}i'.format(self.prefmt, lendat)
            buffer = self._buffer.read_struct(struct.Struct(buffer_fmt))
            grid = np.zeros((self.ky, self.kx))
            print(lendat)
        elif packing_type == PackingType.nmc:
            raise NotImplementedError('NMC unpacking not supported.')
            integer_meta_fmt = [('bits', 'i'), ('missing_flag', 'i'), ('kxky', 'i')]
            real_meta_fmt = [('reference', 'f'), ('scale', 'f')]
            self.grid_meta_int = self._buffer.read_struct(NamedStruct(integer_meta_fmt, self.prefmt, 'GridMetaInt'))
            self.grid_meta_real = self._buffer.read_struct(NamedStruct(real_meta_fmt, self.prefmt, 'GridMetaReal'))
            grid_start = self._buffer.set_mark()
        elif packing_type == PackingType.diff:
            # raise NotImplementedError('GEMPAK DIF unpacking not supported.')
            integer_meta_fmt = [('bits', 'i'), ('missing_flag', 'i'), ('kxky', 'i'), ('kx', 'i')]
            real_meta_fmt = [('reference', 'f'), ('scale', 'f'), ('diffmin', 'f')]
            self.grid_meta_int = self._buffer.read_struct(NamedStruct(integer_meta_fmt, self.prefmt, 'GridMetaInt'))
            self.grid_meta_real = self._buffer.read_struct(NamedStruct(real_meta_fmt, self.prefmt, 'GridMetaReal'))
            grid_start = self._buffer.set_mark()

            imiss = 2**self.grid_meta_int.bits - 1
            lendat = self.data_header_length - part.header_length - 8
            packed_buffer_fmt = '{}{}i'.format(self.prefmt, lendat)
            packed_buffer = self._buffer.read_struct(struct.Struct(packed_buffer_fmt))
            grid = np.zeros((self.ky, self.kx))

            if lendat > 1:
                iword = 0
                ibit = 1
                first = True
                for j in range(self.ky):
                    line = False
                    for i in range(self.kx):
                        print('INFO:: -----iword----- {}'.format(iword+1))
                        jshft = self.grid_meta_int.bits + ibit - 33
                        print('INFO:: idata: {}'.format(packed_buffer[iword]))
                        print('INFO:: jshft: {}'.format(jshft))
                        idat = self.fortran_ishift(packed_buffer[iword], jshft)
                        print('INFO:: idat_shift: {}'.format(idat))
                        idat &= imiss
                        print('INFO:: idat_and: {}'.format(idat))

                        if jshft > 0:
                            jshft -= 32
                            print('INFO:: jshft_gt0: {}'.format(jshft))
                            idat2 = self.fortran_ishift(packed_buffer[iword+1], jshft)
                            print('INFO:: idat2_shift: {}'.format(idat2))
                            idat |= idat2
                            print('INFO:: idat_or: {}'.format(idat))

                        ibit += self.grid_meta_int.bits
                        print('INFO:: ibit: {}'.format(ibit))
                        if ibit > 32:
                            ibit -= 32
                            print('INFO:: ibit_gt32: {}'.format(ibit))
                            iword += 1

                        if (self.grid_meta_int.missing_flag and idat == imiss):
                            grid[j,i] = self.prod_desc.missing_float
                        else:
                            if first:
                                grid[j,i] = self.grid_meta_real.reference
                                print('INFO:: grid_first: {}'.format(grid[j,i]))
                                psav = self.grid_meta_real.reference
                                plin = self.grid_meta_real.reference
                                line = True
                                first = False
                            else:
                                if not line:
                                    grid[j,i] = plin + (self.grid_meta_real.diffmin 
                                                        + idat * self.grid_meta_real.scale)
                                    print('INFO:: grid_line: {}'.format(grid[j,i]))
                                    line = True
                                    plin = grid[j,i]
                                else:
                                    grid[j,i] = psav + (self.grid_meta_real.diffmin
                                                        + idat * self.grid_meta_real.scale)
                                    print('INFO:: grid: {}'.format(grid[j,i]))
                                psav = grid[j,i]
            else:
                grid = None
            return grid

        elif packing_type == PackingType.grib or packing_type == PackingType.dec:
            integer_meta_fmt = [('bits', 'i'), ('missing_flag', 'i'), ('kxky', 'i')]
            real_meta_fmt = [('reference', 'f'), ('scale', 'f')]
            self.grid_meta_int = self._buffer.read_struct(NamedStruct(integer_meta_fmt, self.prefmt, 'GridMetaInt'))
            self.grid_meta_real = self._buffer.read_struct(NamedStruct(real_meta_fmt, self.prefmt, 'GridMetaReal'))
            grid_start = self._buffer.set_mark()
            # print(self._buffer._offset)
            # print(self.grid_meta_real.reference, self.grid_meta_real.scale)
            # print(self.grid_meta_int.missing_flag)

            lendat = self.data_header_length - part.header_length - 6
            # print('lendat: %d' % lendat)
            packed_buffer_fmt = '{}{}i'.format(self.prefmt, lendat)

            grid = np.zeros(self.grid_meta_int.kxky)
            packed_buffer = self._buffer.read_struct(struct.Struct(packed_buffer_fmt))
            # print('\t{}'.format(len(packed_buffer)))
            if lendat > 1:
                imax = 2**self.grid_meta_int.bits - 1
                ibit = 1
                iword = 0
                for cell in range(self.grid_meta_int.kxky):
                    # print('INFO:: -----iword-----: {}'.format(cell))
                    # print('INFO:: idata: {}'.format(packed_buffer[iword]))
                    jshft = self.grid_meta_int.bits + ibit - 33
                    # print('INFO:: jshft1: {}'.format(jshft))
                    idat = self.fortran_ishift(packed_buffer[iword], jshft)
                    # print('INFO:: idat_shift: {}'.format(idat))
                    idat &= imax
                    # print('INFO:: idat_and: {}'.format(idat))

                    if jshft > 0:
                        jshft -= 32
                        # print('INFO:: jshft_gt0: {}'.format(jshft))
                        idat2 = self.fortran_ishift(packed_buffer[iword+1], jshft)
                        # print('INFO:: idat_next: {}'.format(packed_buffer[iword+1]))
                        # print('INFO:: idat2_shift: {}'.format(idat2))
                        idat |= idat2
                        # print('INFO:: idat_or: {}'.format(idat))

                    if (idat == imax) and self.grid_meta_int.missing_flag:
                        grid[cell] = self.prod_desc.missing_float
                    else:
                        grid[cell] = self.grid_meta_real.reference + (idat * self.grid_meta_real.scale)
                    # print(idat, grid[cell])
                    
                    ibit += self.grid_meta_int.bits
                    # print('INFO:: ibit: {}'.format(ibit))
                    if ibit > 32:
                        ibit -= 32
                        # print('INFO:: ibit_gt32: {}'.format(ibit))
                        iword += 1
            else:
                grid = None
            return grid
        elif packing_type == PackingType.grib2:
            raise NotImplementedError('GRIB2 unpacking not supported.')
            integer_meta_fmt = [('iuscal', 'i'), ('kx', 'i'), ('ky', 'i'), ('iscan_mode', 'i')]
            real_meta_fmt = [('rmsval', 'f')]
            self.grid_meta_int = self._buffer.read_struct(NamedStruct(integer_meta_fmt, self.prefmt, 'GridMetaInt'))
            self.grid_meta_real = self._buffer.read_struct(NamedStruct(real_meta_fmt, self.prefmt, 'GridMetaReal'))
            grid_start = self._buffer.set_mark()
        else:
            raise NotImplementedError('No method for unknown grid packing {}'.format(packing_type.name))

    def to_xarray(self):
        grids = []
        for n, (part, header) in enumerate(product(self.parts, self.column_headers)):
            self._buffer.jump_to(0, _word_to_position(self.prod_desc.data_block_ptr + n))
            self.data_ptr = self._buffer.read_int('i')
            self._buffer.jump_to(0, _word_to_position(self.data_ptr))
            self.data_header_length = self._buffer.read_int('i')
            data_header = self._buffer.set_mark()
            self._buffer.jump_to(data_header, _word_to_position(part.header_length + 1))
            packing_type_from_file = PackingType(self._buffer.read_int('i'))
            packing_type = ( packing_type_from_file if self.packing_type_from_user is None 
                             else PackingType[self.packing_type_from_user] )
            # print(n, self.data_ptr, header.GPM1, header.GDT1 + header.GTM1[1], packing_type)
            ftype, ftime = header.GTM1
            init = header.GDT1
            valid = init + ftime
            gvcord = header.GVCD.lower() if header.GVCD is not None else 'none'
            var = GVCORD_TO_VAR[header.GPM1] if header.GPM1 in GVCORD_TO_VAR else header.GPM1.lower()
            # accum = re.search('^[PSCWIZAHGNRL](?P<accum>\d{2,3})[IM]$', header.GPM1, flags=re.I)
            data = self._unpack_grid(packing_type, part)
            if data is not None:
                data = np.ma.array(data.reshape((self.ky, self.kx)), mask=data == self.prod_desc.missing_float)
                # print(n, data.max(), data.min())
                grids.append(data)
            #     xrda = xr.DataArray(data = data[np.newaxis, np.newaxis, ...],
            #                         coords = {'time': [valid], 
            #                                   gvcord: [header.GLV1],
            #                         },
            #                         dims = ['time', gvcord, 'y', 'x'],
            #                         name = var,
            #                         )
            #     grids.append(xrda)

            else:
                warnings.warn('Bad grid for {}'.format(header.GPM1))
            # dataset = xr.merge(grids, compat='override')
            # dataset = dataset.assign_coords({
            #     'x': (('x',), self.x),
            #     'y': (('y',), self.y),
            #     'lat': (('y', 'x'), self.lat),
            #     'lon': (('y', 'x'), self.lon),
            #     })
            # dataset = dataset.assign_attrs(**{
            #     'initialized': init,
            #     'type': ftype,
            #     'crs': self.crs.srs,
            #     'missing_value': self.prod_desc.missing_float,
            #     })
        # return dataset
        return grids

class GempakSounding(GempakFile):
    pass

class GempakSurface(GempakFile):
    pass