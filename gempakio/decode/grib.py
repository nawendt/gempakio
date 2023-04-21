# Copyright (c) 2022 Nathan Wendt.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Classes for decoding GRIB data formats."""

import numpy as np


class Grib2:
    """GRIB2 class."""

    map_id = [2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1]

    def __init__(self, data):
        """Instantiate GRIB2 grid.

        Parameters
        ----------
        data : array_like
            One-dimensional array of packed GRIB2 data.
        """
        if not isinstance(data, (tuple, list, np.ndarray)):
            raise TypeError('Input data should be array-like.')

        if len(np.atleast_1d(data).shape) > 1:
            raise ValueError('Input data should be one-dimensional.')

        self._data = data
        self._offset = 0
        self._pos_sec_1 = -1
        self._pos_sec_2 = -1
        self._pos_sec_3 = -1
        self._pos_sec_4 = -1
        self._pos_sec_5 = -1
        self._pos_sec_6 = -1
        self._pos_sec_7 = -1
        self._len_sec_1 = 0
        self._len_sec_2 = 0
        self._len_sec_3 = 0
        self._len_sec_4 = 0
        self._len_sec_5 = 0
        self._len_sec_6 = 0
        self._len_sec_7 = 0
        self.section_1 = None
        self.section_2 = None
        self.section_3 = None
        self.section_4 = None
        self.section_5 = None
        self.section_6 = None
        self.section_7 = None

        self._get_section_attributes()
        if self._pos_sec_1 > 0:
            self._unpack_section_1()
        if self._pos_sec_2 > 0:
            self._unpack_section_2()
        if self._pos_sec_3 > 0:
            self._unpack_section_3()

    def _get_section_attributes(self):
        """Get section positions and lengths."""
        position = 0
        count = 4

        self._len_sec_1 = self._gbit(self._data, position * 8, count * 8)
        position += count
        self._pos_sec_1 = position

        position += self._len_sec_1
        self._len_sec_2 = self._gbit(self._data, position * 8, count * 8)
        position += count
        if self._len_sec_2 > 0:
            self._pos_sec_2 = position

        position += self._len_sec_2
        self._len_sec_3 = self._gbit(self._data, position * 8, count * 8)
        position += count
        if self._len_sec_3 > 0:
            self._pos_sec_3 = position

        position += self._len_sec_3
        self._len_sec_4 = self._gbit(self._data, position * 8, count * 8)
        position += count
        if self._len_sec_4 > 0:
            self._pos_sec_4 = position

        position += self._len_sec_4
        self._len_sec_5 = self._gbit(self._data, position * 8, count * 8)
        position += count
        if self._len_sec_5 > 0:
            self._pos_sec_5 = position

        position += self._len_sec_5
        self._len_sec_6 = self._gbit(self._data, position * 8, count * 8)
        position += count
        if self._len_sec_6 > 0:
            self._pos_sec_6 = position

        position += self._len_sec_6
        self._len_sec_7 = self._gbit(self._data, position * 8, count * 8)
        position += count
        if self._len_sec_7 > 0:
            self._pos_sec_7 = position

    def _unpack_section_1(self):
        """Unpack Section 1 of GRIB2 data."""
        self.section_1 = Section(1, self._len_sec_1)
        self._offset = 0

        section_length = self._gbit(self._data[self._pos_sec_1:], self._offset, 32)
        self._offset += 32
        section_number = self._gbit(self._data[self._pos_sec_1:], self._offset, 8)
        self._offset += 8

        if section_number != 1 or section_length != self._len_sec_1:
            raise ValueError('Not section 1 data.')

        section_1_names = [
            'originating_center',
            'subcenter',
            'grib_master_table_number',
            'grib_local_table_number',
            'significance_of_reference_time',
            'year',
            'month',
            'day',
            'hour',
            'minute',
            'second',
            'production_status',
            'type_of_data'
        ]
        for key, bits in zip(section_1_names, self.map_id):
            nbits = bits * 8
            self.section_1.add_attribute(**{
                key: self._gbit(self._data[self._pos_sec_1:], self._offset, nbits)
            })
            self._offset += nbits

    def _unpack_section_2(self):
        """Unpack Section 2 of GRIB2 data."""
        self.section_2 = Section(2, self._len_sec_2)
        self._offset = 0

        section_length = self._gbit(self._data[self._pos_sec_2:], self._offset, 32)
        lencsec2 = section_length - 5
        self._offset += 32
        section_number = self._gbit(self._data[self._pos_sec_2:], self._offset, 8)
        self._offset += 8
        ipos = self._offset // 8

        if section_number != 2 or section_length != self._len_sec_2:
            raise ValueError('Not section 2 data.')

        section_2_names = [
            f'local_{n + 1}' for n in range(lencsec2)
        ]

        if lencsec2 > 0:
            for key, i in zip(section_2_names, range(lencsec2)):
                self.section_2.add_attribute(**{
                    key: self._data[ipos + i]
                })
        self._offset += (lencsec2 * 8)

    def _unpack_section_3(self):
        """Unpack Section 3 of GRIB2 data."""
        self.section_3 = Section(3, self._len_sec_3)
        self._offset = 0

        section_length = self._gbit(self._data[self._pos_sec_3:], self._offset, 32)
        self._offset += 32
        section_number = self._gbit(self._data[self._pos_sec_3:], self._offset, 8)
        self._offset += 8

        if section_number != 3 or section_length != self._len_sec_3:
            raise ValueError('Not section 3 data.')

        grid_def_src = self._gbit(self._data[self._pos_sec_3:], self._offset, 8)
        self._offset += 8
        grid_pts = self._gbit(self._data[self._pos_sec_3:], self._offset, 32)
        self._offset += 32
        octets = self._gbit(self._data[self._pos_sec_3:], self._offset, 8)
        self._offset += 8
        interpreter = self._gbit(self._data[self._pos_sec_3:], self._offset, 8)
        self._offset += 8
        grid_temp_num = self._gbit(self._data[self._pos_sec_3:], self._offset, 16)
        self._offset += 16

        self.section_3.add_attribute(
            grid_definition_source=grid_def_src,
            number_points=grid_pts,
            option_list_octets=octets,
            option_list_interpreter=interpreter,
            grid_template_name=grid_temp_num
        )

    @staticmethod
    def _gbit(inbytes, skip, nbytes):
        """Get bits."""
        ones = [1, 3, 7, 15, 31, 63, 127, 255]
        nbit = skip

        bitcnt = nbytes
        index = nbit // 8
        ibit = nbit % 8
        nbit += nbytes

        tbit = bitcnt if bitcnt < (8 - ibit) else (8 - ibit)
        itmp = inbytes[index] & ones[7 - ibit]
        if tbit != (8 - ibit):
            itmp >>= (8 - ibit - tbit)
        index += 1
        bitcnt -= tbit

        while (bitcnt >= 8):
            itmp = (itmp << 8) | inbytes[index]
            bitcnt -= 8
            index += 1

        if (bitcnt > 0):
            itmp = (
                (itmp << bitcnt) | ((inbytes[index] >> (8 - bitcnt)) & ones[bitcnt - 1])
            )

        return itmp


class Section:
    """GRIB data section."""

    def __init__(self, number, length):
        self.number = number
        self.length = length

    def add_attribute(self, **kwargs):
        """Add metadata attribute to a Section."""
        for k, v in kwargs.items():
            self.__setattr__(k, v)

    def __repr__(self) -> str:
        """Return string representation of Section."""
        return (
            f"Section({','.join([f'{k}={v}' for k,v in self.__dict__.items()])})"
        )
