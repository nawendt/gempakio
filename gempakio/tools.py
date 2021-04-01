# Copyright 2021 Nathan Wendt
"""Tools for reading GEMPAK files."""

from collections import namedtuple
import struct

import numpy as np


class NamedStruct(struct.Struct):
    """Parse bytes using `Struct` but provide named fields.

    Class from MetPy.
    """

    def __init__(self, info, prefmt='', tuple_name=None):
        """Initialize the NamedStruct."""
        if tuple_name is None:
            tuple_name = 'NamedStruct'
        names, fmts = zip(*info)
        self.converters = {}
        conv_off = 0
        for ind, i in enumerate(info):
            if len(i) > 2:
                self.converters[ind - conv_off] = i[-1]
            elif not i[0]:  # Skip items with no name
                conv_off += 1
        self._tuple = namedtuple(tuple_name, ' '.join(n for n in names if n))
        super().__init__(prefmt + ''.join(f for f in fmts if f))

    def _create(self, items):
        if self.converters:
            items = list(items)
            for ind, conv in self.converters.items():
                items[ind] = conv(items[ind])
            if len(items) < len(self._tuple._fields):
                items.extend([None] * (len(self._tuple._fields) - len(items)))
        return self.make_tuple(*items)

    def make_tuple(self, *args, **kwargs):
        """Construct the underlying tuple from values."""
        return self._tuple(*args, **kwargs)

    def unpack(self, s):
        """Parse bytes and return a namedtuple."""
        return self._create(super().unpack(s))

    def unpack_from(self, buff, offset=0):
        """Read bytes from a buffer and return as a namedtuple."""
        return self._create(super().unpack_from(buff, offset))

    def unpack_file(self, fobj):
        """Unpack the next bytes from a file object."""
        return self.unpack(fobj.read(self.size))

    def pack(self, **kwargs):
        """Pack the arguments into bytes using the structure."""
        t = self.make_tuple(**kwargs)
        return super().pack(*t)


class IOBuffer:
    """Holds bytes from a buffer to simplify parsing and random access.

    Class from MetPy.
    """

    def __init__(self, source):
        """Initialize the IOBuffer with the source data."""
        self._data = bytearray(source)
        self.reset()

    @classmethod
    def fromfile(cls, fobj):
        """Initialize the IOBuffer with the contents of the file object."""
        return cls(fobj.read())

    def reset(self):
        """Reset buffer back to initial state."""
        self._offset = 0
        self.clear_marks()

    def set_mark(self):
        """Mark the current location and return its id so that the buffer can return later."""
        self._bookmarks.append(self._offset)
        return len(self._bookmarks) - 1

    def jump_to(self, mark, offset=0):
        """Jump to a previously set mark."""
        self._offset = self._bookmarks[mark] + offset

    def offset_from(self, mark):
        """Calculate the current offset relative to a marked location."""
        return self._offset - self._bookmarks[mark]

    def clear_marks(self):
        """Clear all marked locations."""
        self._bookmarks = []

    def splice(self, mark, newdata):
        """Replace the data after the marked location with the specified data."""
        self.jump_to(mark)
        self._data = self._data[:self._offset] + bytearray(newdata)

    def read_struct(self, struct_class):
        """Parse and return a structure from the current buffer offset."""
        struct = struct_class.unpack_from(memoryview(self._data), self._offset)
        self.skip(struct_class.size)
        return struct

    def read_func(self, func, num_bytes=None):
        """Parse data from the current buffer offset using a function."""
        # only advance if func succeeds
        res = func(self.get_next(num_bytes))
        self.skip(num_bytes)
        return res

    def read_ascii(self, num_bytes=None):
        """Return the specified bytes as ascii-formatted text."""
        return self.read(num_bytes).decode('ascii')

    def read_binary(self, num, item_type='B'):
        """Parse the current buffer offset as the specified code."""
        if 'B' in item_type:
            return self.read(num)

        if item_type[0] in ('@', '=', '<', '>', '!'):
            order = item_type[0]
            item_type = item_type[1:]
        else:
            order = '@'

        return list(
            self.read_struct(struct.Struct(order + '{:d}'.format(int(num)) + item_type))
        )

    def read_int(self, size, endian, signed):
        """Parse the current buffer offset as the specified integer code."""
        return int.from_bytes(self.read(size), endian, signed=signed)

    def read_array(self, count, dtype):
        """Read an array of values from the buffer."""
        ret = np.frombuffer(self._data, offset=self._offset, dtype=dtype, count=count)
        self.skip(ret.nbytes)
        return ret

    def read(self, num_bytes=None):
        """Read and return the specified bytes from the buffer."""
        res = self.get_next(num_bytes)
        self.skip(len(res))
        return res

    def get_next(self, num_bytes=None):
        """Get the next bytes in the buffer without modifying the offset."""
        if num_bytes is None:
            return self._data[self._offset:]
        else:
            return self._data[self._offset:self._offset + num_bytes]

    def skip(self, num_bytes):
        """Jump the ahead the specified bytes in the buffer."""
        if num_bytes is None:
            self._offset = len(self._data)
        else:
            self._offset += num_bytes

    def check_remains(self, num_bytes):
        """Check that the number of bytes specified remains in the buffer."""
        return len(self._data[self._offset:]) == num_bytes

    def truncate(self, num_bytes):
        """Remove the specified number of bytes from the end of the buffer."""
        self._data = self._data[:-num_bytes]

    def at_end(self):
        """Return whether the buffer has reached the end of data."""
        return self._offset >= len(self._data)

    def __getitem__(self, item):
        """Return the data at the specified location."""
        return self._data[item]

    def __str__(self):
        """Return a string representation of the IOBuffer."""
        return 'Size: {} Offset: {}'.format(len(self._data), self._offset)

    def __len__(self):
        """Return the amount of data in the buffer."""
        return len(self._data)
