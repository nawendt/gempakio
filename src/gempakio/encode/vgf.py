# Copyright (c) 2024 Nathan Wendt.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Classes for encoding GEMPAK VGF files."""


from dataclasses import dataclass
from enum import IntEnum
import struct

import numpy as np

from gempakio.common import MAX_POINTS
from gempakio.tools import NamedStruct


class FillCodes(IntEnum):
    """Fill type codes."""

    NONE = 0
    SOLID = 2
    DASHED = 3
    DIAGONAL_LOW = 4
    DIAGONAL_MED = 5
    STAIRSTEP = 6
    SQUARE = 7
    DIAGONAL_HIGH = 8


class FontCodes(IntEnum):
    """Font codes."""

    COURIER = 1
    COURIER_ITALIC = 11
    COURIER_BOLD = 21
    COURIER_ITALIC_BOLD = 31
    HELVETICA = 2
    HELVETICA_ITALIC = 12
    HELVETICA_BOLD = 22
    HELVETICA_ITALIC_BOLD = 32
    TIMES = 3
    TIMES_ITALIC = 13
    TIMES_BOLD = 23
    TIMES_ITALIC_BOLD = 33


class FrontCodes(IntEnum):
    """Useful front codes."""

    STATIONARY_FRONT = 20
    STATIONARY_FRONT_DEVELOPING = 25
    STATIONARY_FRONT_DISSIPATING = 28
    WARM_FRONT = 220
    WARM_FRONT_DEVELOPING = 225
    WARM_FRONT_DISSIPATING = 228
    COLD_FRONT = 420
    COLD_FRONT_DEVELOPING = 425
    COLD_FRONT_DISSIPATING = 428
    OCCLUDED_FRONT = 620
    OCCLUDED_FRONT_DEVELOPING = 625
    OCCLUDED_FRONT_DISSIPATING = 628
    DRYLINE = 720
    TROUGH_AXIS_DASH = 820
    TROUGH_AXIS_SOLID = 829
    SQUALL = 940


class LineCodes(IntEnum):
    """Line type codes."""

    DOTTED = 0
    SOLID = 1
    SHORT_DASHED = 2
    MEDIUM_DASHED = 3
    LONG_DASH_SHORT_DASH = 4
    LONG_DASH = 5
    LONG_DASH_THREE_SHORT_DASHES = 6
    LONG_DASH_DOT = 7
    LONG_DASH_THREE_DOTS = 8
    MEDIUM_DASH_DOT = 9


class SpecialLineCodes(IntEnum):
    """Special line codes."""

    BALL_CHAIN = 1
    ZIGZAG = 2
    SCALLOP = 3
    POINTED_ARROW = 4
    ALT_ANGLE_TICKS = 5
    FILLED_ARROW = 6
    BOX_CIRCLES = 7
    TWO_X = 8
    FILLED_CIRCLES = 9
    LINE_FILL_CIRCLE_LINE = 10
    TICK_MARKS = 11
    LINE_X_LINE = 12
    FANCY_ARROW = 13
    FILL_CIRCLE_X = 14
    BOX_X = 15
    LINE_CIRCLE_LINE = 16
    LINE_CARET_LINE1 = 17
    LINE_CARET_LINE2 = 18
    SINE_CURVE = 19
    ARROW_DASHED = 20
    FILL_ARROW_DASH = 21
    STREAMLINE = 22
    DOUBLE_LINE = 23
    KINK_LINE1 = 24
    KINK_LINE2 = 25
    Z_LINE = 26


class SpecialTextCodes(IntEnum):
    """Special text codes."""

    GENERAL_TEXT = 0
    LOW_PRESSURE_BOX = 1
    HIGH_PRESSURE_BOX = 2
    BOX_BORDER_FILLED = 3
    BOX_BORDER_NOFILL = 4
    BOX_NOBORDER_NOFILL = 5
    FREEZING_LEVEL = 6
    LOW_LEVEL_TURBULENCE = 7
    CLOUD_LEVEL = 8
    HIGH_LEVEL_TURBULENCE = 9
    UNDERLINED_TEXT_NOFILL = 10
    UNDERLINED_TEXT_FILLED = 11
    MID_LEVEL_ICING = 12
    OVERLINED_TEXT_NOFILL = 13
    OVERLINED_TEXT_FILLED = 14
    MID_LEVEL_CLOUD = 15
    FLIGHT_LEVEL = 16


@dataclass
class Element:
    """Base VGF element."""

    element_header_struct = NamedStruct(
        [('delete', 'B'), ('vg_type', 'B'), ('vg_class', 'B'), ('filled', 'b'),
         ('closed', 'B'), ('smooth', 'B'), ('version', 'B'), ('group_type', 'B'),
         ('group_number', 'i'), ('major_color', 'i'), ('minor_color', 'i'),
         ('record_size', 'i'), ('min_lat', 'f'), ('min_lon', 'f'), ('max_lat', 'f'),
         ('max_lon', 'f')], '>', 'ElementHeader'
    )

    def __init__(self):
        self._delete = 0
        self._vg_type = 0
        self._vg_class = 0
        self._filled = 0
        self._closed = 0
        self._smooth = 0
        self._version = 0
        self._group_type = 0
        self._group_number = 0
        self._major_color = 0
        self._minor_color = 0
        self._record_size = 0
        self._min_lat = 0.0
        self._min_lon = 0.0
        self._max_lat = 0.0
        self._max_lon = 0.0

    @property
    def delete(self):
        """Get delete."""
        return self._delete

    @property
    def vg_type(self):
        """Get VG type."""
        return self._vg_type

    @property
    def vg_class(self):
        """Get VG class."""
        return self._vg_class

    @property
    def closed(self):
        """Get closed."""
        return self._closed

    @closed.setter
    def closed(self, value):
        """Set closed."""
        if value not in [0, 1]:
            raise ValueError('value must be 0 (open) or 1 (closed).')
        self._closed = value

    @property
    def min_lon(self):
        """Get minimum longitude."""
        return self._min_lon

    @property
    def max_lon(self):
        """Get maximum longitude."""
        return self._max_lon

    @property
    def min_lat(self):
        """Get minimum latitude."""
        return self._min_lat

    @property
    def max_lat(self):
        """Get maximum latitude."""
        return self._max_lat

    @property
    def filled(self):
        """Get fill."""
        return self._filled

    @filled.setter
    def filled(self, value):
        """Set fill."""
        if value not in range(9):
            raise ValueError('fill values must be in range [0, 8].')
        self._filled = value

    @property
    def smooth(self):
        """Get smooth."""
        return self._smooth

    @smooth.setter
    def smooth(self, value):
        """Set smooth."""
        if value not in range(3):
            raise ValueError(
                'smooth option must be 0 (none), 1 (splines), or 2 (parametric).'
            )
        self._smooth = value

    @property
    def group_type(self):
        """Get group type."""
        return self._group_type

    @group_type.setter
    def group_type(self, value):
        """Set group type."""
        if value < 0:
            raise ValueError('group type should be non-negative.')
        self._group_type = value

    @property
    def group_number(self):
        """Get group number."""
        return self._group_number

    @group_number.setter
    def group_number(self, value):
        """Set group number."""
        if value < 0:
            raise ValueError('group number should be non-negative.')
        self._group_number = value

    @property
    def major_color(self):
        """Get major color."""
        return self._major_color

    @major_color.setter
    def major_color(self, value):
        """Set major color."""
        if value not in range(33):
            raise ValueError('GEMPAK colors must be in range of 0 to 32.')
        self._major_color = value

    @property
    def minor_color(self):
        """Get minor color."""
        return self._minor_color

    @minor_color.setter
    def minor_color(self, value):
        """Set minor color."""
        if value not in range(33):
            raise ValueError('GEMPAK colors must be in range of 0 to 32.')
        self._minor_color = value

    def _make_element_header(self):
        packed = self.element_header_struct.pack(
            delete=self._delete,
            vg_type=self._vg_type,
            vg_class=self._vg_class,
            filled=self._filled,
            closed=self._closed,
            smooth=self._smooth,
            version=self._version,
            group_type=self._group_type,
            group_number=self._group_number,
            major_color=self._major_color,
            minor_color=self._minor_color,
            record_size=self._record_size,
            min_lat=self._min_lat,
            min_lon=self._min_lon,
            max_lat=self._max_lat,
            max_lon=self._max_lon
        )

        return packed

    def encode(self):
        """Encode element to VGF binary.

        This must be overridden by subclasses.
        """
        raise NotImplementedError('encode method must be defined in subclass.')


class FileHeader(Element):
    """File header element."""

    file_header_struct = NamedStruct(
        [('version', '128s'), ('notes', '256s')], tuple_name='FileHeader'
    )

    def __init__(self):
        super().__init__()
        self._gempak_version = b'Version 7.19.0'
        self._notes = b'NAWIPS Vector Graphic Format '
        self._vg_type = 22
        self._record_size = self.element_header_struct.size + self.file_header_struct.size

    def encode(self):
        """Encode file header."""
        header = self._make_element_header()
        packed = self.file_header_struct.pack(
            version=self._gempak_version,
            notes=self._notes
        )

        return header + packed


class LineBase(Element):
    """Base line class."""

    def __init__(self, lon, lat, color, width, closed, filled, smooth):
        super().__init__()
        if len(lon) != len(lat):
            raise ValueError('Coordinates arrays must be same size.')

        if len(lon) > MAX_POINTS:
            raise ValueError(f'number of points in line must not exceed {MAX_POINTS}')

        self._lon = np.asarray(lon).astype('float32')
        self._lat = np.asarray(lat).astype('float32')
        self._number_points = len(lon)
        self._min_lon = min(lon)
        self._min_lat = min(lat)
        self._max_lon = max(lon)
        self._max_lat = max(lat)
        self.major_color = color
        self.minor_color = color
        self.width = width
        self.closed = closed
        self.filled = filled
        self.smooth = smooth

    @property
    def width(self):
        """Get line width."""
        return self._width

    @width.setter
    def width(self, value):
        """Set line width."""
        if value < 1 or value > 10:
            raise ValueError('Line width must be in range [1, 10].')
        self._width = value

    def flip(self):
        """Flip line direction."""
        self._direction *= -1


class TextBase(Element):
    """Base text class."""

    def __init__(self, lon, lat, text, text_color, size, font, width, align, rotation,
                 offset_x, offset_y, text_flag):
        super().__init__()
        self._lat = lat
        self._lon = lon
        self._min_lon = lon
        self._min_lat = lat
        self._max_lon = lon
        self._max_lat = lat
        self.size = size
        self.text = text + '\x00'  # add null character
        self.text_color = text_color
        self.major_color = text_color
        self.minor_color = text_color
        self.font = font
        self._width = width
        self.align = align
        self._rotation = rotation
        self._offset_x = offset_x
        self._offset_y = offset_y
        self._text_flag = text_flag

    @staticmethod
    def points_to_size(points):
        """Convert font points to GEMPAK size."""
        return points / 14

    @property
    def text_color(self):
        """Get text color."""
        return self._text_color

    @text_color.setter
    def text_color(self, value):
        """Set text color."""
        if value not in range(33):
            raise ValueError('Invalid text color.')
        self._text_color = value

    @property
    def size(self):
        """Get text size."""
        return self._size

    @size.setter
    def size(self, value):
        """Set text size."""
        if value not in [10, 12, 14, 18, 24, 34]:
            raise ValueError('Invalid hardware font size.')
        self._size = value

    @property
    def text(self):
        """Get text."""
        return self._text

    @text.setter
    def text(self, value):
        """Set text."""
        if not isinstance(value, str):
            raise TypeError('text must be a string.')
        elif len(value) > 255:
            raise ValueError('text string exceeds 255 character limit.')
        self._text = value

    @property
    def font(self):
        """Get font code."""
        return self._font

    @font.setter
    def font(self, value):
        """Set font code."""
        if (self._text_flag == 2
           and value not in [1, 2, 3, 11, 12, 13, 21, 22, 23, 31, 32, 33]):
            raise ValueError('Invalid hardware font code.')
        self._font = value

    @property
    def width(self):
        """Get font width."""
        return self._width

    @width.setter
    def width(self, value):
        """Set text width."""
        self._width = value

    @property
    def align(self):
        """Get text alignment."""
        return self._align

    @align.setter
    def align(self, value):
        """Set text alignment."""
        if value not in [-1, 0, 1]:
            raise ValueError('Invalid text alignment.')
        self._align = value

    @property
    def rotation(self):
        """Get text rotation."""
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        """Set text rotation."""
        self._rotation = value

    @property
    def offset_x(self):
        """Get text offset in x-direction."""
        return self._offset_x

    @offset_x.setter
    def offset_x(self, value):
        """Set text offset in x-direction."""
        self._offset_x = value

    @property
    def offset_y(self):
        """Get text offset in y-direction."""
        return self._offset_y

    @offset_y.setter
    def offset_y(self, value):
        """Set text offset in y-direction."""
        self._offset_y = value

    @property
    def text_flag(self):
        """Get text flag."""
        return self._text_flag

    @text_flag.setter
    def text_flag(self, value):
        """Set text flag."""
        if value not in [1, 2]:
            raise ValueError('Invalid text flag.')


class Line(LineBase):
    """Line element."""

    line_struct = NamedStruct(
        [('number_points', 'i'), ('line_type', 'i'), ('line_type_hardware', 'i'),
         ('width', 'i'), ('line_width_hardware', 'i')], '>', 'LineInfo'
    )

    def __init__(self, lon, lat, color, line_type, closed, filled=0, smooth=0, width=2):
        """Create line element.

        Parameters
        ----------
        lon : `numpy.ndarray`

        lat : `numpy.ndarray`

        color : int
            GEMPAK color code. Must be in the range [0, 32].

        line_type : int
            GEMPAK provides a variety of line types (dashing patterns) to be used when multiple
            plots are overlain. The various line types are specified by integers. There are ten
            basic line types as follows:

                0 - dotted
                1 - solid
                2 - short dashed
                3 - medium dashed
                4 - long dash short dash
                5 - long dash
                6 - long dash three short dashes
                7 - long dash dot
                8 - long dash three dots
                9 - medium dash dot

            These patterns can be expanded or compressed by prefixing the single digit with a
            number from 1 to 9. A prefix of 1 compresses the basic pattern, a prefix of 2 is
            the default, and prefixes 3 - 9 expand the pattern. A single-digit line type n is
            identical to the two-digit number n+20. Line type numbers 10 - 19 cause a
            compressed line pattern while numbers 30 and higher cause expanded line patterns.
            For example, 32 expands line type 2 while 12 compresses the same pattern.

            Additional details and examples can be found in Appendix C of the GEMPAK Users
            Manual.

        closed : int
            Toggle closed polygon. Values are 0 (open) or 1 (closed).

        filled : int
            Fill pattern code. Default is no fill (0). Codes are defined as follows:

                Code        Fill Pattern
                0           No fill
                1           No fill
                2           Solid fill
                3           Dashed fill
                4           Diagonal line (low density)
                5           Diagonal line (medium density)
                6           Stairstep
                7           Square
                8           Diagonal line (high density)

        smooth : int
            Line smoothing parameter. No smoothing (0; default), splines (1),
            or parametric (2).

        width : int
            Width of special line. Must be in range [1, 10]. Defaults to 2.
        """
        super().__init__(lon, lat, color, width, closed, filled, smooth)
        self._line_type_hardware = 0
        self._line_width_hardware = 0
        self._vg_class = 3
        self._vg_type = 1
        self.line_type = line_type
        self._pts_struct = struct.Struct(f'>{self._number_points * 2}f')
        self._record_size = (self.element_header_struct.size + self.line_struct.size
                             + self._pts_struct.size)

    @property
    def line_type(self):
        """Get special line type."""
        return self._line_type

    @line_type.setter
    def line_type(self, value):
        """Set special line type."""
        if value < 0 or value > 99:
            raise ValueError('Invalid line type.')
        self._line_type = value

    def encode(self):
        """Encode line."""
        header = self._make_element_header()

        info = self.line_struct.pack(
            number_points=self._number_points,
            line_type=self._line_type,
            line_type_hardware=self._line_type_hardware,
            width=self._width,
            line_width_hardware=self._line_width_hardware
        )

        pts = self._pts_struct.pack(
            *self._lat, *self._lon
        )

        return header + info + pts


class Front(Element):
    """Front element."""

    front_struct = NamedStruct(
        [('number_points', 'i'), ('front_code', 'i'), ('pip_size', 'i'), ('pip_stroke', 'i'),
         ('pip_direction', 'i'), ('width', 'i'), ('label', '4s')], '>', 'FrontInfo'
    )

    def __init__(self, lon, lat, front_code, major_color, minor_color=None, smooth=2,
                 pip_size=100, pip_stroke=1, pip_direction=1):
        """Create front.

        Parameters
        ----------
        lon : `numpy.ndarray`

        lat : `numpy.ndarray`

        front_code : int
            A three digit number defining the front type, intensity, and character. Leading
            zeros are omitted, however. The three digits are defined as follows:

                Digit 1: Type
                0 = stationary
                1 = stationary above surface
                2 = warm
                3 = warm above surface
                4 = cold
                5 = cold above surface
                6 = occlusion
                7 = instability line
                8 = intertropical line
                9 = convergence line

                Digit 2: Intensity
                0 = unspecified intensity
                1 = weak, decreasing
                2 = weak
                3 = weak, increasing
                4 = moderate, decreasing
                5 = moderate
                6 = moderate, increasing
                7 = strong, decreasing
                8 = strong
                9 = strong, increasing

                Digit 3: Character
                0 = unspecified character
                1 = frontal decreasing
                2 = activity little change
                3 = area increasing
                4 = intertropical
                5 = forming or suspected
                6 = quasi-stationary
                7 = with waves
                8 = diffuse
                9 = position doubtful

        major_color : int
            GEMPAK color code. Must be in the range [0, 32].

        minior_color : int or None
            GEMPAK color code. Must be in the range [0, 32]. This secondary
            color is only used for stationary fronts where the pips can be
            different colors. If None (default), the minor color will be set
            to the major color.

        smooth : int
            Line smoothing parameter. No smoothing (0), splines (1),
            or parametric (2; default).

        pip_size : int
            Size of the pips. Must be in range of [40, 500]. Default is 100.

        pip_stroke : int
            Size multiplier for a stroke. Default is 1. Usage is unknown.

        pip_direction : int
            Direction pips are facing. Right of line (1) or left of line (-1). This
            is reversed for stationary fronts.

        Notes
        -----
        See GEMPAK Appendix C for more information and examples.
        """
        super().__init__()
        if len(lon) != len(lat):
            raise ValueError('Coordinates arrays must be same size.')

        if len(lon) > MAX_POINTS:
            raise ValueError(f'number of points in line must not exceed {MAX_POINTS}')

        self._vg_class = 1
        self._vg_type = 2
        self._lon = np.asarray(lon).astype('float32')
        self._lat = np.asarray(lat).astype('float32')
        self._number_points = len(lon)
        self._min_lon = min(lon)
        self._min_lat = min(lat)
        self._max_lon = max(lon)
        self._max_lat = max(lat)
        self.major_color = major_color
        self.minor_color = major_color if minor_color is None else minor_color
        self.closed = 0
        self.filled = 0
        self._label = b'SPC'
        self.front_code = front_code
        self.pip_size = pip_size
        self.pip_stroke = pip_stroke
        self.pip_direction = pip_direction
        self.smooth = smooth
        self._pts_struct = struct.Struct(f'>{self._number_points * 2}f')
        self._record_size = (self.element_header_struct.size + self.front_struct.size
                             + self._pts_struct.size)

        _code = f'{self._front_code:03d}'
        self._front_type = int(_code[0])
        self._front_intensity = int(_code[1])
        self._front_character = int(_code[2])
        self._width = self._front_intensity

    @property
    def width(self):
        """Get front width."""
        return self._width

    @width.setter
    def width(self, value):
        """Set line width."""
        if value not in range(1, 9):
            raise ValueError('Front width must be in range [1, 8].')
        self._width = value

    def flip(self):
        """Flip front direction."""
        self._pip_direction *= -1

    @property
    def front_code(self):
        """Get front code."""
        return self._front_code

    @front_code.setter
    def front_code(self, value):
        """Set front code."""
        if value not in range(1000):
            raise ValueError('Invalid front code.')
        self._front_code = value

    @property
    def pip_size(self):
        """Get pip size."""
        return self._pip_size

    @pip_size.setter
    def pip_size(self, value):
        """Set pip size."""
        if value not in range(40, 501):
            raise ValueError('Invalid pip size.')
        self._pip_size = value

    @property
    def pip_stroke(self):
        """Get pip stroke multiplier."""
        return self._pip_stroke

    @pip_stroke.setter
    def pip_stroke(self, value):
        """Set pip stroke multiplier."""
        if value <= 0:
            raise ValueError('Invalid pip stroke multiplier.')
        self._pip_stroke = value

    @property
    def pip_direction(self):
        """Get pip direction."""
        return self._pip_direction

    @pip_direction.setter
    def pip_direction(self, value):
        """Set pip direction."""
        if value not in [-1, 1]:
            raise ValueError('Invalid pip direction.')
        self._pip_direction = value

    def encode(self):
        """Encode front."""
        header = self._make_element_header()

        info = self.front_struct.pack(
            number_points=self._number_points,
            front_code=self._front_code,
            pip_size=self._pip_size,
            pip_stroke=self._pip_stroke,
            pip_direction=self._pip_direction,
            width=self._width,
            label=self._label
        )

        pts = self._pts_struct.pack(
            *self._lat, *self._lon
        )

        return header + info + pts


class SpecialLine(LineBase):
    """Special line element."""

    special_line_struct = NamedStruct(
        [('number_points', 'i'), ('line_type', 'i'), ('stroke', 'i'), ('direction', 'i'),
         ('size', 'f'), ('width', 'i')], '>', 'SpecialLineInfo'
    )

    def __init__(self, lon, lat, color, line_type, closed, filled=0, stroke=1, width=2,
                 smooth=0, size=1, direction=1):
        """Create special line.

        Parameters
        ----------
        lon : `numpy.ndarray`

        lat : `numpy.ndarray`

        color : int
            GEMPAK color code. Must be in the range [0, 32].

        line_type : int
            Special line type code. Options are:

                ball_chain = 1
                zigzag = 2
                scallop = 3
                pointed_arrow = 4
                alt_angle_ticks = 5
                filled_arrow = 6
                box_circles = 7
                two_x = 8
                filled_circles = 9
                line_fill_circle_line = 10
                tick_marks = 11
                line_x_line = 12
                fancy_arrow = 13
                fill_circle_x = 14
                box_x = 15
                line_circle_line = 16
                line_caret_line1 = 17
                line_caret_line2 = 18
                sine_curve = 19
                arrow_dashed = 20
                fill_arrow_dash = 21
                streamline = 22
                double_line = 23
                kink_line1 = 24
                kink_line2 = 25
                z_line = 26

            Additional details can be found in PGPALETTE.

        closed : int
            Toggle closed polygon. Values are 0 (open) or 1 (closed).

        filled : int
            Fill pattern code. Default is no fill (0). Codes are defined as follows:

                Code        Fill Pattern
                0           No fill
                1           No fill
                2           Solid fill
                3           Dashed fill
                4           Diagonal line (low density)
                5           Diagonal line (medium density)
                6           Stairstep
                7           Square
                8           Diagonal line (high density)

        stroke : int
            Stroke multiplier. Used for kink position for kinked lines. Values are
            in the range [25, 75]. Not used for other special line types (set to 1).

        width : int
            Width of special line. Must be in range [1, 10]. Defaults to 2.

        smooth : int
            Line smoothing parameter. No smoothing (0; default), splines (1),
            or parametric (2).

        size : float
            Pattern size. Must be in range [0.1, 10]. Defaults to 1.

        direction : int
            Direction of points. Clockwise (1, default) or counter-clockwise (-1).
        """
        super().__init__(lon, lat, color, width, closed, filled, smooth)
        self.line_type = line_type
        self.direction = direction
        self.size = size
        self._vg_class = 3
        self._vg_type = 20
        self._pts_struct = struct.Struct(f'>{self._number_points * 2}f')
        self._record_size = (self.element_header_struct.size + self.special_line_struct.size
                             + self._pts_struct.size)

        if self._line_type in [24, 25]:
            if stroke not in range(25, 76):
                raise ValueError('stroke multiplier must be in range [25, 75].')
            else:
                self._stroke = stroke
        else:
            self._stroke = 1

    @property
    def line_type(self):
        """Get special line type."""
        return self._line_type

    @line_type.setter
    def line_type(self, value):
        """Set special line type."""
        if value not in range(1, 27):
            raise ValueError('Invalid speical line type.')
        self._line_type = value

    @property
    def direction(self):
        """Get line direction."""
        return self._direction

    @direction.setter
    def direction(self, value):
        if value not in [1, -1]:
            raise ValueError('Line direction must be 1 (CW) or -1 (CCW).')
        self._direction = value

    @property
    def size(self):
        """Get pattern size."""
        return self._size

    @size.setter
    def size(self, value):
        """Set pattern size."""
        if value < 0.1 or value > 10:
            raise ValueError('Pattern size must be in range [0.1, 10].')
        self._size = value

    def encode(self):
        """Encode special line."""
        header = self._make_element_header()

        info = self.special_line_struct.pack(
            number_points=self._number_points,
            line_type=self._line_type,
            stroke=self._stroke,
            direction=self._direction,
            size=self._size,
            width=self._width
        )

        pts = self._pts_struct.pack(
            *self._lat, *self._lon
        )

        return header + info + pts


class SpecialText(TextBase):
    """Special text element."""

    special_text_struct = NamedStruct(
        [('rotation', 'f'), ('text_size', 'f'), ('text_type', 'i'), ('turbulence_symbol', 'i'),
         ('font', 'i'), ('text_flag', 'i'), ('width', 'i'), ('text_color', 'i'),
         ('line_color', 'i'), ('fill_color', 'i'), ('align', 'i'), ('lat', 'f'), ('lon', 'f'),
         ('offset_x', 'i'), ('offset_y', 'i')], '>', 'SpecialTextInfo'
    )

    def __init__(self, lon, lat, text, text_color, size=12, text_type=0, font=22, align=0,
                 edgecolor=None, facecolor=None, rotation=0, offset_x=0, offset_y=0,
                 turbulence_symbol=0):
        """Create special text.

        Parameters
        ----------
        lon : float

        lat : float

        text : str
            Special text string. GEMPAK limit is 255 characters.

        text_color : int
            GEMPAK color code. Must be in the range [0, 32].

        size : float
            Font size in points. Valid options are 10, 12, 14, 18, 24, and 34.
            Default is 12.

        text_type : int
            Special text type. Defaults to 0. Codes are defined as follows:

            Code        Type
            0           General text
            1           Low pressure box (Aviation)
            2           High pressure box (Aviation)
            3           Box with border, filled
            4           Box with border, no fill
            5           Box without border, no fill
            6           Freezing level box (Aviation)
            7           Low level turbulence (Aviation)
            8           Cloud level  (Aviation)
            9           High level turbulence (Aviation)
            10          Underlined text, no fill
            11          Underlined text, filled
            12          Mid level icing (Aviation)
            13          Overlined text, no fill
            14          Overlined text, filled
            15          Mid level cloud (Aviation)
            16          Flight level (Aviation)

        font : int
            Font code. Default is 22. Codes are defined as follows:
                        REGULAR	 ITALIC	  BOLD	   ITALIC-BOLD
            Courier	       1	   11	   21	      31
            Helvetica	   2	   12	   22	      32
            Times	       3	   13	   23	      33

        align : int
            Text alignment relative to (x, y) position. Default is 0. Options are
            left (-1), center (0), or right (1).

        edgecolor : int or None
            GEMPAK color code for box border. If None, it will be set to the text
            color code. Default is None. Must be in the range [0, 32].

        facecolor : int or None
            GEMPAK color code for box fill. If None, it will be set to the text
            color code. Default is None. Must be in the range [0, 32].

        rotation : float
            Text roation. Default is 0. Not typically not used.

        offset_x : int
            Offset in x-direction. Default is 0.

        offset_y : int
            Offset in y-direction. Default is 0.

        turbulence_symbol : int
            Turbulence (or icing) symbol code. Must be within the range [0, 10]. Only used
            for certain aviation text types. Default is 0.

        Notes
        -----
        Even when an edgecolor or facecolor are set, whether they are used is determined
        by the text_type parameter.

        The turbulence/icing symbol codes are only used for specific kinds of aviation
        text elements. For details on how this is displayed in NMAP, see cvgv2x.c in the
        GEMPAK source code.

        Only hardware fonts are fully supported currently. Software fonts are an
        option for more size options. See the GEMPAK TEXT parameter documentation
        for further details.
        """
        self._width = 1
        self._text_flag = 2
        super().__init__(lon, lat, text, text_color, size, font, self._width, align, rotation,
                         offset_x, offset_y, self._text_flag)
        self._vg_class = 5
        self._vg_type = 21
        self.text_type = text_type
        self.edgecolor = text_color if edgecolor is None else edgecolor
        self.facecolor = text_color if facecolor is None else facecolor
        self.turbulence_symbol = turbulence_symbol
        self._record_size = (self.element_header_struct.size + self.special_text_struct.size
                             + len(self._text))

    @property
    def text_type(self):
        """Get text type."""
        return self._text_type

    @text_type.setter
    def text_type(self, value):
        """Set text type."""
        if value not in range(17):
            raise ValueError('Invalid text type.')
        self._text_type = value

    @property
    def edgecolor(self):
        """Get textbox edgecolor."""
        return self._edgecolor

    @edgecolor.setter
    def edgecolor(self, value):
        """Set text box edgecolor."""
        if value not in range(33):
            raise ValueError('Invalid edgecolor.')
        self._edgecolor = value

    @property
    def facecolor(self):
        """Get textbox facecolor."""
        return self._facecolor

    @facecolor.setter
    def facecolor(self, value):
        """Set text box facecolor."""
        if value not in range(33):
            raise ValueError('Invalid facecolor.')
        self._facecolor = value

    @property
    def turbulence_symbol(self):
        """Get turbulence symbol code."""
        return self._turbulence_symbol

    @turbulence_symbol.setter
    def turbulence_symbol(self, value):
        """Set turbulence symbol."""
        if self._text_type in [7, 9] and value not in range(9):
            raise ValueError('Invalid turbulence symbol code.')
        elif self._text_type == 12 and value not in range(11):
            raise ValueError('Invalid icing symbol code.')
        self._turbulence_symbol = value

    def encode(self):
        """Encode special text."""
        header = self._make_element_header()

        info = self.special_text_struct.pack(
            rotation=self._rotation,
            text_size=self.points_to_size(self._size),
            text_type=self._text_type,
            turbulence_symbol=self._turbulence_symbol,
            font=self._font,
            text_flag=self._text_flag,
            width=self._width,
            text_color=self._text_color,
            line_color=self._edgecolor,
            fill_color=self._facecolor,
            align=self._align,
            lat=self._lat,
            lon=self._lon,
            offset_x=self._offset_x,
            offset_y=self._offset_y
        )

        text = struct.pack(f'{len(self._text)}s', self._text.encode())

        return header + info + text


class VGFile:
    """GEMPAK Vector Graphics Format encoder class."""

    def __init__(self):
        self.elements = []
        self.add_element(
            FileHeader()
        )

    def add_element(self, element):
        """Add vector graphic element to VGF."""
        self.elements.append(element)

    @classmethod
    def from_elements(cls, *args):
        """Create VGF from elements."""
        new = cls()

        for element in args:
            new.add_element(element)

        return new

    def to_vgf(self, path):
        """Output to VGF.

        Parameters
        ----------
        path : str of `path.Pathlib`
        """
        with open(path, 'wb') as out:
            for e in self.elements:
                out.write(e.encode())
