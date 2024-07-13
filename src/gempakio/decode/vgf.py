# Copyright (c) 2024 Nathan Wendt.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Classes for decoding GEMPAK VGF files."""

import contextlib
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import logging
import re
import sys

import numpy as np

from gempakio.tools import IOBuffer, NamedStruct

logger = logging.getLogger(__name__)

LIST_MEMBER_SIZE = 9
MAX_ASH = 50
MAX_COUNTIES = 400
MAX_JET_POINTS = 50
MAX_POINTS = 500
MAX_SGWX_POINTS = 256
MAX_SIGMET = 100
MAX_TRACKS = 50
TRACK_DT_SIZE = 18
VGF_HEADER_SIZE = 40


class VGClass(Enum):
    """Values for vg_class from drwids.h."""

    header = 0
    fronts = 1
    watches = 2
    lines = 3
    symbols = 4
    text = 5
    winds = 6
    every = 7
    comsym = 8
    products = 9
    tracks = 10
    sigmets = 11
    circle = 12
    marker = 13
    lists = 14
    met = 15
    blank = 16


class VGType(Enum):
    """Values for vg_type from vgstruct.h."""

    line = 1
    front = 2
    circle = 4
    weather_symbol = 5
    watch_box = 6
    watch_county = 7
    wind_barb = 8
    wind_arrow = 9
    cloud_symbol = 10
    icing_symbol = 11
    pressure_tendency_symbol = 12
    past_weather_symbol = 13
    sky_cover = 14
    special_symbol = 15
    turbulence_symbol = 16
    text = 17
    justified_text = 18
    marker = 19
    special_line = 20
    special_text = 21
    file_header = 22
    directional_arrow = 23
    hash_mark = 24
    combination_weather_symbol = 25
    storm_track = 26
    international_sigmet = 27
    nonconvective_sigmet = 28
    convective_sigmet = 29
    convective_outlook = 30
    airmet = 31
    ccf = 32
    watch_status = 33
    lists = 34
    volcano = 35
    ash_cloud = 36
    jet = 37
    gfa = 38
    tca = 39
    tc_error_cone = 40
    tc_track = 41
    tc_break_point = 42
    sgwx = 43


class LineType(Enum):
    """Values for lintyp from GEMPAK."""

    dotted = 0
    solid = 1
    short_dashed = 2
    medium_dashed = 3
    long_dash_short_dash = 4
    long_dash = 5
    long_dash_three_short_dash = 6
    long_dash_dot = 7
    long_dash_three_dot = 8
    extra_long_dash_two_dot = 9


class SpecialLineType(Enum):
    """Values for spltyp from settings.tbl."""

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


class MarkerType(Enum):
    """Values for mrktyp from GEMPAK."""

    none = 0
    plus_sign = 1
    octagon = 2
    triangle = 3
    box = 4
    small_x = 5
    diamond = 6
    up_arrow = 7
    x_bar = 8
    z_bar = 9
    y = 10
    box_diagonals = 11
    asterisk = 12
    hourglass = 13
    star = 14
    dot = 15
    large_x = 16
    filled_octagon = 17
    filled_triangle = 18
    filled_box = 19
    filled_diamond = 20
    filled_star = 21
    minus_sign = 22
    tropical_storm = 23
    hurricane = 24


class ListType(Enum):
    """Values for list types."""

    county = 1
    zone = 2
    wfo = 3
    state = 4
    marine_county = 5


class FrontType(Enum):
    """Values for front types."""

    stationary = 0
    stationary_aloft = 1
    warm = 2
    warm_aloft = 3
    cold = 4
    cold_aloft = 5
    occluded = 6
    dryline = 7
    intertropical = 8
    convergence = 9


class FrontIntensity(Enum):
    """Values for front intensity."""

    unspecified = 0
    weak_decreasing = 1
    weak = 2
    weak_increasing = 3
    moderate_decreasing = 4
    moderate = 5
    moderate_increasing = 6
    strong_decreasing = 7
    strong = 8
    strong_increasing = 9


class FrontCharacter(Enum):
    """Values for front character."""

    unspecified = 0
    frontal_decreasing = 1
    little_change = 2
    area_increasing = 3
    intertropical = 4
    forming_suspected = 5
    quasi_stationary = 6
    with_waves = 7
    diffuse = 8
    position_doubtful = 9


class Basin(Enum):
    """Values for TCA basin."""

    atlantic = 0
    east_pacific = 1
    central_pacific = 2
    west_pacific = 3


class Severity(Enum):
    """Values for TCA severity."""

    tropical_storm = 0
    hurricane = 1


class StormType(Enum):
    """Values for TCA storm type."""

    hurricane = 0
    tropical_storm = 1
    tropical_depression = 2
    subtropical_storm = 3
    subtropical_depression = 4


class AdvisoryType(Enum):
    """Values for TCA advisory type."""

    watch = 0
    warning = 1


class SpecialGeography(Enum):
    """Values for TCA special geography type."""

    no_types = 0
    islands = 1
    water = 2


class TropicalWatchWarningLevel(Enum):
    """Values for watch-warning level."""

    hurricane_warning = 0
    hurricane_watch = 1
    tropical_storm_warning = 2
    tropical_storm_watch = 3


class WatchType(Enum):
    """Values for watch type."""

    tornado = 2
    severe_thunderstorm = 6


@dataclass
class VectorGraphicAttribute:
    """Vector graphic attribute base class."""

    def __repr__(self):
        """Return repr(self)."""
        return (f'{type(self).__qualname__}'
                f'({", ".join([f"{k}={v}" for k, v in vars(self).items()])})')


class BarbAttribute(VectorGraphicAttribute):
    """Barb attribute."""

    def __init__(self, wind_color, number_wind, width, size, wind_type, head_size, speed,
                 direction, lat, lon, flight_level_color, text_rotation, text_size, text_type,
                 turbulence_symbol, font, text_flag, text_width, text_color, line_color,
                 fill_color, align, text_lat, text_lon, offset_x, offset_y, text):
        """Create wind barb attribute.

        Parameters
        ----------
        wind_color : int

        number_wind : int

        width : int

        size : float

        wind_type : int

        head_size : float

        speed : float

        direction : float

        lat : float

        lon : float

        flight_level_color : int

        text_rotation : float

        text_size : float

        text_type : int

        turbulence_symbol : int

        font : int

        text_flag : int

        text_width : int

        text_color : int

        line_color : int

        fill_color : int

        align : int

        text_lat : float

        text_lon : float

        offset_x : int

        offset_y : int

        text : str
        """
        self.wind_color = wind_color
        self.number_wind = number_wind
        self.width = width
        self.size = size
        self.wind_type = wind_type
        self.head_size = head_size
        self.speed = speed
        self.direction = direction
        self.lat = lat
        self.lon = lon
        self.flight_level_color = flight_level_color
        self.text_rotation = text_rotation
        self.text_size = text_size
        self.text_type = text_type
        self.turbulence_symbol = turbulence_symbol
        self.font = font
        self.text_flag = text_flag
        self.text_width = text_width
        self.text_color = text_color
        self.line_color = line_color
        self.fill_color = fill_color
        self.align = align
        self.text_lat = text_lat
        self.text_lon = text_lon
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.text = text


class BreakPointAttribute(VectorGraphicAttribute):
    """Break point attribute."""

    def __init__(self, lat, lon, name):
        """Create break point attribute.

        Parameters
        ----------
        lat : float

        lon : float

        name : str
        """
        self.lat = lat
        self.lon = lon
        self.name = name


class TrackAttribute(VectorGraphicAttribute):
    """Tropical cylcone track attribute."""

    def __init__(self, advisory_date, tau, max_wind, wind_gust, minimum_pressure,
                 development_level, development_label, direction, speed, date_label,
                 storm_source, lat, lon):
        """Create tropical cyclone track attribute.

        Parameters
        ----------
        advisory_date : str

        tau : str

        max_wind : str

        wind_gust : str

        minimum_pressure : str

        development_level : str

        development_label : str

        direction : str

        speed : str

        date_label : str

        storm_source : str

        lat : float

        lon : float
        """
        self.advisory_date = advisory_date
        self.tau = tau
        self.max_wind = max_wind
        self.wind_gust = wind_gust
        self.minimum_pressure = minimum_pressure
        self.development_level = development_level
        self.development_label = development_label
        self.direction = direction
        self.speed = speed
        self.date_label = date_label
        self.storm_source = storm_source
        self.lat = lat
        self.lon = lon


class HashAttribute(VectorGraphicAttribute):
    """Hash attribute."""

    def __init__(self, wind_color, number_wind, width, size, wind_type,
                 head_size, speed, direction, lat, lon):
        """Create hash attribute.

        Parameters
        ----------
        wind_color : int

        number_wind : int

        width : int

        size : float

        wind_type : int

        head_size : float

        speed : float

        direction : float

        lat : float

        lon : float
        """
        self.wind_color = wind_color
        self.number_wind = number_wind
        self.width = width
        self.size = size
        self.wind_type = wind_type
        self.head_size = head_size
        self.speed = speed
        self.direction = direction
        self.lat = lat
        self.lon = lon


class LineAttribute(VectorGraphicAttribute):
    """Line attribute."""

    def __init__(self, line_color, number_points, line_type, stroke,
                 direction, size, width, lat, lon):
        """Create line attribute.

        Parameters
        ----------
        line_color : int

        number_points : int

        line_type : int

        stroke : int

        direction : float

        size : float

        width : int

        lat : `numpy.ndarray`

        lon : `numpy.ndarray`
        """
        self.line_color = line_color
        self.number_points = number_points
        self.line_type = line_type
        self.stroke = stroke
        self.direction = direction
        self.size = size
        self.width = width
        self.lat = lat
        self.lon = lon


@dataclass
class VectorGraphicElement:
    """Base class for VGF elements."""

    def __init__(self, header_struct):
        """Vector graphic element.

        Parameters
        ----------
        header_struct : `NamedStruct`
        """
        self.delete = header_struct.delete
        self.vg_type = header_struct.vg_type
        self.vg_class = header_struct.vg_class
        self.filled = header_struct.filled
        self.closed = header_struct.closed
        self.smooth = header_struct.smooth
        self.version = header_struct.version
        self.group_type = header_struct.group_type
        self.group_number = header_struct.group_number
        self.major_color = header_struct.major_color
        self.minor_color = header_struct.minor_color
        self.record_size = header_struct.record_size
        self.min_lat = header_struct.min_lat
        self.min_lon = header_struct.min_lon
        self.max_lat = header_struct.max_lat
        self.max_lon = header_struct.max_lon

    def __repr__(self):
        """Return repr(self)."""
        return (f'{type(self).__qualname__}'
                f'[{VGClass(self.vg_class).name}, {VGType(self.vg_type).name}]')

    @property
    def bounds(self):
        """Get bounding box of element."""
        if (hasattr(self, 'lat') and hasattr(self, 'lon')
           and len(np.atleast_1d(self.lat)) > 1):
            xmin = min(self.lon)
            xmax = max(self.lon)
            ymin = min(self.lat)
            ymax = max(self.lat)

            return (ymin, xmin, ymax, xmax)
        else:
            raise NotImplementedError(f'bounds undefined for {type(self).__qualname__}')


class AshCloudElement(VectorGraphicElement):
    """Ash cloud element."""

    def __init__(self, header_struct, subtype, number_points, distance, forecast_hour,
                 line_type, line_width, side_of_line, speed, speeds, direction,
                 flight_level1, flight_level2, rotation, text_size, text_type,
                 turbulence_symbol, font, text_flag, width, text_color, line_color,
                 fill_color, align, text_lat, text_lon, offset_x, offset_y, text,
                 lat, lon):
        """Create ash cloud element.

        Parameters
        ----------
        header_struct : `NamedStruct`

        subtype : int
            Type of ash cloud (side of line, line, area).

        number_points : int

        distance : float

        forecast_hour : int

        line_type : int

        line_width : int

        side_of_line : int

        speed : float

        speeds : str

        direction : str

        flight_level1 : str

        flight_level2 : str

        rotation : float

        text_size : float

        text_type : int

        turbulence_symbol : int

        font : int

        text_flag : int

        text_width : int

        text_color : int

        line_color : int

        fill_color : int

        align : int

        text_lat : float

        text_lon : float

        offset_x : int

        offset_y : int

        text : str

        lat : `numpy.ndarray`

        lon : `numpy.ndarray`
        """
        super().__init__(header_struct)
        self.subtype = subtype
        self.number_points = number_points
        self.distance = distance
        self.forecast_hour = forecast_hour
        self.line_type = line_type
        self.line_width = line_width
        self.side_of_line = side_of_line
        self.speed = speed
        self.speeds = speeds
        self.direction = direction
        self.flight_level1 = flight_level1
        self.flight_level2 = flight_level2
        self.rotation = rotation
        self.text_size = text_size
        self.text_type = text_type
        self.turbulence_symbol = turbulence_symbol
        self.font = font
        self.text_flag = text_flag
        self.width = width
        self.text_color = text_color
        self.line_color = line_color
        self.fill_color = fill_color
        self.align = align
        self. text_lat = text_lat
        self.text_lon = text_lon
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.text = text
        self.lat = lat
        self.lon = lon


class CircleElement(VectorGraphicElement):
    """Circle element."""

    def __init__(self, header_struct, number_points, line_type,
                 line_type_hardware, width, line_width_hardware,
                 lat, lon):
        """Create circle element.

        Parameters
        ----------
        header_struct : `NamedStruct`

        number_points : int

        line_type : int
            Integer code defining line type. See Appendix C in
            GEMPAK documentation for details.

        line_type_hardware : int

        width : int

        line_width_hardware : int

        lat : `numpy.ndarray`

        lon : `numpy.ndarray`
        """
        super().__init__(header_struct)
        self.number_points = number_points
        self.line_type = line_type
        self.line_type_hardware = line_type_hardware
        self.width = width
        self.line_width_hardware = line_width_hardware
        self.lat = lat
        self.lon = lon

        self._set_properties()

    def _set_properties(self):
        """Decode line type number to set properties."""
        code = f'{self.line_type:02d}'
        self.line_modifier = int(code[0])
        self.line_style = LineType(int(code[1])).name


class CollaborativeConvectiveForecastElement(VectorGraphicElement):
    """CCF element."""

    def __init__(self, header_struct, subtype, number_points, coverage, storm_tops,
                 probability, growth, speed, direction, text_lat, text_lon, arrow_lat,
                 arrow_lon, high_fill, med_fill, low_fill, line_type, arrow_size,
                 rotation, text_size, text_type, turbulence_symbol, font, text_flag, width,
                 fill_color, align, offset_x, offset_y, text, text_layout, lat, lon):
        """Create CCF element.

        Parameters
        ----------
        header_struct : `NamedStruct`

        subtype : int

        number_points : int

        coverage : int

        storm_tops : int

        probabiltiy : int

        growth : int

        speed : float

        direction : float

        text_lat : float

        text_lon : float

        arrow_lat : float

        arrow_lon : float

        high_fill : int

        med_fill : int

        low_fill : int

        line_type : int

        arrow_size : float

        rotation : float

        text_size : float

        text_type : int

        turbulence_symbol : int

        font : int

        text_flag : int

        text_width : int

        text_color : int

        line_color : int

        fill_color : int

        align : int

        text_lat : float

        text_lon : float

        offset_x : int

        offset_y : int

        text : str

        lat : `numpy.ndarray`

        lon : `numpy.ndarray`
        """
        super().__init__(header_struct)
        self.subtype = subtype
        self.number_points = number_points
        self.coverage = coverage
        self.storm_tops = storm_tops
        self.probability = probability
        self.growth = growth
        self.speed = speed
        self.direction = direction
        self.text_lat = text_lat
        self.text_lon = text_lon
        self.arrow_lat = arrow_lat
        self.arrow_lon = arrow_lon
        self.high_fill = high_fill
        self.med_fill = med_fill
        self.low_fill = low_fill
        self.line_type = line_type
        self.arrow_size = arrow_size
        self.rotation = rotation
        self.text_size = text_size
        self.text_type = text_type
        self.turbulence_symbol = turbulence_symbol
        self.font = font
        self.text_flag = text_flag
        self.width = width
        self.fill_color = fill_color
        self.align = align
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.text = text
        self.text_layout = text_layout
        self.lat = lat
        self.lon = lon


class FileHeaderElement(VectorGraphicElement):
    """File header element."""

    def __init__(self, header_struct, version, notes):
        """Create file header element.

        Parameters
        ----------
        header_struct : `NamedStruct`

        version : str

        notes : str
        """
        super().__init__(header_struct)
        self.gempak_version = version
        self.notes = notes


class FrontElement(VectorGraphicElement):
    """Front element."""

    def __init__(self, header_struct, number_points, front_code, pip_size,
                 pip_stroke, pip_direction, width, label, lat, lon):
        """Create front element.

        Parameters
        ----------
        header_struct : `NamedStruct`

        number_points : int

        front_code : int
            Integer code defining the front type, intensity, and character.
            See Appendix C in GEMPAK documentation for details.

        pip_size : int
            Size of barbs, scallops, etc.

        pip_stroke : int
            GEMPAK color code.

        pip_direction : int
            Direction pips are facing. Right of line (1) or left of line (-1).

        width : int
            Width of front line.

        label : str
            Front label. If present, typically will be string front code.

        lat : `numpy.ndarray`

        lon : `numpy.ndarray`
        """
        super().__init__(header_struct)
        self.number_points = number_points
        self.front_code = front_code
        self.pip_size = pip_size
        self.pip_stroke = pip_stroke
        self.pip_direction = pip_direction
        self.width = width
        self.label = label
        self.lat = lat
        self.lon = lon

        self._set_properties()

    def _set_properties(self):
        """Decode front code to set properties."""
        code = f'{self.front_code:03d}'
        self.front_type = FrontType(int(code[0])).name
        self.front_intensity = FrontIntensity(int(code[1])).name
        self.front_character = FrontCharacter(int(code[2])).name


class GraphicalForecastAreaElement(VectorGraphicElement):
    """GFA element."""

    def __init__(self, header_struct, number_blocks, number_points, blocks,
                 lat, lon):
        """Create GFA element.

        Parameters
        ----------
        header_struct : `NamedStruct`

        number_blocks : int

        number_points : int

        blocks : str

        lat : `numpy.ndarray`

        lon : `numpy.ndarray`
        """
        super().__init__(header_struct)
        self.number_blocks = number_blocks
        self.number_points = number_points
        self.blocks = blocks
        self.lat = lat
        self.lon = lon


class JetElement(VectorGraphicElement):
    """Jet element."""

    def __init__(self, header_struct, line_attribute, number_barbs, barb_attributes,
                 number_hashes, hash_attributes):
        """Create jet element.

        Parameters
        ----------
        header_struct : `NamedStruct`

        line_attribute : `LineAttribute`

        number_barbs : int

        barb_attributes : array_like of `BarbAttribute`

        number_hashes : int

        hash_attribute : array_like of `HashAttribute`
        """
        super().__init__(header_struct)
        self.line_attribute = line_attribute
        self.number_barbs = number_barbs
        self.barb_attributes = barb_attributes
        self.number_hashes = number_hashes
        self.hash_attributes = hash_attributes


class LineElement(VectorGraphicElement):
    """Line element."""

    def __init__(self, header_struct, number_points, line_type,
                 line_type_hardware, width, line_width_hardware,
                 lat, lon):
        """Create line element.

        Parameters
        ----------
        header_struct : `NamedStruct`

        number_points : int

        line_type : int
            Integer code defining line type. See Appendix C in
            GEMPAK documentation for details.

        line_type_hardware : int

        line_type_hardware : int

        width : int

        line_width_hardware : int

        lat : `numpy.ndarray`

        lon : `numpy.ndarray`
        """
        super().__init__(header_struct)
        self.number_points = number_points
        self.line_type = line_type
        self.line_type_hardware = line_type_hardware
        self.width = width
        self.line_width_hardware = line_width_hardware
        self.lat = lat
        self.lon = lon

        self._set_properties()

    def _set_properties(self):
        """Decode line type number to set properties."""
        code = f'{self.line_type:02d}'
        self.line_modifier = int(code[0])
        self.line_style = LineType(int(code[1])).name


class ListElement(VectorGraphicElement):
    """List element."""

    def __init__(self, header_struct, list_type, marker_type,
                 marker_size, marker_width, number_items,
                 list_items, lat, lon):
        """Create list element.

        Parameters
        ----------
        header_struct : `NamedStruct`

        list_type : int
            County (1), zone (2), WFO (3), state (4), or marine (5).

        marker_type : int
            Integer code defining the symbol used for the marker. See
            Appendix C in GEMPAK documentation for details.

        marker_size : float

        number_items : int

        list_items : array_like of str

        lat : `numpy.ndarray`
            Latitude of list item centroid.

        lon : `numpy.ndarray`
            Longitude of list item centroid.
        """
        super().__init__(header_struct)
        self.list_type = list_type
        self.marker_type = marker_type
        self.marker_size = marker_size
        self.marker_width = marker_width
        self.number_items = number_items
        self.list_items = list_items
        self.lat = lat
        self.lon = lon

        self._set_properties()

    def _set_properties(self):
        """Set list properties using header information."""
        self.list_type_name = ListType(self.list_type).name
        self.marker_name = MarkerType(self.marker_type).name


class SigmetElement(VectorGraphicElement):
    """"SIGMET element."""

    def __init__(self, header_struct, subtype, number_points, line_type, line_width,
                 side_of_line, area, flight_info_region, status, distance, message_id,
                 sequence_number, start_time, end_time, remarks, sonic, phenomena,
                 phenomena2, phenomena_name, phenomena_lat, phenomena_lon, pressure,
                 max_wind, free_text, trend, movement, type_indicator, type_time,
                 flight_level, speed, direction, tops, forecaster, lat, lon):
        """Create SIGMET element.

        Parameters
        ----------
        header_struct : `NamedStruct`

        subtype : int
            Type of SIGMET area (side of line, line, area).

        number_points : int

        line_type : int
            Integer code defining special line type. See Appendix C
            in GEMPAK documentation for details.

        line_width : int

        side_of_line : int

        area : str
            MWO inidicator of unit.

        flight_info_region : str
            Location indicator of FIR unit.

        status : int
            New (0), amend (1), or cancel (2).

        distance : float

        message_id : str

        sequence_number : int

        start_time : str
            Format: ddHHMM.

        end_time : str
            Format: ddHHMM.

        marker_type : int
            Integer code defining the symbol used for the marker. See
            Appendix C in GEMPAK documentation for details.

        marker_size : float

        number_items : int

        remarks : str

        sonic : int
            Supersonic indicator (0 or 1).

        phenomena : str

        phenomena2 : str

        phenomena_name : str

        phenomena_lat : str

        phenomena_lon : str

        pressure : int

        max_wind : int

        free_text : str

        trend : str

        movement : str

        type_indicator : int
            Obs or forecast. (0, 1, 2)

        type_time :  str
            Format: ddHHMM.

        flight_level : int

        speed : int

        direction : str
            Direction of phenomena.

        tops : str

        forecaster : str

        lat : `numpy.ndarray`
            Latitude of list item centroid.

        lon : `numpy.ndarray`
            Longitude of list item centroid.
        """
        super().__init__(header_struct)
        self.subtype = subtype
        self.number_points = number_points
        self.line_type = line_type
        self.line_width = line_width
        self.side_of_line = side_of_line
        self.area = area
        self.flight_info_region = flight_info_region
        self.status = status
        self.distance = distance
        self.message_id = message_id
        self.sequence_number = sequence_number
        self.start_time = start_time
        self.end_time = end_time
        self.remarks = remarks
        self.sonic = sonic
        self.phenomena = phenomena
        self.phenomena2 = phenomena2
        self.phenomena_name = phenomena_name
        self.phenomena_lat = phenomena_lat
        self.phenomena_lon = phenomena_lon
        self.pressure = pressure
        self.max_wind = max_wind
        self.free_text = free_text
        self.trend = trend
        self.movement = movement
        self.type_indicator = type_indicator
        self.type_time = type_time
        self.flight_level = flight_level
        self.speed = speed
        self.direction = direction
        self.tops = tops
        self.forecaster = forecaster
        self.lat = lat
        self.lon = lon


class SignificantWeatherElement(VectorGraphicElement):
    """SGWX element."""

    def __init__(self, header_struct, subtype, number_points, text_lat, text_lon,
                 arrow_lat, arrow_lon, line_element, line_type, line_width,
                 arrow_size, special_symbol, weather_symbol, text_rotation,
                 text_size, text_type, turbulence_symbol, font, text_flag,
                 text_width, text_color, line_color, fill_color, text_align,
                 offset_x, offset_y, text, area_lat, area_lon):
        """Create special line element.

        Parameters
        ----------
        header_struct : `NamedStruct`

        subtype : int

        number_points : int

        text_lat : float

        text_lon : float

        arrow_lat : float

        arrow_lon : float

        line_element : int

        line_type : int

        line_width : int

        arrow_size : float

        special_symbol : int

        weather_symbol : int

        text_rotation : float

        rotation : float
            Text rotation.

        size : float
            GEMPAK font size code. See getSztext in cvgv2x.c. in
            GEMPAK source for conversion to points.

        text_type : int
            Integer code for the type of text (e.g., general, aviation).
            cvgv2x.c in GEMPAK source contains more details.

        turbulence_symbol : int
            Integer code for turbulence symbol. Used if the text_type is
            in the aviation family. See Appendix C in GEMPAK documentation
            for details on turbulence symbol codes.

        font : int
            Integer code defining the font to use. See getFontStyle in cvgv2x.c
            in GEMPAK source for details.

        text_flag : int
            Flag for using hardware or software font.

        width : int

        text_color : int
            GEMPAK color code for text.

        line_color : int
            GEMPAK color code for lines (i.e., box, underline).

        fill_color : int
            GEMPAK color code for text box fill.

        text_align : int
            Integer code for text alignment. Center (0), left (-1),
            or right (1).

        offset_x : int
            Symbol offset in x direction.

        offset_y : int
            Symbole offset in y direction.

        text : str

        area_lat : `numpy.ndarray`

        area_lon : `numpy.ndarray`
        """
        super().__init__(header_struct)
        self.subtype = subtype
        self.number_points = number_points
        self.text_lat = text_lat
        self.text_lon = text_lon
        self.arrow_lat = arrow_lat
        self.arrow_lon = arrow_lon
        self.line_element = line_element
        self.line_type = line_type
        self.line_width = line_width
        self.arrow_size = arrow_size
        self.special_symbol = special_symbol
        self.weather_symbol = weather_symbol
        self.text_rotation = text_rotation
        self.text_size = text_size
        self.text_type = text_type
        self.turbulence_symbol = turbulence_symbol
        self.font = font
        self.text_flag = text_flag
        self.text_width = text_width
        self.text_color = text_color
        self.line_color = line_color
        self.fill_color = fill_color
        self.text_align = text_align
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.text = text
        self.area_lat = area_lat
        self.area_lon = area_lon


class SpecialLineElement(VectorGraphicElement):
    """Special line element."""

    def __init__(self, header_struct, number_points, line_type,
                 stroke, direction, size, width, lat, lon):
        """Create special line element.

        Parameters
        ----------
        header_struct : `NamedStruct`

        number_points : int

        line_type : int
            Integer code defining special line type. See Appendix C
            in GEMPAK documentation for details.

        stroke : int
            GEMPAK color code.

        direction : int
            CW (1) or CCW (-1).

        size : float
            Size of special line elements (e.g., dots, crosses, arrow heads, etc.).

        width : int

        lat : `numpy.ndarray`

        lon : `numpy.ndarray`
        """
        super().__init__(header_struct)
        self.number_points = number_points
        self.line_type = line_type
        self.stroke = stroke
        self.direction = direction
        self.size = size
        self.width = width
        self.lat = lat
        self.lon = lon

        self._set_properties()

    def _set_properties(self):
        """Set properties using special line type code."""
        self.line_style = SpecialLineType(self.line_type).name


class SpecialTextElement(VectorGraphicElement):
    """Special text element."""

    def __init__(self, header_struct, rotation, size, text_type,
                 turbulence_symbol, font, text_flag, width, text_color,
                 line_color, fill_color, align, lat, lon, offset_x,
                 offset_y, text):
        """Create special text element.

        Parameters
        ----------
        header_struct : `NamedStruct`

        rotation : float
            Text rotation.

        size : float
            GEMPAK font size code. See getSztext in cvgv2x.c. in
            GEMPAK source for conversion to points.

        text_type : int
            Integer code for the type of text (e.g., general, aviation).
            cvgv2x.c in GEMPAK source contains more details.

        turbulence_symbol : int
            Integer code for turbulence symbol. Used if the text_type is
            in the aviation family. See Appendix C in GEMPAK documentation
            for details on turbulence symbol codes.

        font : int
            Integer code defining the font to use. See getFontStyle in cvgv2x.c
            in GEMPAK source for details.

        text_flag : int
            Flag for using hardware or software font.

        width : int

        text_color : int
            GEMPAK color code for text.

        line_color : int
            GEMPAK color code for lines (i.e., box, underline).

        fill_color : int
            GEMPAK color code for text box fill.

        align : int
            Integer code for text alignment. Center (0), left (-1),
            or right (1).

        lat : `numpy.ndarray`

        lon : `numpy.ndarray`

        offset_x : int
            Symbol offset in x direction.

        offset_y : int
            Symbole offset in y direction.

        text : str
        """
        super().__init__(header_struct)
        self.rotation = rotation
        self.size = size
        self.text_type = text_type
        self.turbulence_symbol = turbulence_symbol
        self.font = font
        self.text_flag = text_flag
        self.width = width
        self.text_color = text_color
        self.line_color = line_color
        self.fill_color = fill_color
        self.align = align
        self.lat = lat
        self.lon = lon
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.text = text


class SymbolElement(VectorGraphicElement):
    """Symbol element."""

    def __init__(self, header_struct, number_symbols, width, size,
                 symbol_type, symbol_code, offset_x, offset_y, lat, lon):
        """Create symbol element.

        Parameters
        ----------
        header_struct : `NamedStruct`

        number_symbols : int

        width : int

        size : float

        symbol_type : int
            Not filled (1) or filled (2).

        symbol_code : int
            Integer code defining the symbole. See Appendix C in
            GEMPAK documentation for details.

        offset_x : int
            Symbol offset in x direction.

        offset_y : int
            Symbole offset in y direction.

        lat : `numpy.ndarray`

        lon : `numpy.ndarray`
        """
        super().__init__(header_struct)
        self.number_symbols = number_symbols
        self.width = width
        self.size = size
        self.symbol_type = symbol_type
        self.symbol_code = symbol_code
        self.lat = lat
        self.lon = lon
        self.offset_x = offset_x
        self.offset_y = offset_y


class TextElement(VectorGraphicElement):
    """Text element."""

    def __init__(self, header_struct, rotation, size, font,
                 text_flag, width, align, lat, lon, offset_x,
                 offset_y, text):
        """Create text element.

        Parameters
        ----------
        header_struct : `NamedStruct`

        rotation : float
            Text rotation.

        size : float
            GEMPAK font size code. See getSztext in cvgv2x.c. in
            GEMPAK source for conversion to points.

        font : int
            Integer code defining the font to use. See getFontStyle in cvgv2x.c
            in GEMPAK source for details.

        text_flag : int
            Flag for using hardware or software font.

        width : int

        align : int
            Integer code for text alignment. Center (0), left (-1),
            or right (1).

        lat : `numpy.ndarray`

        lon : `numpy.ndarray`

        offset_x : int
            Symbol offset in x direction.

        offset_y : int
            Symbole offset in y direction.
        """
        super().__init__(header_struct)
        self.rotation = rotation
        self.size = size
        self.font = font
        self.text_flag = text_flag
        self.width = width
        self.align = align
        self.lat = lat
        self.lon = lon
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.text = text


class TrackElement(VectorGraphicElement):
    """Storm track element."""

    def __init__(self, header_struct, track_type, total_points, initial_points,
                 initial_line_type, extrapolated_line_type, initial_mark_type,
                 extrapolated_mark_type, line_width, speed, direction, increment,
                 skip, font, font_flag, font_size, times, lat, lon):
        """Create storm track element.

        Parameters
        ----------
        header_struct : `NamedStruct`

        track_type : int

        total_points : int

        initial_points : int
            Number of points from user to calculate track.

        initial_line_type : int
            Integer code for line type of initial points. See Appendix C
            in GEMPAK documentation for details.

        extrapolated_line_type : int
            Integer code for line type of extrapolated points. See
            Appendix C in GEMPAK documentation for details.

        initial_mark_type : int
            Integer code for markers of initial points. See Appendix C
            in GEMPAK documentation for details.

        extrapolated_mark_type : int
            Integer code for markers of extrapolated points. See
            Appendix C in GEMPAK documentation for details.

        line_width : int

        speed : float
            Storm speed in knots between last two initial points.

        direction : float
            Storm direction in degrees between last two initial points.

        increment : int
            Increment (in minutes) between extrapolated points.

        skip : int
            Skip factor for extrapolated point labels.

        font : int
            Integer code defining the font to use. See getFontStyle in cvgv2x.c
            in GEMPAK source for details.

        font_flag : int
            Flag for using hardware or software font.

        font_size : float
            GEMPAK font size code. See getSztext in cvgv2x.c. in
            GEMPAK source for conversion to points.

        times : array_like of str
            Array of time labels for points.

        lat : `numpy.ndarray`

        lon : `numpy.ndarray`
        """
        super().__init__(header_struct)
        self.track_type = track_type
        self.total_points = total_points
        self.initial_points = initial_points
        self.initial_line_type = initial_line_type
        self.extrapolated_line_type = extrapolated_line_type
        self.initial_mark_type = initial_mark_type
        self.extrapolated_mark_type = extrapolated_mark_type
        self.line_width = line_width
        self.speed = speed
        self.direction = direction
        self.increment = increment
        self.skip = skip
        self.font = font
        self.font_flag = font_flag
        self.font_size = font_size
        self.times = times
        self.lat = lat
        self.lon = lon


class TropicalCycloneBase(VectorGraphicElement):
    """Base class for TC elements."""

    def __init__(self, header_struct, storm_number, issue_status, basin, advisory_number,
                 storm_name, storm_type, valid_time, tz, forecast_period):
        """TC base class.

        Parameters
        ----------
        header_struct : `NamedStruct`

        storm_number : str

        issue_status : str

        basin : int

        advisory_number : str

        storm_name : str

        valid_time : str

        tz : str

        forecast_period : str
        """
        super().__init__(header_struct)
        self.storm_number = storm_number
        self.issue_status = issue_status
        self.basin = Basin(basin)
        self.advisory_number = advisory_number
        self.storm_name = storm_name
        self.storm_type = StormType(storm_type)
        self.valid_time = datetime.strptime(
            valid_time, '%y%m%d/%H%M'
        ).replace(tzinfo=timezone.utc)
        self.timezone = tz
        self.forecast_period = forecast_period


class TropicalCycloneAdvisoryElement(VectorGraphicElement):
    """TCA element."""

    def __init__(self, header_struct, storm_number, issue_status, basin, advisory_number,
                 storm_name, storm_type, valid_time, tz, text_lat, text_lon,
                 text_font, text_size, text_width, number_ww, ww):
        """Create TCA element.

        Parameters
        ----------
        header_struct : `NamedStruct`

        storm_number : str

        issue_status : str

        basin : int

        advisory_number : str

        storm_name : str

        valid_time : str

        tz : str

        text_lat : float

        text_lon : float

        text_font : int

        text_size : float

        text_width : int

        number_ww : int

        ww : array_like of dict
            Array-like object containing dict of watches/warnings.
        """
        super().__init__(header_struct)
        self.storm_number = storm_number
        self.issue_status = issue_status
        self.basin = basin
        self.advisory_number = advisory_number
        self.storm_name = storm_name
        self.storm_type = storm_type
        self.valid_time = valid_time
        self.timezone = tz
        self.text_lat = text_lat
        self.text_lon = text_lon
        self.text_font = text_font
        self.text_size = text_size
        self.text_width = text_width
        self.number_ww = number_ww
        self.ww = ww


class TropicalCycloneBreakPointElement(TropicalCycloneBase):
    """Tropical cyclone break point element."""

    def __init__(self, header_struct, storm_number, issue_status, basin, advisory_number,
                 storm_name, storm_type, valid_time, tz, forecast_period, line_color,
                 line_width, ww_level, number_points, breakpoints):
        """Create TC break point element.

        Parameters
        ----------
        header_struct : `NamedStruct`

        storm_number : str

        issue_status : str

        basin : int

        advisory_number : str

        storm_name : str

        valid_time : str

        tz : str

        forecast_period : str

        line_color : int

        line_width : int

        ww_level : int

        number_points : int

        breakpoints : array_like of `BreakPointAttribute`
        """
        super().__init__(header_struct, storm_number, issue_status, basin, advisory_number,
                         storm_name, storm_type, valid_time, tz, forecast_period)
        self.line_color = line_color
        self.line_width = line_width
        self.ww_level = TropicalWatchWarningLevel(ww_level)
        self.number_points = number_points
        self.breakpoints = breakpoints


class TropicalCycloneErrorElement(TropicalCycloneBase):
    """Tropical cyclone error cone element."""

    def __init__(self, header_struct, storm_number, issue_status, basin, advisory_number,
                 storm_name, storm_type, valid_time, tz, forecast_period, line_color,
                 line_type, fill_color, fill_type, number_points, lat, lon):
        """Create tropical cyclone error cone element.

        Parameters
        ----------
        header_struct : `NamedStruct`

        storm_number : str

        issue_status : str

        basin : int

        advisory_number : str

        storm_name : str

        valid_time : str

        tz : str

        forecast_period : str

        line_color : int

        line_width : int

        ww_level : int

        number_points : int

        breakpoints : array_like of `BreakPointAttribute`
        """
        super().__init__(header_struct, storm_number, issue_status, basin, advisory_number,
                         storm_name, storm_type, valid_time, tz, forecast_period)
        self.forecast_period = forecast_period
        self.line_color = line_color
        self.line_type = line_type
        self.fill_color = fill_color
        self.fill_type = fill_type
        self.number_points = number_points
        self.lat = lat
        self.lon = lon


class TropicalCycloneTrackElement(TropicalCycloneBase):
    """Tropical cyclone track element."""

    def __init__(self, header_struct, storm_number, issue_status, basin, advisory_number,
                 storm_name, storm_type, valid_time, tz, forecast_period, line_color,
                 line_type, number_points, track_points):
        """Create tropical cyclone track element.

        Parameters
        ----------
        header_struct : `NamedStruct`

        storm_number : str

        issue_status : str

        basin : int

        advisory_number : str

        storm_name : str

        valid_time : str

        tz : str

        forecast_period : str

        line_color : int

        line_width : int

        ww_level : int

        number_points : int

        breakpoints : array_like of `BreakPointAttribute`
        """
        super().__init__(header_struct, storm_number, issue_status, basin, advisory_number,
                         storm_name, storm_type, valid_time, tz, forecast_period)
        self.forecast_period = forecast_period
        self.line_color = line_color
        self.line_type = line_type
        self.number_points = number_points
        self.track_points = track_points


class VolcanoElement(VectorGraphicElement):
    """Volcano element."""

    def __init__(self, header_struct, name, code, size, width, number, location, area,
                 origin_station, vaac, wmo_id, header_number, elevation, year, advisory_number,
                 correction, info_source, additional_source, aviation_color, details, obs_date,
                 obs_time, obs_ash, forecast_6hr, forecast_12hr, forecast_18hr, remarks,
                 next_advisory, forecaster, offset_x, offset_y, lat, lon):
        """Create volcano element.

        Parameters
        ----------
        header_struct : `NamedStruct`

        name : str

        code : float

        size : float

        width : int

        number : str

        location : str

        area : str

        origin_station : str

        vaac : str

        wmo_id : str

        elevation : str

        year : int

        advisory_number : str

        correction : str

        info_source : str

        additional_source : str

        aviation_color : str

        details : str

        obs_date : str

        obs_time : str

        obs_ash : str

        forecat_6hr : str

        forecat_12hr : str

        forecat_18hr : str

        remarks : str

        next_advisory : str

        forecaster : str

        offset_x : int

        offset_y : int

        lat : `numpy.ndarray`

        lon : `numpy.ndarray`
        """
        super().__init__(header_struct)
        self.name = name
        self.code = code
        self.size = size
        self.width = width
        self.number = number
        self.location = location
        self.area = area
        self.origin_station = origin_station
        self.vaac = vaac
        self.wmo_id = wmo_id
        self.header_number = header_number
        self.elevation = elevation
        self.year = year
        self.advisory_number = advisory_number
        self.correction = correction
        self.info_source = info_source
        self.additional_source = additional_source
        self.aviation_color = aviation_color
        self.details = details
        self.obs_date = obs_date
        self.obs_time = obs_time
        self.obs_ash = obs_ash
        self.forecast_6hr = forecast_6hr
        self.forecast_12hr = forecast_12hr
        self.forecast_18hr = forecast_18hr
        self.remarks = remarks
        self.next_advisory = next_advisory
        self.forecaster = forecaster
        self.lat = lat
        self.lon = lon
        self.offset_x = offset_x
        self.offset_y = offset_y


class WatchStatusMessageElement(SpecialLineElement):
    """Watch status message element."""

    def __init__(self, header_struct, number_points, line_type, stroke, direction, size, width,
                 lat, lon):
        """Create watch status message line element.

        Parameters
        ----------
        header_struct : `NamedStruct`

        number_points : int

        line_type : int
            Integer code defining special line type. See Appendix C
            in GEMPAK documentation for details.

        stroke : int
            GEMPAK color code.

        direction : int
            CW (1) or CCW (-1).

        size : float
            Size of special line elements (e.g., dots, crosses, arrow heads, etc.).

        width : int

        lat : `numpy.ndarray`

        lon : `numpy.ndarray`
        """
        super().__init__(header_struct, number_points, line_type, stroke, direction, size,
                         width, lat, lon)


class WindElement(VectorGraphicElement):
    """Wind element."""

    def __init__(self, header_struct, number_wind, width, size, wind_type,
                 head_size, speed, direction, lat, lon):
        """Create wind element.

        Parameters
        ----------
        header_struct : `NamedStruct`

        number_wind : int
            Number of wind vectors.

        width : int
            Vector line width.

        size : float
            Scaling factor for wind vector.

        wind_type : int
            GEMPAK integer code for wind. See WIND parameter
            in GEMPAK GDPLOT documentation.

        head_size : float
            Size of wind element head (barb or arrow).

        speed : float
            Wind speed in knots.

        direction : float
            Wind direction in degrees.

        lat : `numpy.ndarray`

        lon : `numpy.ndarray`
        """
        super().__init__(header_struct)
        self.number_wind = number_wind
        self.width = width
        self.size = size
        self.wind_type = wind_type
        self.head_size = head_size
        self.speed = speed
        self.direction = direction
        self.lat = lat
        self.lon = lon


class WatchBoxElement(VectorGraphicElement):
    """Watch box element."""

    def __init__(self, header_struct, number_points, style, shape,
                 marker_type, marker_size, marker_width, anchor0_station,
                 anchor0_lat, anchor0_lon, anchor0_distance, anchor0_direction,
                 anchor1_station, anchor1_lat, anchor1_lon, anchor1_distance,
                 anchor1_direction, status, number, issue_time, expire_time,
                 watch_type, severity, timezone, max_hail, max_wind, max_tops,
                 mean_storm_direction, mean_storm_speed, states, adjacent_areas,
                 replacing, forecaster, filename, issue_flag, wsm_issue_time,
                 wsm_expire_time, wsm_reference_direction, wsm_recent_from_line,
                 wsm_md_number, wsm_forecaster, number_counties, plot_counties,
                 county_fips, county_lat, county_lon, lat, lon):
        """Create watch box element.

        Parameters
        ----------
        header_struct : `NamedStruct`

        number_points : int

        style : int
            Integer code designating watch-by-county (6) or parallelogram (4).

        shape : int
            PGRAM watch shape. North-South (1), East-West (2), or either side
            of line (3).

        marker_type : int
            Integer code defining the county marker symbols. See Appendix C in
            GEMPAK documentation for details.

        marker_size : float

        marker_width : int

        anchor0_station : str
            Station ID of anchor point 0.

        anchor0_lat : float
            Latitude of anchor point 0.

        anchor0_lon : float
            Longitude of anchor point 0.

        anchor0_distance : int
            Distance (statute miles) from anchor point 0.

        anchor0_direction : str
            Compass direction (16-point) from anchor point 0.

        anchor1_station : str
            Station ID of anchor point 1.

        anchor1_lat : float
            Latitude of anchor point 1.

        anchor1_lon : float
            Longitude of anchor point 1.

        anchor1_distance : int
            Distance (statute miles) from anchor point 1.

        anchor1_direction : str
            Compass direction (16-point) from anchor point 1.

        status : int
            Active (1) or test (0).

        number : int
            Watch number.

        issue_time : str
            Issue time in format `%m/%d/%Y/%H%M/`.

        expire_time : str
            Expiration time in format `%m/%d/%Y/%H%M/`.

        watch_type : int
            Severe Thunderstorm (6) or Tornado (2).

        severity : int
            PDS (1) or normal (0).

        timezone : str
            Primary timezone.

        max_hail : str
            Max hail size in inches.

        max_wind : str
            Max wind gust in knots.

        max_tops : str
            Max storm tops in hundreds of feet (e.g., 500 for 50000 ft.).

        mean_storm_direction : str
            Mean storm direction in degrees.

        mean_storm_speed : str
            Mean storm speed in knots.

        states : str
            States in watch.

        adjacent_areas : str
            Adjacent areas in watch.

        replacing : str
            Watche numbers of watches being replaced.

        forecaster : str
            Issuing forecaster name(s).

        filename : str
            Wactch filename.

        issue_flag : int
            Watch issued (1) or not issued (0).

        wsm_issue_time : str
            Watch status message issue time in format `%d%H%M`.

        wsm_expire_time : str
            Watch status message expiration time in format `%d%H%M`.

        wsm_reference direction : str

        wsm_recent_from_line : str

        wsm_md_number : str
            Associated mesoscale discussion number.

        wsm_forecaster : str
            Watch status message issuing forecaster name(s).

        number_counties : int

        plot_counties : int
            Toggle for plotting couties.

        county_fips : `numpy.ndarray`
            County FIPS codes.

        county_lat : `numpy.ndarray`
            Latitude of county centroid.

        county_lon : `numpy.ndarray`
            Longitude of county centroid.

        lat : `numpy.ndarray`
            Latitude of parallelogram vertices.

        lon : `numpy.ndarray`
            Longitude of parallelogram vertices.
        """
        super().__init__(header_struct)
        self.number_points = number_points
        self.style = style
        self.shape = shape
        self.marker_type = marker_type
        self.marker_size = marker_size
        self.marker_width = marker_width
        self.anchor0_station = anchor0_station
        self.anchor0_lat = anchor0_lat
        self.anchor0_lon = anchor0_lon
        self.anchor0_distance = anchor0_distance
        self.anchor0_direction = anchor0_direction
        self.anchor1_station = anchor1_station
        self.anchor1_lat = anchor1_lat
        self.anchor1_lon = anchor1_lon
        self.anchor1_distance = anchor1_distance
        self.anchor1_direction = anchor1_direction
        self.status = status
        self.number = number
        self.issue_time = issue_time
        self.expire_time = expire_time
        self.watch_type = watch_type
        self.severity = severity
        self.timezone = timezone
        self.max_hail = max_hail
        self.max_wind = max_wind
        self.max_tops = max_tops
        self.mean_storm_speed = mean_storm_speed
        self.mean_storm_direction = mean_storm_direction
        self.states = states
        self.adjacent_areas = adjacent_areas
        self.replacing = replacing
        self.forecaster = forecaster
        self.filename = filename
        self.issue_flag = issue_flag
        self.wsm_issue_time = wsm_issue_time
        self.wsm_expire_time = wsm_expire_time
        self.wsm_reference_direction = wsm_reference_direction
        self.wsm_recent_from_line = wsm_recent_from_line
        self.wsm_md_number = wsm_md_number
        self.wsm_forecaster = wsm_forecaster
        self.number_counties = number_counties
        self.plot_counties = plot_counties
        self.county_fips = county_fips
        self.county_lat = county_lat
        self.county_lon = county_lon
        self.lat = lat
        self.lon = lon

        self._set_properties()

    def _set_properties(self):
        """Set properties using header and data."""
        self.marker_name = MarkerType(self.marker_type).name


class Group:
    """GEMPAK VGF group."""

    def __init__(self):
        self._number_elements = 0
        self._elements = []
        self._group_type = None
        self._group_number = None

    def __iter__(self):
        """Return iterator of group elements."""
        yield from self._elements

    def __repr__(self):
        """Return repr(self)."""
        return (f'{type(self).__qualname__}'
                f'[{self.group_type}, {self.group_number}]')

    @property
    def elements(self):
        """Get elements."""
        return self._elements

    @property
    def group_size(self):
        """Get numerber of elements in group."""
        return self._number_elements

    @property
    def group_type(self):
        """Get group type."""
        return self._group_type

    @property
    def group_number(self):
        """Get group number."""
        return self._group_number

    def add_element(self, element):
        """Add element to group.

        Parameters
        ----------
        element : subclass of `mdgpu.io.vgf.VectorGraphicElement`
        """
        if (self._group_type != element.group_type
           and self._group_type is None):
            self._group_type = element.group_type
        elif self._group_type == element.group_type:
            pass
        else:
            raise ValueError('Group cannot have multiple types.')

        if (self._group_number != element.group_number
           and self._group_number is None):
            self._group_number = element.group_number
        elif self._group_number == element.group_number:
            pass
        else:
            raise ValueError('Cannot have multiple groups.')

        self._elements.append(element)
        self._number_elements += 1


class NumpyEncoder(json.JSONEncoder):
    """JSONEncoder class extension for numpy arrays."""

    def default(self, obj):
        """`NumpyEncoder` default serializer."""
        if isinstance(obj, np.ndarray):
            return [float(f'{x:6.2f}') for x in obj]
        return json.JSONEncoder.default(self, obj)


class VectorGraphicFile:
    """GEMPAK Vector Graphics Format decoder class."""

    def __init__(self, file):
        """Read and decode a GEMPAK Vector Graphic file.

        Parameters
        ----------
        file : str or `pathlib.Path`
        """
        with contextlib.closing(open(file, 'rb')) as fobj:  # noqa: SIM115
            self._buffer = IOBuffer.fromfile(fobj)

        if sys.byteorder == 'little':
            self.prefmt = '>'
        else:
            self.prefmt = ''

        self._elements = []
        self._groups = []

        self._decode_elements()

    @property
    def groups(self):
        """Get present group types."""
        return self._groups

    @property
    def elements(self):
        """Get elements."""
        return self._elements

    @property
    def header(self):
        """Get file header."""
        return self._header

    @property
    def bounds(self):
        """Get bounding box of all elements."""
        xmin = 9999
        xmax = -9999
        ymin = 9999
        ymax = -9999

        for element in self.elements:
            if 0 in [element.max_lat, element.max_lon,
                     element.min_lat, element.min_lon]:
                continue
            else:
                if element.max_lat > ymax:
                    ymax = element.max_lat
                if element.max_lon > xmax:
                    xmax = element.max_lon
                if element.min_lat < ymin:
                    ymin = element.min_lat
                if element.min_lon < xmin:
                    xmin = element.min_lon

        return ymin, xmin, ymax, xmax

    @property
    def has_fronts(self):
        """Check for front elements."""
        return bool(self.filter_elements(vg_class=VGClass.fronts.value))

    @property
    def has_text(self):
        """Check for text elements."""
        return bool(self.filter_elements(vg_class=VGClass.text.value,
                                         vg_type=VGType.text.value))

    @property
    def has_special_text(self):
        """Check for special text elements."""
        return bool(self.filter_elements(vg_class=VGClass.text.value,
                                         vg_type=VGType.special_text.value))

    @property
    def has_symbols(self):
        """Check for symbols elements."""
        return bool(self.filter_elements(vg_class=VGClass.symbols.value))

    @property
    def has_special_lines(self):
        """Check for special lines elements."""
        return bool(self.filter_elements(vg_class=VGClass.lines.value,
                                         vg_type=VGType.special_line.value))

    @property
    def has_lines(self):
        """Check for lines elements."""
        return bool(self.filter_elements(vg_class=VGClass.lines.value,
                                         vg_type=VGType.line.value))

    @property
    def has_winds(self):
        """Check for wind elements."""
        return bool(self.filter_elements(vg_class=VGClass.winds.value))

    @property
    def has_tracks(self):
        """Check for track elements."""
        return bool(self.filter_elements(vg_class=VGClass.tracks.value))

    @property
    def has_watch_box(self):
        """Check for watch box elements."""
        return bool(self.filter_elements(vg_class=VGClass.watches.value))

    @property
    def has_sigmet(self):
        """Check for SIGMET elements."""
        return bool(self.filter_elements(vg_class=VGClass.sigmets.value))

    @property
    def has_met(self):
        """Check for MET elements."""
        return bool(self.filter_elements(vg_class=VGClass.met.value))

    def get_fronts(self):
        """Extract front elements.

        Returns
        -------
        List of `mdgpu.io.vgf.FrontElement`.
        """
        if self.has_fronts:
            return self.filter_elements(vg_class=VGClass.fronts.value)

        return None

    def get_text(self):
        """Extract text elements.

        Returns
        -------
        List of `mdgpu.io.vgf.TextElement`.
        """
        if self.has_text:
            return self.filter_elements(vg_class=VGClass.text.value)

        return None

    def get_special_text(self):
        """Extract special text elements.

        Returns
        -------
        List of `mdgpu.io.vgf.SpecialTextElement`.
        """
        if self.has_special_text:
            return self.filter_elements(vg_class=VGClass.text.value,
                                        vg_type=VGType.special_text.value)

        return None

    def get_symbols(self):
        """Extract symbol elements.

        Returns
        -------
        List of `mdgpu.io.vgf.SymbolElement`.
        """
        if self.has_symbols:
            return self.filter_elements(vg_class=VGClass.symbols.value)

        return None

    def get_special_lines(self):
        """Extract special line elements.

        Returns
        -------
        List of `mdgpu.io.vgf.SpecialLineElement`.

        Notes
        -----
        This will exclude the MD area line.
        """
        if self.has_special_lines:
            special_lines = self.filter_elements(vg_class=VGClass.lines.value,
                                                 vg_type=VGType.special_line.value)
            return [x for x in special_lines
                    if x.line_type != SpecialLineType.scallop.value]

        return None

    def get_lines(self):
        """Extract line elements.

        Returns
        -------
        List of `mdgpu.io.vgf.LineElement`.
        """
        if self.has_lines:
            return self.filter_elements(vg_class=VGClass.lines.value,
                                        vg_type=VGType.line.value)

        return None

    def get_tracks(self):
        """Extract track elements.

        Returns
        -------
        List of `mdgpu.io.vgf.TrackElement`.
        """
        if self.has_tracks:
            return self.filter_elements(vg_class=VGClass.tracks.value)

        return None

    def get_winds(self):
        """Extract wind elements.

        Returns
        -------
        List of `mdgpu.io.vgf.WindElement`.
        """
        if self.has_winds:
            return self.filter_elements(vg_class=VGClass.winds.value)

        return None

    def get_watch_box(self):
        """Extract watch box elements.

        Returns
        -------
        List of `mdgpu.io.vgf.WatchBoxElement`.
        """
        if self.has_watch_box:
            return self.filter_elements(vg_class=VGClass.watches.value)

        return None

    def get_sigmet(self):
        """Extract SIGMET elements.

        Returns
        -------
        List of `mdgpu.io.vgf.SigmetElement`.
        """
        if self.has_sigmet:
            return self.filter_elements(vg_class=VGClass.sigmets.value)

        return None

    def get_met(self):
        """Extract MET elements.

        Returns
        -------
        List of MET elements.
        """
        if self.has_sigmet:
            return self.filter_elements(vg_class=VGClass.met.value)

        return None

    def get_group(self, group_type):
        """Get elements that are part of a group type.

        Parameters
        ----------
        group_type : int

        Returns
        -------
        List of `mdgpu.io.vgf.VectorGraphicElement` subclasses in a group.
        """
        type_temp = []
        numbers = []
        grouped = []

        for element in self._elements:
            if element.group_type == group_type:
                type_temp.append(element)

        for element in type_temp:
            if element.group_number not in numbers:
                numbers.append(element.group_number)

        for gn in sorted(numbers):
            g = Group()
            for element in type_temp:
                if element.group_number == gn:
                    g.add_element(element)
            if g.group_size:
                grouped.append(g)

        if not grouped:
            grouped = None

        return grouped

    def filter_elements(self, vg_class=None, vg_type=None, operation='and'):
        """Filter elements by class and type.

        Parameters
        ----------
        vg_class : int
            Integer code for vector graphic classes.

        vg_type : int
            Integer code for vector graphic types.

        operation : str
            How queries should be handled when vg_class and vg_type are both
            used. `and` is logical and and `or` uses logical or. The default
            is `and`.

        Returns
        -------
        List of `mdgpu.io.vgf.VectorGraphicElement` subclasses.
        """
        if operation not in ['and', 'or']:
            raise ValueError(f'Illegal operation `{operation}`.')

        filtered = []
        if vg_class is not None and vg_type is None:
            for e in self.elements:
                if e.vg_class == vg_class:
                    filtered.append(e)
        elif vg_class is None and vg_type is not None:
            for e in self.elements:
                if e.vg_type == vg_type:
                    filtered.append(e)
        elif vg_class is not None and vg_type is not None:
            if operation == 'and':
                for e in self.elements:
                    if e.vg_class == vg_class and e.vg_type == vg_type:
                        filtered.append(e)
            elif operation == 'or':
                for e in self.elements:
                    if e.vg_class == vg_class or e.vg_type == vg_type:
                        filtered.append(e)

        if not filtered:
            filtered = None

        return filtered

    def to_json(self, **kwargs):
        """Convert VGF elements to JSON.

        Parameters
        ----------
        kwargs : Keyword arguments to pass to `json.dumps`.

        Returns
        -------
        string
            JSON string representation of VGF elements.
        """
        serialized = [e.__dict__ for e in self._elements]

        return json.dumps(serialized,
                          cls=kwargs.get('cls', NumpyEncoder),
                          **kwargs)

    def _decode_elements(self):
        """Decode elements of a VGF."""
        while not self._buffer.at_end():
            header_struct = self._read_header()
            rec_size = header_struct.record_size
            vg_type = header_struct.vg_type
            vg_class = header_struct.vg_class
            data_size = rec_size - VGF_HEADER_SIZE

            group_info = header_struct.group_type

            # Ignores the file header group
            if (group_info not in self._groups
               and group_info
               and vg_class != VGClass.header.value):
                self._groups.append(group_info)

            if vg_class == VGClass.header.value and vg_type == VGType.file_header.value:
                version_size = 128
                version = self._decode_strip_null(self._buffer.read(version_size))
                notes_size = data_size - version_size
                notes = self._decode_strip_null(self._buffer.read(notes_size))
                self._header = FileHeaderElement(header_struct, version, notes)
            elif vg_class == VGClass.fronts.value:
                front_info = [
                    ('number_points', 'i'), ('front_code', 'i'), ('pip_size', 'i'),
                    ('pip_stroke', 'i'), ('pip_direction', 'i'), ('width', 'i'),
                    ('label', '4s', self._decode_strip_null)
                ]
                front = self._buffer.read_struct(
                    NamedStruct(front_info, self.prefmt, 'FrontInfo')
                )

                lat, lon = self._get_latlon(front.number_points)

                self._elements.append(
                    FrontElement(header_struct, front.number_points, front.front_code,
                                 front.pip_size, front.pip_stroke, front.pip_direction,
                                 front.width, front.label, lat, lon)
                )
            elif vg_class == VGClass.symbols.value:
                symbol_info = [
                    ('number_symbols', 'i'), ('width', 'i'),
                    ('symbol_size', 'f', self._round_one), ('symbol_type', 'i'),
                    ('symbol_code', 'f', int), ('lat', 'f', self._round_two),
                    ('lon', 'f', self._round_two), ('offset_x', 'i'), ('offset_y', 'i')
                ]
                symbol = self._buffer.read_struct(
                    NamedStruct(symbol_info, self.prefmt, 'SymbolInfo')
                )

                self._elements.append(
                    SymbolElement(header_struct, symbol.number_symbols, symbol.width,
                                  symbol.symbol_size, symbol.symbol_type, symbol.symbol_code,
                                  symbol.offset_x, symbol.offset_y, symbol.lat, symbol.lon)
                )
            elif vg_class == VGClass.circle.value:
                if vg_type == VGType.circle.value:
                    line_info = [
                        ('number_points', 'i'), ('line_type', 'i'),
                        ('line_type_hardware', 'i'), ('width', 'i'),
                        ('line_width_hardware', 'i')
                    ]
                    line = self._buffer.read_struct(
                        NamedStruct(line_info, self.prefmt, 'LineInfo')
                    )

                    # This is CircData
                    lat, lon = self._get_latlon(line.number_points)

                    self._elements.append(
                        CircleElement(header_struct, line.number_points, line.line_type,
                                      line.line_type_hardware, line.width,
                                      line.line_width_hardware, lat, lon)
                    )
            elif vg_class == VGClass.lines.value:
                if vg_type == VGType.line.value:
                    line_info = [
                        ('number_points', 'i'), ('line_type', 'i'),
                        ('line_type_hardware', 'i'), ('width', 'i'),
                        ('line_width_hardware', 'i')
                    ]
                    line = self._buffer.read_struct(
                        NamedStruct(line_info, self.prefmt, 'LineInfo')
                    )

                    lat, lon = self._get_latlon(line.number_points)
                    if isinstance(lat, int) and lat == -9999:
                        continue

                    if header_struct.closed and line.number_points > 2:
                        lon, lat = self.close_coordinates(lon, lat)

                    self._elements.append(
                        LineElement(header_struct, line.number_points, line.line_type,
                                    line.line_type_hardware, line.width,
                                    line.line_width_hardware, lat, lon)
                    )
                elif vg_type == VGType.special_line.value:
                    special_line_info = [
                        ('number_points', 'i'), ('line_type', 'i'), ('stroke', 'i'),
                        ('direction', 'i'), ('line_size', 'f', self._round_one), ('width', 'i')
                    ]
                    special_line = self._buffer.read_struct(
                        NamedStruct(special_line_info, self.prefmt, 'SpecialLineInfo')
                    )

                    lat, lon = self._get_latlon(special_line.number_points)
                    if isinstance(lat, int) and lat == -9999:
                        continue

                    if header_struct.closed and special_line.number_points > 2:
                        lon, lat = self.close_coordinates(lon, lat)

                    if special_line.direction == -1:
                        lon, lat = self.flip_coordinates(lon, lat)

                    self._elements.append(
                        SpecialLineElement(header_struct, special_line.number_points,
                                           special_line.line_type, special_line.stroke,
                                           special_line.direction, special_line.line_size,
                                           special_line.width, lat, lon)
                    )
                else:
                    raise NotImplementedError(f'Line type `{vg_type}` is not implemented.')
            elif vg_class == VGClass.lists.value:
                list_info = [
                    ('list_type', 'i'), ('marker_type', 'i'),
                    ('marker_size', 'f', self._round_one),
                    ('marker_width', 'i'), ('number_items', 'i')
                ]
                list_struct = NamedStruct(list_info, self.prefmt, 'ListInfo')
                list_meta = self._buffer.read_struct(list_struct)

                list_items = [
                    self._buffer.read_ascii(LIST_MEMBER_SIZE).replace('\x00', '')
                    for _n in range(list_meta.number_items)
                ]
                list_item_blank_size = (MAX_POINTS - list_meta.number_items) * LIST_MEMBER_SIZE
                self._buffer.skip(list_item_blank_size)

                coord_blank_size = 4 * (MAX_POINTS - list_meta.number_items)
                lat = self._buffer.read_array(list_meta.number_items, f'{self.prefmt}f')
                self._buffer.skip(coord_blank_size)
                lon = self._buffer.read_array(list_meta.number_items, f'{self.prefmt}f')
                self._buffer.skip(coord_blank_size)

                self._elements.append(
                    ListElement(header_struct, list_meta.list_type, list_meta.marker_type,
                                list_meta.marker_size, list_meta.marker_width,
                                list_meta.number_items, list_items, lat, lon)
                )
            elif vg_class == VGClass.text.value:
                if vg_type == VGType.text.value or vg_type == VGType.justified_text.value:
                    text_info = [
                        ('rotation', 'f', self._round_one),
                        ('text_size', 'f', self._round_one), ('font', 'i'),
                        ('text_flag', 'i'), ('width', 'i'), ('align', 'i'),
                        ('lat', 'f', self._round_two), ('lon', 'f', self._round_two),
                        ('offset_x', 'i'), ('offset_y', 'i')
                    ]
                    text_struct = NamedStruct(text_info, self.prefmt, 'TextInfo')
                    text = self._buffer.read_struct(text_struct)

                    text_length = rec_size - VGF_HEADER_SIZE - text_struct.size
                    text_string = self._buffer.read_ascii(text_length)
                    clean_text = text_string.replace('$$', '\n').replace('\x00', '').strip()

                    self._elements.append(
                        TextElement(header_struct, text.rotation, text.text_size, text.font,
                                    text.text_flag, text.width, text.align, text.lat,
                                    text.lon, text.offset_x, text.offset_y, clean_text)
                    )
                elif vg_type == VGType.special_text.value:
                    special_text_info = [
                        ('rotation', 'f', self._round_one),
                        ('text_size', 'f', self._round_one), ('text_type', 'i'),
                        ('turbulence_symbol', 'i'), ('font', 'i'), ('text_flag', 'i'),
                        ('width', 'i'), ('text_color', 'i'), ('line_color', 'i'),
                        ('fill_color', 'i'), ('align', 'i'), ('lat', 'f', self._round_two),
                        ('lon', 'f', self._round_two), ('offset_x', 'i'), ('offset_y', 'i')
                    ]
                    text_struct = NamedStruct(
                        special_text_info, self.prefmt, 'SpecialTextInfo'
                    )
                    text = self._buffer.read_struct(text_struct)

                    text_length = rec_size - VGF_HEADER_SIZE - text_struct.size
                    text_string = self._buffer.read_ascii(text_length)
                    clean_text = text_string.replace('$$', '\n').replace('\x00', '').strip()

                    self._elements.append(
                        SpecialTextElement(header_struct, text.rotation, text.text_size,
                                           text.text_type, text.turbulence_symbol, text.font,
                                           text.text_flag, text.width, text.text_color,
                                           text.line_color, text.fill_color, text.align,
                                           text.lat, text.lon, text.offset_x, text.offset_y,
                                           clean_text)
                    )
                else:
                    raise NotImplementedError(f'Text type `{vg_type}` is not implemented.')
            elif vg_class == VGClass.tracks.value:
                track_info = [
                    ('track_type', 'i'), ('total_points', 'i'), ('initial_points', 'i'),
                    ('initial_line_type', 'i'), ('extrapolated_line_type', 'i'),
                    ('initial_mark_type', 'i'), ('extrapolated_mark_type', 'i'),
                    ('line_width', 'i'), ('speed', 'f', self._round_two),
                    ('direction', 'f', self._round_two), ('increment', 'i'),
                    ('skip', 'i'), ('font', 'i'), ('font_flag', 'i')
                ]
                track = self._buffer.read_struct(
                    NamedStruct(track_info, self.prefmt, 'TrackInfo')
                )

                # sztext seems to always be little endian. It is not swapped like other
                # elements in the storm track struct. See storm track section of cvgswap.c.
                track_font_size = round(self._buffer.read_binary(1, '<f')[0], 1)

                times = [
                    self._buffer.read_ascii(TRACK_DT_SIZE).replace('\x00', '')
                    for _n in range(track.total_points)
                ]

                blank_times_size = (MAX_TRACKS - track.total_points) * TRACK_DT_SIZE
                self._buffer.skip(blank_times_size)

                lat, lon = self._get_latlon(track.total_points)
                blank_latlon_size = 8 * (MAX_TRACKS - track.total_points)
                self._buffer.skip(blank_latlon_size)

                self._elements.append(
                    TrackElement(header_struct, track.track_type, track.total_points,
                                 track.initial_points, track.initial_line_type,
                                 track.extrapolated_line_type, track.initial_mark_type,
                                 track.extrapolated_mark_type, track.line_width, track.speed,
                                 track.direction, track.increment, track.skip, track.font,
                                 track.font_flag, track_font_size, times, lat, lon)
                )
            elif vg_class == VGClass.sigmets.value:
                if vg_type in [VGType.convective_outlook.value, VGType.convective_sigmet.value,
                               VGType.nonconvective_sigmet.value, VGType.airmet.value,
                               VGType.international_sigmet.value]:
                    sigmet_info = [
                        ('subtype', 'i'), ('number_points', 'i'), ('line_type', 'i'),
                        ('line_width', 'i'), ('side_of_line', 'i'),
                        ('area', '8s', self._decode_strip_null),
                        ('flight_info_region', '32s', self._decode_strip_null),
                        ('status', 'i'), ('distance', 'f'),
                        ('message_id', '12s', self._decode_strip_null),
                        ('sequence_number', 'i'),
                        ('start_time', '20s', self._decode_strip_null),
                        ('end_time', '20s', self._decode_strip_null),
                        ('remarks', '80s', self._decode_strip_null), ('sonic', 'i'),
                        ('phenomena', '32s', self._decode_strip_null),
                        ('phenomena2', '32s', self._decode_strip_null),
                        ('phenomena_name', '36s', self._decode_strip_null),
                        ('phenomena_lat', '8s', self._decode_strip_null),
                        ('phenomena_lon', '8s', self._decode_strip_null), ('pressure', 'i'),
                        ('max_wind', 'i'), ('free_text', '256s', self._decode_strip_null),
                        ('trend', '8s', self._decode_strip_null),
                        ('movement', '8s', self._decode_strip_null),
                        ('type_indicator', 'i'), ('type_time', '20s', self._decode_strip_null),
                        ('flight_level', 'i'), ('speed', 'i'),
                        ('direction', '4s', self._decode_strip_null),
                        ('tops', '80s', self._decode_strip_null),
                        ('forecaster', '16s', self._decode_strip_null)
                    ]

                    sigmet = self._buffer.read_struct(
                        NamedStruct(sigmet_info, self.prefmt, 'SigmetInfo')
                    )

                    lat = self._buffer.read_array(sigmet.number_points, f'{self.prefmt}f')
                    lon = self._buffer.read_array(sigmet.number_points, f'{self.prefmt}f')

                    self._elements.append(
                        SigmetElement(header_struct, sigmet.subtype, sigmet.number_points,
                                      sigmet.line_type, sigmet.line_width, sigmet.side_of_line,
                                      sigmet.area, sigmet.flight_info_region, sigmet.area,
                                      sigmet.distance, sigmet.message_id,
                                      sigmet.sequence_number, sigmet.start_time,
                                      sigmet.end_time, sigmet.remarks, sigmet.sonic,
                                      sigmet.phenomena, sigmet.phenomena2,
                                      sigmet.phenomena_name, sigmet.phenomena_lat,
                                      sigmet.phenomena_lon, sigmet.pressure, sigmet.max_wind,
                                      sigmet.free_text, sigmet.trend, sigmet.movement,
                                      sigmet.type_indicator, sigmet.type_time,
                                      sigmet.flight_level, sigmet.speed, sigmet.direction,
                                      sigmet.tops, sigmet.forecaster, lat, lon
                        )
                    )
                elif vg_type == VGType.ccf.value:
                    ccf_info = [
                        ('subtype', 'i'), ('number_points', 'i'), ('coverage', 'i'),
                        ('storm_tops', 'i'), ('probability', 'i'), ('growth', 'i'),
                        ('speed', 'f', self._round_two), ('direction', 'f', self._round_two),

                    ]

                    # See cvgswap.c in GEMPAK source. These are not swapped.
                    ccf_info_noswap = [
                        ('text_lat', 'f', self._round_two), ('text_lon', 'f', self._round_two),
                        ('arrow_lat', 'f', self._round_two),
                        ('arrow_lon', 'f', self._round_two), ('high_fill', 'i'),
                        ('med_fill', 'i'), ('low_fill', 'i'), ('line_type', 'i'),
                        ('arrow_size', 'f', self._round_one)
                    ]

                    ccf = self._buffer.read_struct(
                        NamedStruct(ccf_info, self.prefmt, 'CCFInfo')
                    )

                    ccf_noswap = self._buffer.read_struct(
                        NamedStruct(ccf_info_noswap, '', 'CCFInfoRaw')
                    )

                    # Because of how CCF elements are handled, swapping does not occur on
                    # the special text element in the struct. We handle that manually here
                    # for the few attributes it affects. See cvgswap.c in GEMPAK source.
                    ccf_text_info = [
                        ('rotation', 'f', self._round_one),
                        ('text_size', 'f', self._round_one), ('text_type', 'i'),
                        ('turbulence_symbol', 'i'), ('font', 'i', self._swap32),
                        ('text_flag', 'i', self._swap32), ('width', 'i', self._swap32),
                        ('text_color', 'i'), ('line_color', 'i'), ('fill_color', 'i'),
                        ('align', 'i'), ('lat', 'f', self._round_two),
                        ('lon', 'f', self._round_two), ('offset_x', 'i'), ('offset_y', 'i'),
                        ('text', '255s', self._decode_strip_null),
                        (None, '1x')  # skip struct alignment padding byte
                    ]

                    ccf_text = self._buffer.read_struct(
                        NamedStruct(ccf_text_info, self.prefmt, 'CCFTextInfo')
                    )

                    text_layout = self._buffer.read_ascii(256).replace('\x00', '')

                    lat, lon = self._get_latlon(ccf.number_points)

                    self._elements.append(
                        CollaborativeConvectiveForecastElement(
                            header_struct, ccf.subtype, ccf.number_points, ccf.coverage,
                            ccf.storm_tops, ccf.probability, ccf.growth, ccf.speed,
                            ccf.direction, ccf_noswap.text_lat, ccf_noswap.text_lon,
                            ccf_noswap.arrow_lat, ccf_noswap.arrow_lon, ccf_noswap.high_fill,
                            ccf_noswap.med_fill, ccf_noswap.low_fill, ccf_noswap.line_type,
                            ccf_noswap.arrow_size, ccf_text.rotation, ccf_text.text_size,
                            ccf_text.text_type, ccf_text.turbulence_symbol, ccf_text.font,
                            ccf_text.text_flag, ccf_text.width, ccf_text.fill_color,
                            ccf_text.align, ccf_text.offset_x, ccf_text.offset_y,
                            ccf_text.text, text_layout, lat, lon
                        )
                    )
                    self._buffer.skip((MAX_SIGMET - ccf.number_points) * 8)
                elif vg_type == VGType.volcano.value:
                    volcano_info = [
                        ('name', '64s', self._decode_strip_null),
                        ('code', 'f', self._round_one), ('size', 'f', self._round_one),
                        ('width', 'i'), ('number', '17s', self._decode_strip_null),
                        ('location', '17s', self._decode_strip_null),
                        ('area', '33s', self._decode_strip_null),
                        ('origin_station', '17s', self._decode_strip_null),
                        ('vaac', '33s', self._decode_strip_null),
                        ('wmo_id', '8s', self._decode_strip_null),
                        ('header_number', '9s', self._decode_strip_null),
                        ('elevation', '9s', self._decode_strip_null),
                        ('year', '9s', self._decode_strip_null),
                        ('advisory_number', '9s', self._decode_strip_null),
                        ('correction', '4s', self._decode_strip_null),
                        ('info_source', '256s', self._decode_strip_null),
                        ('additional_source', '256s', self._decode_strip_null),
                        ('aviation_color', '16s', self._decode_strip_null),
                        ('details', '256s', self._decode_strip_null),
                        ('obs_date', '16s', self._decode_strip_null),
                        ('obs_time', '16s', self._decode_strip_null),
                        ('obs_ash', '1024s', self._decode_strip_null),
                        ('forecast_6hr', '1024s', self._decode_strip_null),
                        ('forecast_12hr', '1024s', self._decode_strip_null),
                        ('forecast_18hr', '1024s', self._decode_strip_null),
                        ('remarks', '512s', self._decode_strip_null),
                        ('next_advisory', '128s', self._decode_strip_null),
                        ('forecaster', '64s', self._decode_strip_null),
                        (None, '3x'),  # skip struct alignment padding bytes
                        ('lat', 'f', self._round_two), ('lon', 'f', self._round_two),
                        ('offset_x', 'i'), ('offset_y', 'i')
                    ]

                    volcano = self._buffer.read_struct(
                        NamedStruct(volcano_info, self.prefmt, 'VolcanoInfo')
                    )

                    self._elements.append(
                        VolcanoElement(header_struct, volcano.name, volcano.code, volcano.size,
                                       volcano.width, volcano.number, volcano.location,
                                       volcano.area, volcano.origin_station, volcano.vaac,
                                       volcano.wmo_id, volcano.header_number,
                                       volcano.elevation, volcano.year,
                                       volcano.advisory_number, volcano.correction,
                                       volcano.info_source, volcano.additional_source,
                                       volcano.aviation_color, volcano.details,
                                       volcano.obs_date, volcano.obs_time, volcano.obs_ash,
                                       volcano.forecast_6hr, volcano.forecast_12hr,
                                       volcano.forecast_18hr, volcano.remarks,
                                       volcano.next_advisory, volcano.forecaster,
                                       volcano.offset_x, volcano.offset_y, volcano.lat,
                                       volcano.lon)
                    )
                elif vg_type == VGType.ash_cloud.value:
                    ash_info = [
                        ('subtype', 'i'), ('number_points', 'i'),
                        ('distance', 'f', self._round_two),
                        ('forecast_hour', 'i'), ('line_type', 'i'),
                        ('line_width', 'i'), ('side_of_line', 'i'),
                        ('speed', 'f', self._round_two),
                        ('speeds', '16s', self._decode_strip_null),
                        ('direction', '4s', self._decode_strip_null),
                        ('flight_level_1', '16s', self._decode_strip_null),
                        ('flight_level_2', '16s', self._decode_strip_null)
                    ]

                    ash = self._buffer.read_struct(
                        NamedStruct(ash_info, self.prefmt, 'AshInfo')
                    )

                    special_text_info = [
                        ('rotation', 'f', self._round_one),
                        ('text_size', 'f', self._round_one), ('text_type', 'i'),
                        ('turbulence_symbol', 'i'), ('font', 'i'), ('text_flag', 'i'),
                        ('width', 'i'), ('text_color', 'i'), ('line_color', 'i'),
                        ('fill_color', 'i'), ('align', 'i'), ('lat', 'f', self._round_two),
                        ('lon', 'f', self._round_two), ('offset_x', 'i'), ('offset_y', 'i')
                    ]

                    text = self._buffer.read_struct(
                        NamedStruct(special_text_info, self.prefmt, 'SpecialTextInfo')
                    )

                    text_string = self._buffer.read_ascii(255).replace('\x00', '')
                    self._buffer.skip(1)  # skip byte for struct alignment

                    lat, lon = self._get_latlon(ash.number_points)
                    self. _buffer.skip((MAX_ASH - ash.number_points) * 8)

                    self._elements.append(
                        AshCloudElement(header_struct, ash.subtype, ash.number_points,
                                        ash.distance, ash.forecast_hour, ash.line_type,
                                        ash.line_width, ash.side_of_line, ash.speed,
                                        ash.speeds, ash.direction, ash.flight_level_1,
                                        ash.flight_level_2, text.rotation, text.text_size,
                                        text.text_type, text.turbulence_symbol, text.font,
                                        text.text_flag, text.width, text.text_color,
                                        text.line_color, text.fill_color, text.align,
                                        text.lat, text.lon, text.offset_x, text.offset_y,
                                        text_string, lat, lon)
                    )
                else:
                    raise NotImplementedError(f'SIGMET type `{vg_type}` not implemented.')
            elif vg_class == VGClass.met.value:
                if vg_type == VGType.gfa.value:
                    gfa_info = [
                        ('number_blocks', 'i'), ('number_points', 'i')
                    ]

                    gfa = self._buffer.read_struct(
                        NamedStruct(gfa_info, self.prefmt, 'GFAInfo')
                    )

                    blocks = []
                    for _ in range(gfa.number_blocks):
                        blocks.append(
                            self._decode_strip_null(self._buffer.read_binary(1024, 's')[0])
                        )

                    lat, lon = self._get_latlon(gfa.number_points)

                    self._elements.append(
                        GraphicalForecastAreaElement(header_struct, gfa.number_blocks,
                                                     gfa.number_points, blocks, lat, lon)
                    )
                elif vg_type == VGType.jet.value:
                    jet_line_info = [
                        ('line_color', 'i'), ('number_points', 'i'), ('line_type', 'i'),
                        ('stroke', 'i'), ('direction', 'i'), ('size', 'f', self._round_one),
                        ('width', 'i')
                    ]
                    jet_line = self._buffer.read_struct(
                        NamedStruct(jet_line_info, self.prefmt, 'JetLineInfo')
                    )
                    jet_line_lat, jet_line_lon = self._get_latlon(jet_line.number_points)
                    line_attr = LineAttribute(jet_line.line_color, jet_line.number_points,
                                              jet_line.line_type, jet_line.stroke,
                                              jet_line.direction, jet_line.size,
                                              jet_line.width, jet_line_lat, jet_line_lon)

                    self._buffer.skip((MAX_POINTS - jet_line.number_points) * 8)

                    number_barbs = self._buffer.read_int(4, 'big', False)

                    jet_barb_info = [
                        ('wind_color', 'i'), ('number_wind', 'i'), ('width', 'i'),
                        ('size', 'f', self._round_one), ('wind_type', 'i'),
                        ('head_size', 'f', self._round_one), ('speed', 'f', self._round_two),
                        ('direction', 'f', self._round_two), ('lat', 'f', self._round_two),
                        ('lon', 'f', self._round_two), ('flight_level_color', 'i'),
                        ('text_rotation', 'f', self._round_one),
                        ('text_size', 'f', self._round_one), ('text_type', 'i'),
                        ('turbulence_symbol', 'i'), ('font', 'i'), ('text_flag', 'i'),
                        ('text_width', 'i'), ('text_color', 'i'), ('line_color', 'i'),
                        ('fill_color', 'i'), ('align', 'i'),
                        ('text_lat', 'f', self._round_two), ('text_lon', 'f', self._round_two),
                        ('offset_x', 'i'), ('offset_y', 'i'),
                        ('text', '255s', self._decode_strip_null),
                        (None, '1x')  # skip struct alignment padding byte
                    ]

                    barbs = []
                    barb_struct = NamedStruct(jet_barb_info, self.prefmt, 'JetBarbInfo')
                    for _ in range(number_barbs):
                        jet_barb = self._buffer.read_struct(barb_struct)

                        barbs.append(
                            BarbAttribute(jet_barb.wind_color, jet_barb.number_wind,
                                        jet_barb.width, jet_barb.size,
                                        jet_barb.wind_type, jet_barb.head_size,
                                        jet_barb.speed, jet_barb.direction,
                                        jet_barb.lat, jet_barb.lon,
                                        jet_barb.flight_level_color,
                                        jet_barb.text_rotation, jet_barb.text_size,
                                        jet_barb.text_type, jet_barb.turbulence_symbol,
                                        jet_barb.font, jet_barb.text_flag,
                                        jet_barb.text_width, jet_barb.text_color,
                                        jet_barb.line_color, jet_barb.fill_color,
                                        jet_barb.align, jet_barb.text_lat,
                                        jet_barb.text_lon, jet_barb.offset_x,
                                        jet_barb.offset_y, jet_barb.text)
                        )

                    self._buffer.skip((MAX_JET_POINTS - number_barbs) * barb_struct.size)

                    number_hash = self._buffer.read_int(4, 'big', False)

                    jet_hash_info = [
                        ('wind_color', 'i'), ('number_wind', 'i'), ('width', 'i'),
                        ('size', 'f', self._round_one), ('wind_type', 'i'),
                        ('head_size', 'f', self._round_one), ('speed', 'f', self._round_two),
                        ('direction', 'f', self._round_two), ('lat', 'f', self._round_two),
                        ('lon', 'f', self._round_two)
                    ]

                    hashes = []
                    hash_struct = NamedStruct(jet_hash_info, self.prefmt, 'JetHashInfo')
                    for _ in range(number_hash):
                        jet_hash = self._buffer.read_struct(hash_struct)

                        hashes.append(
                            HashAttribute(jet_hash.wind_color, jet_hash.number_wind,
                                        jet_hash.width, jet_hash.size, jet_hash.wind_type,
                                        jet_hash.head_size, jet_hash.speed,
                                        jet_hash.direction, jet_hash.lat, jet_hash.lon)
                        )

                    self._buffer.skip((MAX_JET_POINTS - number_hash) * hash_struct.size)

                    self._elements.append(
                        JetElement(header_struct, line_attr, number_barbs, barbs,
                                   number_hash, hashes)
                    )
                elif vg_type == VGType.tca.value:
                    tca_string_length = header_struct.record_size - VGF_HEADER_SIZE
                    tca_string = self._buffer.read_ascii(tca_string_length)

                    storm_number = int(re.search(
                        r'(?<=<tca_stormNum>)(.+?)(?=<)', tca_string
                    ).group())

                    issue_status = re.search(
                        r'(?<=<tca_issueStatus>)(.+?)(?=<)', tca_string
                    ).group()

                    basin = int(re.search(
                        r'(?<=<tca_basin>)(.+?)(?=<)', tca_string
                    ).group())

                    advisory_number = int(re.search(
                        r'(?<=<tca_advisoryNum>)(.+?)(?=<)', tca_string
                    ).group())

                    storm_name = re.search(
                        r'(?<=<tca_stormName>)(.+?)(?=<)', tca_string
                    ).group()

                    storm_type = int(re.search(
                        r'(?<=<tca_stormType>)(.+?)(?=<)', tca_string
                    ).group())

                    valid = re.search(
                        r'(?<=<tca_validTime>)(.+?)(?=<)', tca_string
                    ).group()

                    tz = re.search(
                        r'(?<=<tca_timezone>)(.+?)(?=<)', tca_string
                    ).group()

                    text_lat = float(re.search(
                        r'(?<=<tca_textLat>)(.+?)(?=<)', tca_string
                    ).group())

                    text_lon = float(re.search(
                        r'(?<=<tca_textLon>)(.+?)(?=<)', tca_string
                    ).group())

                    text_font = int(re.search(
                        r'(?<=<tca_textFont>)(.+?)(?=<)', tca_string
                    ).group())

                    text_size = float(re.search(
                        r'(?<=<tca_textSize>)(.+?)(?=<)', tca_string
                    ).group())

                    text_width = int(re.search(
                        r'(?<=<tca_textWidth>)(.+?)(?=<)', tca_string
                    ).group())

                    wwnum = int(re.search(
                        r'(?<=<tca_wwNum>)(.+?)(?=<)', tca_string
                    ).group())

                    ww = []
                    for n in range(wwnum):
                        wwstr = re.search(
                            rf'(?<=<tca_tcawwStr_{n}>)(.+?)(?=<)', tca_string
                        ).group()
                        nbreaks = int(re.search(
                            rf'(?<=<tca_numBreakPts_{n}>)(.+?)(?=<)', tca_string
                        ).group())
                        breakpts = re.search(
                            rf'(?<=<tca_breakPts_{n}>)(.+?)(?=<|$)', tca_string
                        ).group()

                        severity, advisory_type, special_geog = wwstr.split('|')

                        parsed_breaks = np.array_split(breakpts.split('|'), nbreaks)
                        decode_breaks = [[float(lat), float(lon), bname]
                                         for lat, lon, bname in parsed_breaks]

                        ww.append(
                            {
                                'severity': Severity(int(severity)),
                                'advisory_type': AdvisoryType(int(advisory_type)),
                                'special_geography': SpecialGeography(int(special_geog)),
                                'number_breaks': nbreaks,
                                'break_points': decode_breaks
                            }
                        )

                        self._elements.append(
                            TropicalCycloneAdvisoryElement(
                                header_struct, storm_number, issue_status, basin,
                                advisory_number, storm_name, storm_type, valid, tz, text_lat,
                                text_lon, text_font, text_size, text_width, wwnum, ww)
                        )
                elif vg_type in [VGType.tc_error_cone.value, VGType.tc_track.value,
                                 VGType.tc_break_point.value]:
                        tc_info = [
                            ('storm_number', '5s', self._decode_strip_null),
                            ('issue_status', '2s', self._decode_strip_null),
                            ('basin', '5s', self._decode_strip_null),
                            ('advisory_number', '5s', self._decode_strip_null),
                            ('storm_name', '128s', self._decode_strip_null),
                            ('storm_type', '5s', self._decode_strip_null),
                            ('valid_time', '21s', self._decode_strip_null),
                            ('timezone', '4s', self._decode_strip_null),
                            ('forecast_period', '5s', self._decode_strip_null)
                        ]

                        tc = self._buffer.read_struct(
                            NamedStruct(tc_info, self.prefmt, 'TCInfo')
                        )

                        if vg_type == VGType.tc_error_cone.value:
                            cone_info = [
                                ('line_color', 'i'), ('line_type', 'i'), ('fill_color', 'i'),
                                ('fill_type', 'i'), ('number_points', 'i')
                            ]

                            cone = self._buffer.read_struct(
                                NamedStruct(cone_info, self.prefmt, 'TCConeInfo')
                            )

                            lat, lon = self._get_latlon(cone.number_points)

                            self._elements.append(
                                TropicalCycloneErrorElement(
                                    header_struct, tc.storm_number, tc.issue_status, tc.basin,
                                    tc.advisory_number, tc.storm_name, tc.storm_type,
                                    tc.valid_time, tc.timezone, tc.forecast_period,
                                    cone.line_color, cone.line_type, cone.fill_color,
                                    cone.fill_type, cone.number_points, lat, lon)
                            )
                        elif vg_type == VGType.tc_track.value:
                            track_info = [
                                ('line_color', 'i'), ('line_type', 'i'), ('number_points', 'i')
                            ]

                            track = self._buffer.read_struct(
                                NamedStruct(track_info, self.prefmt, 'TCTrackInfo')
                            )

                            track_point_info = [
                                ('lat', 'f', self._round_two), ('lon', 'f', self._round_two),
                                ('advisory_date', '50s', self._decode_strip_null),
                                ('tau', '50s', self._decode_strip_null),
                                ('max_wind', '50s', self._decode_strip_null),
                                ('wind_gust', '50s', self._decode_strip_null),
                                ('minimum_pressure', '50s', self._decode_strip_null),
                                ('development_level', '50s', self._decode_strip_null),
                                ('development_label', '50s', self._decode_strip_null),
                                ('direction', '50s', self._decode_strip_null),
                                ('speed', '50s', self._decode_strip_null),
                                ('date_label', '50s', self._decode_strip_null),
                                ('storm_source', '50s', self._decode_strip_null),
                                (None, '2x')  # skip struct alignment padding bytes
                            ]

                            point_struct = NamedStruct(track_point_info, self.prefmt,
                                                       'TrackPointInfo')

                            track_points = []
                            for _ in range(track.number_points):
                                pt = self._buffer.read_struct(point_struct)
                                track_points.append(
                                    TrackAttribute(
                                        pt.advisory_date, pt.tau, pt.max_wind, pt.wind_gust,
                                        pt.minimum_pressure, pt.development_level,
                                        pt.development_label, pt.direction, pt.speed,
                                        pt.date_label, pt.storm_source, pt.lat, pt.lon)
                                )

                            self.elements.append(
                                TropicalCycloneTrackElement(
                                    header_struct, tc.storm_number, tc.issue_status, tc.basin,
                                    tc.advisory_number, tc.storm_name, tc.storm_type,
                                    tc.valid_time, tc.timezone, tc.forecast_period,
                                    track.line_color, track.line_type, track.number_points,
                                    track_points)
                            )
                        elif vg_type == VGType.tc_break_point.value:
                            break_info = [
                                ('line_color', 'i'), ('line_width', 'i'), ('ww_level', 'i'),
                                ('number_points', 'i')
                            ]

                            brkpt = self._buffer(break_info, self.prefmt,
                                                      'BreakPointInfo')

                            break_meta = [
                                ('lat', 'f', self._round_two), ('lon', 'f', self._round_two),
                                ('name', '256s', self._decode_strip_null)
                            ]

                            break_struct = NamedStruct(break_meta, self.prefmt, 'BreakPoint')

                            breakpoints = []
                            for _ in range(brkpt.number_points):
                                bp = self._buffer.read_struct(break_struct)
                                breakpoints.append(
                                    BreakPointAttribute(bp.lat, bp.lon, bp.name)
                                )

                            self._elements.append(
                                TropicalCycloneBreakPointElement(
                                    header_struct, tc.storm_number, tc.issue_status,
                                    tc.basin, tc.advisory_number, tc.storm_name,
                                    tc.storm_type, tc.valid_time, tc.timezone,
                                    tz.forecast_period, brkpt.line_color, brkpt.line_width,
                                    brkpt.ww_level, brkpt.number_points, breakpoints
                                )
                            )
                        elif vg_type == VGType.sgwx.value:
                            sgwx_info = [
                                ('subtype', 'i'), ('number_points', 'i'),
                                ('text_lat', 'f', self._round_two),
                                ('text_lon', 'f', self._round_two),
                                ('arrow_lat', 'f', self._round_two),
                                ('arrow_lon', 'f', self._round_two),
                                ('line_element', 'i'), ('line_type', 'i'),
                                ('line_width', 'i'), ('arrow_size', 'f', self._round_one),
                                ('special_symbol', 'i'), ('weather_symbol', 'i')
                            ]

                            sgwx = self._buffer.read_struct(
                                NamedStruct(sgwx_info, self.prefmt, 'SGWXInfo')
                            )

                            special_text_info = [
                                ('rotation', 'f', self._round_one),
                                ('text_size', 'f', self._round_one), ('text_type', 'i'),
                                ('turbulence_symbol', 'i'), ('font', 'i'), ('text_flag', 'i'),
                                ('width', 'i'), ('text_color', 'i'), ('line_color', 'i'),
                                ('fill_color', 'i'), ('align', 'i'),
                                ('lat', 'f', self._round_two), ('lon', 'f', self._round_two),
                                ('offset_x', 'i'), ('offset_y', 'i')
                            ]

                            text = self._buffer.read_struct(
                                NamedStruct(special_text_info, self.prefmt, 'SpecialTextInfo')
                            )

                            lat, lon = self._get_latlon(sgwx.number_points)

                            self._elements.append(
                                SignificantWeatherElement(
                                    header_struct, sgwx.subtype, sgwx.number_points,
                                    sgwx.text_lat, sgwx.text_lon, sgwx.arrow_lat,
                                    sgwx.arrow_lon, sgwx.line_element, sgwx.line_type,
                                    sgwx.line_width, sgwx.arrow_size, sgwx.special_symbol,
                                    sgwx.weather_symbol, text.rotation, text.text_size,
                                    text.text_type, text.turbulence_symbol, text.font,
                                    text.text_flag, text.text_width, text.text_color,
                                    text.line_color, text.fill_color, text.text_align,
                                    text.offset_x, text.offset_y, text.text, lat, lon
                                )
                            )

                            self. _buffer.skip((MAX_SGWX_POINTS - sgwx.number_points) * 4)
                else:
                    raise NotImplementedError(f'MET type `{vg_type}` not implemented.')
            elif vg_class == VGClass.watches.value:
                watch_info = [
                    ('number_points', 'i'), ('style', 'i'), ('shape', 'i'),
                    ('marker_type', 'i'), ('marker_size', 'f', self._round_one),
                    ('marker_width', 'i'), ('anchor0_station', '8s', self._decode_strip_null),
                    ('anchor0_lat', 'f', self._round_two),
                    ('anchor0_lon', 'f', self._round_two), ('anchor0_distance', 'i'),
                    ('anchor0_direction', '4s', self._decode_strip_null),
                    ('anchor1_station', '8s', self._decode_strip_null),
                    ('anchor1_lat', 'f', self._round_two),
                    ('anchor1_lon', 'f', self._round_two), ('anchor1_distance', 'i'),
                    ('anchor1_direction', '4s', self._decode_strip_null), ('status', 'i'),
                    ('number', 'i'), ('issue_time', '20s', self._decode_strip_null),
                    ('expire_time', '20s', self._decode_strip_null), ('watch_type', 'i'),
                    ('severity', 'i'), ('timezone', '4s', self._decode_strip_null),
                    ('max_hail', '8s', self._decode_strip_null),
                    ('max_wind', '8s', self._decode_strip_null),
                    ('max_tops', '8s', self._decode_strip_null),
                    ('mean_storm_direction', '8s', self._decode_strip_null),
                    ('mean_storm_speed', '8s', self._decode_strip_null),
                    ('states', '80s', self._decode_strip_null),
                    ('adjacent_areas', '80s', self._decode_strip_null),
                    ('replacing', '24s', self._decode_strip_null),
                    ('forecaster', '64s', self._decode_strip_null),
                    ('filename', '128s', self._decode_strip_null),
                    ('issue_flag', 'i'), ('wsm_issue_time', '20s', self._decode_strip_null),
                    ('wsm_expire_time', '20s', self._decode_strip_null),
                    ('wsm_reference_direction', '32s', self._decode_strip_null),
                    ('wsm_recent_from_line', '128s', self._decode_strip_null),
                    ('wsm_md_number', '8s', self._decode_strip_null),
                    ('wsm_forecaster', '64s', self._decode_strip_null),
                    ('number_counties', 'i'), ('plot_counties', 'i')
                ]

                watch = self._buffer.read_struct(
                    NamedStruct(watch_info, self.prefmt, 'WatchBoxInfo')
                )

                county_fips = self._buffer.read_array(watch.number_counties, f'{self.prefmt}i')
                county_fips_blank_size = 4 * (MAX_COUNTIES - watch.number_counties)
                self._buffer.skip(county_fips_blank_size)

                county_lat = self._buffer.read_array(watch.number_counties, f'{self.prefmt}f')
                county_lon = self._buffer.read_array(watch.number_counties, f'{self.prefmt}f')
                county_loc_blank_size = 8 * (MAX_COUNTIES - watch.number_counties)
                self._buffer.skip(county_loc_blank_size)

                lat = self._buffer.read_array(watch.number_points, f'{self.prefmt}f')
                lon = self._buffer.read_array(watch.number_points, f'{self.prefmt}f')

                # Manually close watch parallelogram
                if watch.number_points > 2:
                    lon, lat = self.close_coordinates(lon, lat)

                self._elements.append(
                    WatchBoxElement(header_struct, watch.number_points, watch.style,
                                    watch.shape, watch.marker_type, watch.marker_size,
                                    watch.marker_width, watch.anchor0_station,
                                    watch.anchor0_lat, watch.anchor0_lon,
                                    watch.anchor0_distance, watch.anchor0_direction,
                                    watch.anchor1_station, watch.anchor1_lat,
                                    watch.anchor1_lon, watch.anchor1_distance,
                                    watch.anchor1_direction, watch.status,
                                    watch.number, watch.issue_time,
                                    watch.expire_time, watch.watch_type,
                                    watch.severity, watch.timezone,
                                    watch.max_hail, watch.max_wind,
                                    watch.max_tops, watch.mean_storm_direction,
                                    watch.mean_storm_speed, watch.states,
                                    watch.adjacent_areas, watch.replacing,
                                    watch.forecaster, watch.filename, watch.issue_flag,
                                    watch.wsm_issue_time, watch.wsm_expire_time,
                                    watch.wsm_reference_direction,
                                    watch.wsm_recent_from_line, watch.wsm_md_number,
                                    watch.wsm_forecaster, watch.number_counties,
                                    watch.plot_counties, county_fips, county_lat,
                                    county_lon, lat, lon)
                )
            elif vg_class == VGClass.winds.value:
                wind_info = [
                    ('number_wind', 'i'), ('width', 'i'), ('size', 'f', self._round_one),
                    ('wind_type', 'i'), ('head_size', 'f', self._round_one),
                    ('speed', 'f', self._round_two), ('direction ', 'f', self._round_two),
                    ('lat', 'f', self._round_two), ('lon', 'f', self._round_two)
                ]
                wind = self._buffer.read_struct(
                    NamedStruct(wind_info, self.prefmt, 'WindInfo')
                )

                self._elements.append(
                    WindElement(header_struct, wind.number_wind, wind.width, wind.size,
                                wind.wind_type, wind.head_size, wind.speed, wind.direction,
                                wind.lat, wind.lon)
                )
            else:
                logger.warning('Could not decode element with class `%s` and type `%s`',
                               VGClass(vg_class).name, VGType(vg_type).name)
                _ = self._buffer.skip(data_size)

    def _get_latlon(self, points):
        """Extract latitude and longitude from VGF element.

        Parameters
        ----------
        points : int
            Number of points to be decoded.

        Returns
        -------
        tuple of lat, lon of `points` dimension
        """
        truncated = points > MAX_POINTS
        if truncated:
            raise ValueError('Exceeded maximum number of points in element.')
        else:
            if points >= 1:
                lat = np.around(self._buffer.read_array(points, f'{self.prefmt}f'), 2)
                lon = np.around(self._buffer.read_array(points, f'{self.prefmt}f'), 2)
            else:
                lat = -9999
                lon = -9999

        return lat, lon

    def _read_header(self):
        """Read VGF header.

        Notes
        -----
        Header size should be 40 bytes (see GEMPAK vgstruct.h).
        """
        vgf_header_info = [
            ('delete', 'c', ord), ('vg_type', 'c', ord), ('vg_class', 'c', ord),
            ('filled', 'b'), ('closed', 'c', ord), ('smooth', 'c', ord),
            ('version', 'c', ord), ('group_type', 'c', ord), ('group_number', 'i'),
            ('major_color', 'i'), ('minor_color', 'i'), ('record_size', 'i'),
            ('min_lat', 'f', self._round_two), ('min_lon', 'f', self._round_two),
            ('max_lat', 'f', self._round_two), ('max_lon', 'f', self._round_two)
        ]

        return self._buffer.read_struct(NamedStruct(
            vgf_header_info, self.prefmt, 'Header'
        ))

    @staticmethod
    def close_coordinates(x, y):
        """Close polygon coordinates.

        Parameters
        ----------
        x : array_like

        y : array_like

        Returns
        -------
        tuple of x, y of closed polygon
        """
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            x = np.concatenate([x, [x[0]]])
            y = np.concatenate([y, [y[0]]])
        elif isinstance(x, list) and isinstance(y, list):
            x += x[0]
            y += y[0]
        else:
            raise TypeError('x and y must be of the same type: list or array.')

        return x, y

    @staticmethod
    def flip_coordinates(x, y):
        """Flip coordinate direction.

        Parameters
        ----------
        x : array_like

        y : array_like

        Returns
        -------
        tuple of reversed x, y
        """
        if ((isinstance(x, np.ndarray) and isinstance(y, np.ndarray))
           or (isinstance(x, list) and isinstance(y, list))):
            x = x[::-1]
            y = y[::-1]
        else:
            raise TypeError('x and y must be of the same type: list or array.')

        return x, y

    @staticmethod
    def _decode_strip_null(x):
        """Decode bytes into string and truncate based on null terminator.

        Parameters
        ----------
        x : bytes

        Returns
        -------
        str with whitespace and null stripped

        Notes
        -----
        The bytes array is first decoded UTF-8 to get rid of non-UTF-8
        characters. GEMPAK used static char array sizes and often had
        junk data after the string. The string is properly truncated
        after finding the null terminator and then whitespace is
        stripped.
        """
        decoded = x.decode('utf-8', errors='ignore')
        null = decoded.find('\x00')
        return decoded[:null].strip()

    @staticmethod
    def _round_one(x):
        """Round to one decimal.

        Parameters
        ----------
        x : float

        Returns
        -------
        float rounded to 1 decimal
        """
        return round(x, 1)

    @staticmethod
    def _round_two(x):
        """Round to two decimals.

        Parameters
        ----------
        x : float

        Returns
        -------
        float rounded to 2 decimals
        """
        return round(x, 2)

    @staticmethod
    def _swap32(x):
        """Swap bytes of 32-bit float or int.

        Parameters
        ----------
        x : float or int

        Returns
        -------
        byte-swapped float or int
        """
        return int.from_bytes(x.to_bytes(4, byteorder='little'),
                              byteorder='big', signed=False)
