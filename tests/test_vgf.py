# Copyright (c) 2025 Nathan Wendt.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for decoding GEMPAK VGF files."""

from pathlib import Path

import numpy as np

from gempakio import VectorGraphicFile
from gempakio.decode.vgf import (
    AdvisoryType,
    BarbAttribute,
    HashAttribute,
    LineAttribute,
    Severity,
    SpecialGeography,
    VGType,
)


def test_fills():
    """Test decoding of fill colors."""
    expected_fills = [2, 3, 4, 5, 6, 7, 8]

    vgf = Path(__file__).parent / 'data' / 'fills.vgf'

    v = VectorGraphicFile(vgf)

    decoded_fills = [e.filled for e in v.elements]

    assert decoded_fills == expected_fills


def test_fronts():
    """Test decoding of front elements."""
    expected_character = [
        'unspecified',
        'forming_suspected',
        'diffuse',
        'forming_suspected',
        'unspecified',
        'diffuse',
        'forming_suspected',
        'unspecified',
        'diffuse',
        'forming_suspected',
        'unspecified',
        'diffuse',
        'unspecified',
        'unspecified',
        'position_doubtful',
        'unspecified',
    ]

    expected_code = [
        420,
        425,
        428,
        225,
        220,
        228,
        25,
        20,
        28,
        625,
        620,
        628,
        720,
        820,
        829,
        940,
    ]

    expected_type = [
        'cold',
        'cold',
        'cold',
        'warm',
        'warm',
        'warm',
        'stationary',
        'stationary',
        'stationary',
        'occluded',
        'occluded',
        'occluded',
        'dryline',
        'intertropical',
        'intertropical',
        'convergence',
    ]

    expected_lat = [
        np.array([44.39, 46.1, 48.72], dtype='f4'),
        np.array([45.52, 46.99, 49.66], dtype='f4'),
        np.array([43.2, 45.18, 47.77], dtype='f4'),
        np.array([46.75, 43.23, 41.41], dtype='f4'),
        np.array([45.88, 41.44, 40], dtype='f4'),
        np.array([44.37, 40.28, 38.38], dtype='f4'),
        np.array([43.33, 40.27, 37.25], dtype='f4'),
        np.array([42.23, 39.2, 36.18], dtype='f4'),
        np.array([41.06, 38.08, 35.42, 34.55], dtype='f4'),
        np.array([39.77, 36.59, 33.98, 33.56], dtype='f4'),
        np.array([38.8, 35.08, 32.99, 32.26], dtype='f4'),
        np.array([37.44, 33.88, 32.19, 31.04], dtype='f4'),
        np.array([36.1, 33.29, 31.91, 30.31], dtype='f4'),
        np.array([34.94, 32.11, 30.09, 28.98], dtype='f4'),
        np.array([33.95, 31.08, 28.1], dtype='f4'),
        np.array([32.9, 28.97, 27.02], dtype='f4'),
    ]

    expected_lon = [
        np.array([-116.26, -109.71, -103.07], dtype='f4'),
        np.array([-116.94, -111.77, -104.71], dtype='f4'),
        np.array([-115.27, -108.24, -101.73], dtype='f4'),
        np.array([-101.06, -108.17, -114.44], dtype='f4'),
        np.array([-99.52, -107.8, -112.03], dtype='f4'),
        np.array([-99.12, -105.8, -110.78], dtype='f4'),
        np.array([-97.61, -102.49, -109.13], dtype='f4'),
        np.array([-96.42, -100.62, -107.37], dtype='f4'),
        np.array([-95.14, -98.52, -103.62, -106.13], dtype='f4'),
        np.array([-94.18, -97.52, -102.4, -104.39], dtype='f4'),
        np.array([-93.06, -96.71, -100.6, -102.41], dtype='f4'),
        np.array([-91.72, -95.66, -98.62, -101.39], dtype='f4'),
        np.array([-90.63, -94.15, -96.42, -99.77], dtype='f4'),
        np.array([-88.81, -92.72, -95.76, -98.85], dtype='f4'),
        np.array([-87.5, -90.96, -97.18], dtype='f4'),
        np.array([-86, -90.88, -96.05], dtype='f4'),
    ]

    vgf = Path(__file__).parent / 'data' / 'fronts.vgf'

    v = VectorGraphicFile(vgf)

    decoded_character = [e.front_character for e in v.elements]
    decoded_code = [e.front_code for e in v.elements]
    decoded_type = [e.front_type for e in v.elements]
    decoded_lat = [e.lat for e in v.elements]
    decoded_lon = [e.lon for e in v.elements]

    assert decoded_character == expected_character
    assert decoded_code == expected_code
    assert decoded_type == expected_type
    np.testing.assert_equal(decoded_lat, expected_lat)
    np.testing.assert_equal(decoded_lon, expected_lon)


def test_jet():
    """Test decoding of jet element."""
    expected_jet = {
        'delete': 0,
        'vg_type': 37,
        'vg_class': 15,
        'filled': 0,
        'closed': 0,
        'smooth': 2,
        'version': 0,
        'group_type': 0,
        'group_number': 0,
        'major_color': 2,
        'minor_color': 2,
        'record_size': 24076,
        'min_lat': 42.88,
        'min_lon': -112.21,
        'max_lat': 44.08,
        'max_lon': -94.33,
        'line_attribute': LineAttribute(
            line_color=2,
            number_points=3,
            line_type=6,
            stroke=1,
            direction=0,
            size=1.0,
            width=7,
            lat=[42.88, 43.06, 44.08],
            lon=[-112.21, -105.11, -94.33],
        ),
        'number_barbs': 1,
        'barb_attributes': [
            BarbAttribute(
                wind_color=2,
                number_wind=1,
                width=801,
                size=1.0,
                wind_type=114,
                head_size=0.0,
                speed=100.0,
                direction=-96.99,
                lat=43.06,
                lon=-105.07,
                flight_level_color=2,
                text_rotation=1367.0,
                text_size=1.1,
                text_type=5,
                turbulence_symbol=0,
                font=1,
                text_flag=1,
                text_width=1,
                text_color=2,
                line_color=2,
                fill_color=2,
                align=0,
                text_lat=43.06,
                text_lon=-105.07,
                offset_x=0,
                offset_y=-2,
                text='FL300',
            )
        ],
        'number_hashes': 1,
        'hash_attributes': [
            HashAttribute(
                wind_color=2,
                number_wind=1,
                width=2,
                size=1.0,
                wind_type=1,
                head_size=0.0,
                speed=0.0,
                direction=-89.47,
                lat=42.86,
                lon=-110.08,
            )
        ],
    }

    vgf = Path(__file__).parent / 'data' / 'jet.vgf'

    v = VectorGraphicFile(vgf)

    jet = v.elements[0]

    assert jet.__dict__ == expected_jet


def test_lines():
    """Test decoding of line elements."""
    expected_style = [
        'solid',
        'long_dash',
        'fancy_arrow',
        'pointed_arrow',
        'filled_arrow',
        'line_circle_line',
        'line_fill_circle_line',
        'alt_angle_ticks',
        'tick_marks',
        'scallop',
        'zigzag',
        'sine_curve',
        'ball_chain',
        'box_circles',
        'filled_circles',
        'line_x_line',
        'two_x',
        'box_x',
        'line_caret_line2',
        'line_caret_line1',
        'fill_circle_x',
        'arrow_dashed',
        'fill_arrow_dash',
        'streamline',
        'double_line',
        'short_dashed',
        'medium_dashed',
        'long_dash_short_dash',
        'long_dash',
        'long_dash_three_short_dash',
        'long_dash_dot',
        'long_dash_three_dot',
        'extra_long_dash_two_dot',
        'dotted',
        'kink_line1',
        'kink_line2',
        'z_line',
    ]

    expected_type = [
        1,
        5,
        13,
        4,
        6,
        16,
        10,
        5,
        11,
        3,
        2,
        19,
        1,
        7,
        9,
        12,
        8,
        15,
        18,
        17,
        14,
        20,
        21,
        22,
        23,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        24,
        25,
        26,
    ]

    expected_color = [
        3,
        5,
        17,
        18,
        18,
        17,
        2,
        7,
        17,
        18,
        5,
        2,
        3,
        2,
        2,
        17,
        2,
        17,
        1,
        1,
        17,
        18,
        18,
        7,
        2,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        2,
        2,
        2,
    ]

    expected_lat = [
        np.array([47.07, 48.35, 47.42, 46.03, 46.51, 47.07], dtype='f4'),
        np.array([47.74, 48.69, 46.44, 46.44, 47.74], dtype='f4'),
        np.array([44.14, 46.3, 47.83, 48.46], dtype='f4'),
        np.array([42.8, 44.63, 46.51], dtype='f4'),
        np.array([41.75, 43.91, 45.05, 46.25], dtype='f4'),
        np.array([40.68, 43.21, 44.79], dtype='f4'),
        np.array([39.75, 41.01, 42.77, 44.22], dtype='f4'),
        np.array([33.35, 35.43, 37.46, 39.05], dtype='f4'),
        np.array([32, 34.07, 36.45, 38.29], dtype='f4'),
        np.array([31.49, 33.13, 35.67, 37.49, 38.89], dtype='f4'),
        np.array([30.43, 32.35, 34.68, 36.14, 37.78, 38.71], dtype='f4'),
        np.array([30.24, 32.23, 34.46, 36.42, 37.77, 38.82], dtype='f4'),
        np.array([30.22, 33.27, 34.9, 36.88, 38.74], dtype='f4'),
        np.array([29.8, 32.46, 35.12, 37.11, 38.22], dtype='f4'),
        np.array([29.11, 31.96, 34.34, 36.49, 37.95], dtype='f4'),
        np.array([29.44, 32.46, 35.51, 37.16, 38.14], dtype='f4'),
        np.array([28.8, 31.27, 33.85, 35.73, 36.94], dtype='f4'),
        np.array([28.23, 30.75, 32.83, 35.23, 36.71], dtype='f4'),
        np.array([28.02, 30.89, 33.85, 35.25, 36.52], dtype='f4'),
        np.array([28.03, 31.79, 34.1, 35.61, 36.85], dtype='f4'),
        np.array([27.59, 30.75, 33.24, 34.87, 36], dtype='f4'),
        np.array([27.57, 30.26, 32.74, 34.37, 35.43], dtype='f4'),
        np.array([26.91, 29.62, 31.92, 33.43, 34.39], dtype='f4'),
        np.array([26.84, 29.69, 32.54, 33.81], dtype='f4'),
        np.array([26.23, 29.35, 32.41, 33.8], dtype='f4'),
        np.array([39.43, 40.88, 42.79, 45.21], dtype='f4'),
        np.array([38.6, 39.8, 41.49, 43.96], dtype='f4'),
        np.array([38.09, 39.87, 41.88, 43.15], dtype='f4'),
        np.array([37.6, 39.18, 41.16, 42.3], dtype='f4'),
        np.array([37.14, 38.68, 40.31, 41.48], dtype='f4'),
        np.array([36.44, 38.02, 39.63, 40.7], dtype='f4'),
        np.array([35.3, 38.14, 39.76], dtype='f4'),
        np.array([34.57, 36.27, 38.35, 39.52], dtype='f4'),
        np.array([33.4, 34.93, 37.19, 38.84], dtype='f4'),
        np.array([37.86, 40.68], dtype='f4'),
        np.array([39.99, 42.45], dtype='f4'),
        np.array([42.81, 41.18, 39.98], dtype='f4'),
    ]

    expected_lon = [
        np.array([-121.06, -119.31, -118.09, -120.04, -120.87, -121.06], dtype='f4'),
        np.array([-113.58, -108.01, -110.21, -113.02, -113.58], dtype='f4'),
        np.array([-108.63, -103.41, -102.13, -102.34], dtype='f4'),
        np.array([-106.99, -102.73, -99.56], dtype='f4'),
        np.array([-105.27, -100.43, -98.16, -96.73], dtype='f4'),
        np.array([-103.64, -98.01, -95.44], dtype='f4'),
        np.array([-102.66, -99.05, -95.96, -93.87], dtype='f4'),
        np.array([-117.15, -115.95, -116.56, -117.14], dtype='f4'),
        np.array([-114.46, -113.63, -113.4, -113.77], dtype='f4'),
        np.array([-112.5, -112.13, -111.94, -111.96, -112.25], dtype='f4'),
        np.array([-110.05, -109.57, -109.66, -109.76, -110.14, -110.37], dtype='f4'),
        np.array([-107.6, -107.33, -107.24, -107.2, -107.42, -107.77], dtype='f4'),
        np.array([-105.72, -105.44, -105.23, -104.98, -105.13], dtype='f4'),
        np.array([-104.02, -103.18, -103.08, -102.95, -103.1], dtype='f4'),
        np.array([-102.7, -101.51, -101.16, -100.75, -100.87], dtype='f4'),
        np.array([-100.84, -99.61, -98.65, -98.71, -98.66], dtype='f4'),
        np.array([-99.39, -98.2, -97.41, -97.09, -96.96], dtype='f4'),
        np.array([-97.77, -96.24, -95.57, -95.29, -95.14], dtype='f4'),
        np.array([-95.38, -94.05, -93.22, -93.14, -93.12], dtype='f4'),
        np.array([-93.28, -91.54, -91.39, -91.33, -91.27], dtype='f4'),
        np.array([-90.9, -89.38, -88.37, -87.98, -88.11], dtype='f4'),
        np.array([-89.24, -87.89, -86.93, -86.41, -86.25], dtype='f4'),
        np.array([-87.41, -86.46, -85.83, -85.3, -85.06], dtype='f4'),
        np.array([-85.45, -84.8, -83.98, -83.96], dtype='f4'),
        np.array([-83.23, -82.49, -82.35, -82.08], dtype='f4'),
        np.array([-97.78, -94.2, -92.43, -90.99], dtype='f4'),
        np.array([-95.11, -92.55, -90.97, -89.06], dtype='f4'),
        np.array([-93.12, -89.84, -87.63, -86.83], dtype='f4'),
        np.array([-90.99, -88.1, -86.08, -85.29], dtype='f4'),
        np.array([-88.81, -86, -84.38, -83.49], dtype='f4'),
        np.array([-86.01, -83.57, -82.35, -81.78], dtype='f4'),
        np.array([-83.01, -80.67, -79.9], dtype='f4'),
        np.array([-81.27, -79.73, -78.14, -77.62], dtype='f4'),
        np.array([-119.12, -118, -118.21, -118.74], dtype='f4'),
        np.array([-122.27, -121.27], dtype='f4'),
        np.array([-124.43, -121.66], dtype='f4'),
        np.array([-118.1, -111.86, -106.45], dtype='f4'),
    ]

    vgf = Path(__file__).parent / 'data' / 'lines.vgf'

    v = VectorGraphicFile(vgf)

    decoded_style = [e.line_style for e in v.elements]
    decoded_type = [e.line_type for e in v.elements]
    decoded_color = [e.major_color for e in v.elements]
    decoded_lat = [e.lat for e in v.elements]
    decoded_lon = [e.lon for e in v.elements]

    assert decoded_style == expected_style
    assert decoded_type == expected_type
    assert decoded_color == expected_color
    np.testing.assert_equal(decoded_lat, expected_lat)
    np.testing.assert_equal(decoded_lon, expected_lon)


def test_markers():
    """Test decoding of marker elements."""
    expected_code = [10, 5, 16, 11]
    expected_lat = [47.42, 44.16, 43.68, 47.17]
    expected_lon = [-119.59, -120.34, -114.85, -109.22]

    vgf = Path(__file__).parent / 'data' / 'misc.vgf'

    v = VectorGraphicFile(vgf)

    markers = v.filter_elements(vg_type=19)

    decoded_code = [e.symbol_code for e in markers]
    decoded_lat = [e.lat for e in markers]
    decoded_lon = [e.lon for e in markers]

    assert decoded_code == expected_code
    assert decoded_lat == expected_lat
    assert decoded_lon == expected_lon


def test_sigmet_airmet():
    """Test decoding AIRMET element."""
    expected_airmet = {
        'delete': 0,
        'vg_type': 31,
        'vg_class': 11,
        'filled': 0,
        'closed': 1,
        'smooth': 0,
        'version': 1,
        'group_type': 0,
        'group_number': 0,
        'major_color': 3,
        'minor_color': 3,
        'record_size': 816,
        'min_lat': 40.0,
        'min_lon': -104.87,
        'max_lat': 40.0,
        'max_lon': -104.87,
        'subtype': 0,
        'number_points': 5,
        'line_type': 1,
        'line_width': 2,
        'side_of_line': 0,
        'area': 'KSFO',
        'flight_info_region': '',
        'status': 'KSFO',
        'distance': 10.0,
        'message_id': 'SIERRA',
        'sequence_number': 0,
        'start_time': '180010',
        'end_time': '180410',
        'remarks': '',
        'sonic': -9999,
        'phenomena': 'IFR',
        'phenomena2': 'IFR',
        'phenomena_name': '-',
        'phenomena_lat': '',
        'phenomena_lon': '',
        'pressure': -9999,
        'max_wind': -9999,
        'free_text': '',
        'trend': '',
        'movement': 'MVG',
        'type_indicator': -9999,
        'type_time': '',
        'flight_level': -9999,
        'speed': 5,
        'direction': 'N',
        'tops': '-none-|TO| |-none-| |',
        'forecaster': '',
        'lat': np.array([41.852974, 44.174183, 43.476295, 41.47952, 39.995544], dtype='f4'),
        'lon': np.array(
            [-104.86926, -101.10242, -98.236664, -98.69969, -104.125145], dtype='f4'
        ),
    }

    vgf = Path(__file__).parent / 'data' / 'sig_airmet.vgf'

    v = VectorGraphicFile(vgf)

    jet = v.elements[0]

    np.testing.assert_equal(jet.__dict__, expected_airmet)


def test_sigmet_ccf():
    """Test decoding CCF element."""
    expected_ccf = {
        'delete': 0,
        'vg_type': 32,
        'vg_class': 11,
        'filled': 2,
        'closed': 1,
        'smooth': 0,
        'version': 0,
        'group_type': 127,
        'group_number': 1,
        'major_color': 26,
        'minor_color': 26,
        'record_size': 1480,
        'min_lat': 33.79,
        'min_lon': -110.87,
        'max_lat': 33.79,
        'max_lon': -110.87,
        'subtype': 0,
        'number_points': 8,
        'coverage': 3,
        'storm_tops': 1,
        'probability': 1,
        'growth': 2,
        'speed': 45.0,
        'direction': 90.0,
        'text_lat': 43.82,
        'text_lon': -112.48,
        'arrow_lat': 38.8,
        'arrow_lon': -99.5,
        'high_fill': 6,
        'med_fill': 4,
        'low_fill': 2,
        'line_type': 1,
        'arrow_size': 1.0,
        'rotation': 0.0,
        'text_size': 0.0,
        'text_type': 0,
        'turbulence_symbol': 0,
        'font': 1,
        'text_flag': 2,
        'width': 1,
        'fill_color': 0,
        'align': -1,
        'offset_x': 0,
        'offset_y': 0,
        'text': '',
        'text_layout': 'IBDR|221;TEXT|TOPS::ETR;TEXT|GWTH::GWTH;TEXT|CONF::CONF;T',
        'lat': np.array([39.7, 43.65, 43.86, 41.52, 37.4, 33.79, 34, 36.71], dtype='f4'),
        'lon': np.array(
            [-110.87, -105.94, -97.59, -90.56, -87.36, -94.02, -104.24, -110.05], dtype='f4'
        ),
    }

    vgf = Path(__file__).parent / 'data' / 'sig_ccf.vgf'

    v = VectorGraphicFile(vgf)

    ccf = v.elements[0]

    np.testing.assert_equal(ccf.__dict__, expected_ccf)


def test_sigmet_international():
    """Test decoding international SIGMET."""
    expected_intnl = {
        'delete': 0,
        'vg_type': 27,
        'vg_class': 11,
        'filled': 0,
        'closed': 1,
        'smooth': 0,
        'version': 1,
        'group_type': 0,
        'group_number': 0,
        'major_color': 6,
        'minor_color': 6,
        'record_size': 824,
        'min_lat': 38.57,
        'min_lon': -122.63,
        'max_lat': 38.57,
        'max_lon': -122.63,
        'subtype': 0,
        'number_points': 6,
        'line_type': 1,
        'line_width': 2,
        'side_of_line': 0,
        'area': 'KKCI',
        'flight_info_region': '',
        'status': 'KKCI',
        'distance': 10.0,
        'message_id': 'ALFA',
        'sequence_number': 1,
        'start_time': '180005',
        'end_time': '180405',
        'remarks': 'BASED_ON_SATELLITE_OBS',
        'sonic': -9999,
        'phenomena': 'FRQ_TS',
        'phenomena2': 'FRQ_TS',
        'phenomena_name': '-',
        'phenomena_lat': '',
        'phenomena_lon': '',
        'pressure': -9999,
        'max_wind': -9999,
        'free_text': '',
        'trend': 'INTSF',
        'movement': 'MVG',
        'type_indicator': -9999,
        'type_time': '',
        'flight_level': -9999,
        'speed': 5,
        'direction': 'N',
        'tops': 'TOPS|ABV|30000|-none-| |',
        'forecaster': '',
        'lat': np.array(
            [41.1635, 44.738934, 47.12407, 41.649994, 39.819687, 38.569443], dtype='f4'
        ),
        'lon': np.array(
            [-122.63099, -119.81144, -107.51364, -108.90593, -115.86577, -120.534325],
            dtype='f4',
        ),
    }

    vgf = Path(__file__).parent / 'data' / 'sig_intnl.vgf'

    v = VectorGraphicFile(vgf)

    intnl = v.elements[0]

    np.testing.assert_equal(intnl.__dict__, expected_intnl)


def test_sigmet_nonconvective():
    """Test decoding of nonconvective SIGMET."""
    expected_nonconv = {
        'delete': 0,
        'vg_type': 28,
        'vg_class': 11,
        'filled': 0,
        'closed': 1,
        'smooth': 0,
        'version': 1,
        'group_type': 0,
        'group_number': 0,
        'major_color': 7,
        'minor_color': 7,
        'record_size': 816,
        'min_lat': 32.49,
        'min_lon': -100.03,
        'max_lat': 32.49,
        'max_lon': -100.03,
        'subtype': 0,
        'number_points': 5,
        'line_type': 1,
        'line_width': 2,
        'side_of_line': 0,
        'area': 'KSFO',
        'flight_info_region': '',
        'status': 'KSFO',
        'distance': 10.0,
        'message_id': 'NOVEMBER',
        'sequence_number': 1,
        'start_time': '180010',
        'end_time': '180410',
        'remarks': '',
        'sonic': -9999,
        'phenomena': 'TURBULENCE',
        'phenomena2': 'TURBULENCE',
        'phenomena_name': '-',
        'phenomena_lat': '',
        'phenomena_lon': '',
        'pressure': -9999,
        'max_wind': -9999,
        'free_text': '',
        'trend': '',
        'movement': 'MVG',
        'type_indicator': -9999,
        'type_time': '',
        'flight_level': -9999,
        'speed': 5,
        'direction': 'N',
        'tops': '-none-|TO| |-none-| |',
        'forecaster': '',
        'lat': np.array([35.586006, 36.903584, 34.845173, 32.494267, 33.518486], dtype='f4'),
        'lon': np.array([-100.02967, -96.04128, -93.95028, -96.28716, -98.671616], dtype='f4'),
    }

    vgf = Path(__file__).parent / 'data' / 'sig_nonconv.vgf'

    v = VectorGraphicFile(vgf)

    nonconv = v.elements[0]

    np.testing.assert_equal(nonconv.__dict__, expected_nonconv)


def test_symbols():
    """Test decoding of symbol elements."""
    expected_code = [
        12,
        13,
        9,
        10,
        45,
        10,
        51,
        56,
        25,
        61,
        71,
        26,
        63,
        73,
        27,
        65,
        75,
        28,
        80,
        85,
        32,
        95,
        105,
        33,
        66,
        79,
        34,
        9,
        38,
        39,
        5,
        40,
        35,
        47,
    ]

    expected_lat = [
        47.29,
        47.28,
        47.29,
        46.22,
        44.2,
        43.21,
        43.89,
        43.39,
        43.0,
        44.38,
        42.12,
        40.38,
        40.18,
        40.39,
        40.85,
        43.22,
        40.53,
        39.77,
        39.1,
        38.9,
        38.64,
        38.39,
        37.58,
        37.61,
        36.78,
        34.3,
        34.55,
        35.59,
        34.83,
        35.83,
        30.84,
        35.23,
        35.74,
        32.47,
    ]

    expected_lon = [
        -120.05,
        -110.51,
        -100.84,
        -94.73,
        -89.63,
        -84.92,
        -121.18,
        -114.09,
        -107.4,
        -100.77,
        -93.53,
        -89.18,
        -86.38,
        -83.2,
        -77.75,
        -75.31,
        -122.37,
        -116.6,
        -112.09,
        -106.68,
        -98.31,
        -92.69,
        -84.51,
        -78.55,
        -119.83,
        -112.05,
        -106.88,
        -97.55,
        -92.55,
        -86.59,
        -103.83,
        -101.7,
        -79.35,
        -86.95,
    ]

    vgf = Path(__file__).parent / 'data' / 'symbols.vgf'

    v = VectorGraphicFile(vgf)

    decoded_code = [e.symbol_code for e in v.elements]
    decoded_lat = [e.lat for e in v.elements]
    decoded_lon = [e.lon for e in v.elements]

    assert decoded_code == expected_code
    assert decoded_lat == expected_lat
    assert decoded_lon == expected_lon


def test_tca():
    """Test decoding of TCA element."""
    expected_tca = {
        'delete': 0,
        'vg_type': 39,
        'vg_class': 15,
        'filled': 0,
        'closed': 0,
        'smooth': 0,
        'version': 0,
        'group_type': 8,
        'group_number': 0,
        'major_color': 0,
        'minor_color': 0,
        'record_size': 409,
        'min_lat': 29.48,
        'min_lon': -93.3,
        'max_lat': 29.78,
        'max_lon': -91.29,
        'storm_number': 1,
        'issue_status': 'T',
        'basin': 0,
        'advisory_number': 1,
        'storm_name': 'Suss',
        'storm_type': 0,
        'valid_time': '240617/0000',
        'timezone': 'EDT',
        'text_lat': 25.799999,
        'text_lon': -80.400002,
        'text_font': 1,
        'text_size': 1.0,
        'text_width': 3,
        'number_ww': 1,
        'ww': [
            {
                'severity': Severity(1),
                'advisory_type': AdvisoryType(1),
                'special_geography': SpecialGeography(0),
                'number_breaks': 2,
                'break_points': [
                    [29.780001, -93.300003, 'CAMERON'],
                    [29.48, -91.290001, 'MORGAN_CITY'],
                ],
            }
        ],
    }

    vgf = Path(__file__).parent / 'data' / 'tca.vgf'

    v = VectorGraphicFile(vgf)

    jet = v.elements[0]

    assert jet.__dict__ == expected_tca


def test_text():
    """Test the decoding of text elements."""
    expected_text = [
        'Test',
        'Test',
        'Test',
        'Test',
        'Test',
        'Test',
        'Test',
        'Test',
        'Test',
        'Test',
    ]

    expected_type = [0, 11, 4, 5, 14, 13, 3, 2, 1, 10]

    expected_color = [20, 6, 2, 4, 30, 7, 19, 12, 19, 3]

    expected_lat = [43.55, 42.57, 39.7, 36.69, 38.76, 42.12, 47.48, 47.0, 43.66, 39.8]

    expected_lon = [
        -121.61,
        -106.06,
        -117.46,
        -118.08,
        -104.56,
        -100.29,
        -99.72,
        -108.78,
        -114.15,
        -110.9,
    ]

    vgf = Path(__file__).parent / 'data' / 'text.vgf'

    v = VectorGraphicFile(vgf)

    decoded_text = [e.text for e in v.elements]
    decoded_type = [e.text_type for e in v.elements]
    decoded_color = [e.text_color for e in v.elements]
    decoded_lat = [e.lat for e in v.elements]
    decoded_lon = [e.lon for e in v.elements]

    assert decoded_text == expected_text
    assert decoded_type == expected_type
    assert decoded_color == expected_color
    assert decoded_lat == expected_lat
    assert decoded_lon == expected_lon


def test_tracks():
    """Test decoding of storm track elements."""
    expected_track = {
        'delete': 0,
        'vg_type': 26,
        'vg_class': 10,
        'filled': 0,
        'closed': 0,
        'smooth': 0,
        'version': 1,
        'group_type': 0,
        'group_number': 0,
        'major_color': 2,
        'minor_color': 32,
        'record_size': 1400,
        'min_lat': 37.4,
        'min_lon': -97.27,
        'max_lat': 37.4,
        'max_lon': -97.27,
        'track_type': 0,
        'total_points': 5,
        'initial_points': 2,
        'initial_line_type': 1,
        'extrapolated_line_type': 2,
        'initial_mark_type': 20,
        'extrapolated_mark_type': 20,
        'line_width': 1,
        'speed': 6.03,
        'direction': 141.85,
        'increment': 60,
        'skip': 0,
        'font': 21,
        'font_flag': 2,
        'font_size': 1.0,
        'times': ['240422/2300', '240423/0000', '240423/0100', '240423/0200', '240423/0300'],
        'lat': np.array([38.03, 37.87, 37.72, 37.56, 37.4], dtype='f4'),
        'lon': np.array([-97.27, -97.12, -96.97, -96.83, -96.68], dtype='f4'),
    }

    vgf = Path(__file__).parent / 'data' / 'tracks.vgf'

    v = VectorGraphicFile(vgf)

    decoded_track = v.elements[0].__dict__

    np.testing.assert_equal(decoded_track, expected_track)


def test_volcano():
    """Test decoding of volcano and ash elements."""
    expected_volcano = {
        'delete': 0,
        'vg_type': 35,
        'vg_class': 11,
        'filled': 0,
        'closed': 0,
        'smooth': 0,
        'version': 0,
        'group_type': 0,
        'group_number': 0,
        'major_color': 6,
        'minor_color': 6,
        'record_size': 5916,
        'min_lat': 46.2,
        'min_lon': -122.18,
        'max_lat': 46.2,
        'max_lon': -122.18,
        'name': 'St._Helens',
        'code': 201.0,
        'size': 2.0,
        'width': 2,
        'number': '321050',
        'location': 'N4620W12218',
        'area': 'US-Washington',
        'origin_station': 'KNES',
        'vaac': 'WASHINGTON',
        'wmo_id': 'XX',
        'header_number': '20',
        'elevation': '8363',
        'year': '2024',
        'advisory_number': '001',
        'correction': '',
        'info_source': 'GOES-17.',
        'additional_source': 'TEST',
        'aviation_color': '',
        'details': 'TEST',
        'obs_date': 'NIL',
        'obs_time': 'NIL',
        'obs_ash': '',
        'forecast_6hr': '18/0500Z',
        'forecast_12hr': '18/1100Z',
        'forecast_18hr': '18/1700Z',
        'remarks': 'THIS IS A TEST.',
        'next_advisory': '',
        'forecaster': 'WENDT',
        'lat': 46.2,
        'lon': -122.18,
        'offset_x': 0,
        'offset_y': 0,
    }

    expected_ash = {
        'delete': 0,
        'vg_type': 36,
        'vg_class': 11,
        'filled': 5,
        'closed': 1,
        'smooth': 0,
        'version': 0,
        'group_type': 0,
        'group_number': 0,
        'major_color': 2,
        'minor_color': 5,
        'record_size': 520,
        'min_lat': 44.19,
        'min_lon': -122.88,
        'max_lat': 44.19,
        'max_lon': -122.88,
        'subtype': 0,
        'number_points': 10,
        'distance': 0.0,
        'forecast_hour': 0,
        'line_type': 1,
        'line_width': 2,
        'side_of_line': 0,
        'speed': 0.0,
        'speeds': '25',
        'direction': '270',
        'flight_level1': 'SFC',
        'flight_level2': '30000',
        'rotation': 0.0,
        'text_size': 1.1,
        'text_type': 4,
        'turbulence_symbol': 0,
        'font': 1,
        'text_flag': 2,
        'width': 1,
        'text_color': 5,
        'line_color': 5,
        'fill_color': 32,
        'align': 0,
        'text_lat': 0.0,
        'text_lon': 0.0,
        'offset_x': 0,
        'offset_y': 0,
        'text': '',
        'lat': np.array(
            [45.84, 46.33, 46.55, 46.6, 46.78, 48.56, 48.25, 46.51, 44.19, 44.19], dtype='f4'
        ),
        'lon': np.array(
            [
                -122.66,
                -122.88,
                -122.22,
                -121.36,
                -120.48,
                -116.92,
                -112.56,
                -113.2,
                -117.89,
                -121.07,
            ],
            dtype='f4',
        ),
    }

    vgf = Path(__file__).parent / 'data' / 'volcano.vgf'

    v = VectorGraphicFile(vgf)

    volcano = v.filter_elements(vg_type=VGType.volcano)[0]
    ash = v.filter_elements(vg_type=VGType.ash_cloud)[0]

    assert volcano.__dict__ == expected_volcano
    np.testing.assert_equal(ash.__dict__, expected_ash)


def test_watch():
    """Test decoding of watch box elements."""
    expected_watch = {
        'delete': 0,
        'vg_type': 6,
        'vg_class': 2,
        'filled': 1,
        'closed': 0,
        'smooth': 0,
        'version': 6,
        'group_type': 0,
        'group_number': 0,
        'major_color': 2,
        'minor_color': 2,
        'record_size': 5736,
        'min_lat': 32.37,
        'min_lon': -101.2,
        'max_lat': 32.37,
        'max_lon': -101.2,
        'number_points': 8,
        'style': 4,
        'shape': 3,
        'marker_type': 1,
        'marker_size': 1.0,
        'marker_width': 3,
        'anchor0_station': 'CDS',
        'anchor0_lat': 34.43,
        'anchor0_lon': -100.28,
        'anchor0_distance': 115,
        'anchor0_direction': 'S',
        'anchor1_station': 'AVK',
        'anchor1_lat': 36.77,
        'anchor1_lon': -98.67,
        'anchor1_distance': 5,
        'anchor1_direction': 'E',
        'status': 0,
        'number': -9999,
        'issue_time': '',
        'expire_time': '',
        'watch_type': 7,
        'severity': 0,
        'timezone': '',
        'max_hail': '',
        'max_wind': '',
        'max_tops': '',
        'mean_storm_speed': '',
        'mean_storm_direction': '',
        'states': 'KS OK TX',
        'adjacent_areas': '',
        'replacing': '',
        'forecaster': '',
        'filename': '',
        'issue_flag': 0,
        'wsm_issue_time': '',
        'wsm_expire_time': '',
        'wsm_reference_direction': '',
        'wsm_recent_from_line': '',
        'wsm_md_number': '',
        'wsm_forecaster': '',
        'number_counties': 57,
        'plot_counties': 1,
        'county_fips': np.array(
            [
                20025,
                20033,
                40003,
                40009,
                40011,
                40015,
                40017,
                40031,
                40033,
                40039,
                40043,
                40045,
                40047,
                40051,
                40055,
                40057,
                40059,
                40065,
                40073,
                40075,
                40093,
                40129,
                40141,
                40149,
                40151,
                40153,
                48009,
                48023,
                48059,
                48075,
                48087,
                48101,
                48107,
                48125,
                48129,
                48151,
                48155,
                48169,
                48191,
                48197,
                48207,
                48211,
                48253,
                48263,
                48269,
                48275,
                48295,
                48345,
                48415,
                48417,
                48433,
                48441,
                48447,
                48483,
                48485,
                48487,
                48503,
            ],
            dtype='i4',
        ),
        'county_status': np.array([], dtype='>i4'),
        'county_lat': np.array(
            [
                37.24,
                37.19,
                36.72,
                35.27,
                35.88,
                35.18,
                35.54,
                34.65,
                34.3,
                35.64,
                36,
                36.21,
                36.38,
                35.02,
                34.92,
                34.74,
                36.78,
                34.53,
                35.93,
                34.96,
                36.31,
                35.7,
                34.37,
                35.29,
                36.79,
                36.41,
                33.62,
                33.62,
                32.3,
                34.53,
                34.95,
                34.06,
                33.62,
                33.61,
                34.97,
                32.75,
                34.03,
                33.18,
                34.53,
                34.28,
                33.18,
                35.83,
                32.74,
                33.2,
                33.62,
                33.6,
                36.28,
                34.07,
                32.74,
                32.73,
                33.17,
                32.3,
                33.18,
                35.41,
                33.99,
                34.06,
                33.17,
            ],
            dtype='f4',
        ),
        'county_lon': np.array(
            [
                -99.83,
                -99.27,
                -98.29,
                -99.7,
                -98.44,
                -98.39,
                -97.97,
                -98.43,
                -98.38,
                -99,
                -99.03,
                -99.73,
                -97.79,
                -97.88,
                -99.52,
                -99.83,
                -99.64,
                -99.26,
                -97.91,
                -99.1,
                -98.54,
                -99.73,
                -98.92,
                -98.99,
                -98.93,
                -99.23,
                -98.7,
                -99.21,
                -99.38,
                -100.22,
                -100.24,
                -100.22,
                -101.3,
                -100.77,
                -100.84,
                -100.42,
                -99.93,
                -101.31,
                -100.68,
                -99.72,
                -99.74,
                -100.26,
                -99.87,
                -100.83,
                -100.26,
                -99.74,
                -100.27,
                -100.77,
                -100.91,
                -99.35,
                -100.24,
                -99.89,
                -99.22,
                -100.3,
                -98.72,
                -99.18,
                -98.68,
            ],
            dtype='f4',
        ),
        'lat': np.array(
            [
                32.765636,
                33.15998,
                35.159996,
                37.159996,
                36.76996,
                36.37,
                34.37,
                32.36999,
                32.765636,
            ],
            dtype='f4',
        ),
        'lon': np.array(
            [
                -100.27998,
                -101.19998,
                -100.37998,
                -99.54999,
                -98.57964,
                -97.61999,
                -98.48998,
                -99.359985,
                -100.27998,
            ],
            dtype='f4',
        ),
        'marker_name': 'plus_sign',
    }

    vgf = Path(__file__).parent / 'data' / 'watch.vgf'

    v = VectorGraphicFile(vgf)

    decoded_watch = v.elements[0].__dict__

    np.testing.assert_equal(decoded_watch, expected_watch)


def test_wind_vectors():
    """Test decoding of wind vector elements."""
    expected_direction = [185.0, 78.86, 227.17]
    expected_speed = [25.0, 100.0, 0.0]
    expected_lat = [38.06, 38.57, 42.39]
    expected_lon = [-106.31, -100.48, -98.87]

    vgf = Path(__file__).parent / 'data' / 'misc.vgf'

    v = VectorGraphicFile(vgf)

    wind = v.filter_elements(vg_class=6)

    decoded_direction = [e.direction for e in wind]
    decoded_speed = [e.speed for e in wind]
    decoded_lat = [e.lat for e in wind]
    decoded_lon = [e.lon for e in wind]

    assert decoded_direction == expected_direction
    assert decoded_speed == expected_speed
    assert decoded_lat == expected_lat
    assert decoded_lon == expected_lon
