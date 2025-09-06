# Copyright (c) 2025 Nathan Wendt.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for encoding GEMPAK VGF files."""

from pathlib import Path
import tempfile

import numpy as np
import pytest

from gempakio.decode.vgf import VectorGraphicFile, VGClass, VGType
from gempakio.encode.vgf import (
    ColorCodes,
    Element,
    FontCodes,
    Front,
    FrontCodes,
    Line,
    LineBase,
    LineCodes,
    Marker,
    MarkerCodes,
    SpecialLine,
    SpecialLineCodes,
    SpecialSymbol,
    SpecialSymbolCodes,
    SpecialText,
    SpecialTextCodes,
    SymbolBase,
    TextBase,
    VGFile,
)


def test_element_attributes():
    """Test element attributes."""
    element = Element()

    with pytest.raises(ValueError):
        element.closed = 9

    with pytest.raises(ValueError):
        element.filled = -1

    with pytest.raises(ValueError):
        element.major_color = -1

    with pytest.raises(ValueError):
        element.minor_color = -1

    with pytest.raises(ValueError):
        element.group_type = -1

    with pytest.raises(ValueError):
        element.group_number = -1

    with pytest.raises(ValueError):
        element.smooth = 20

    with pytest.raises(ValueError):
        element.filled = 55

    with pytest.raises(AttributeError):
        element.vg_class = 77

    with pytest.raises(AttributeError):
        element.vg_type = 77

    with pytest.raises(AttributeError):
        element.delete = 77

    with pytest.raises(AttributeError):
        element.min_lat = 77

    with pytest.raises(AttributeError):
        element.min_lon = 77

    with pytest.raises(AttributeError):
        element.max_lat = 77

    with pytest.raises(AttributeError):
        element.max_lon = 77


def test_front_attributes():
    """Test front attributes."""
    front = Front([0, 1], [0, 1], FrontCodes.COLD_FRONT, 4)

    with pytest.raises(ValueError):
        front.front_code = -1

    with pytest.raises(ValueError):
        front.front_code = 1000

    with pytest.raises(ValueError):
        front.width = -1

    with pytest.raises(ValueError):
        front.width = 20

    with pytest.raises(ValueError):
        front.pip_direction = 0

    with pytest.raises(ValueError):
        front.pip_size = -1

    with pytest.raises(ValueError):
        front.pip_size = 550

    with pytest.raises(ValueError):
        front.pip_stroke = -1

    front.flip()
    assert front.pip_direction == -1

    front.width = 3
    assert front.front_code == 430

    front.front_code = 640
    assert front.width == 4


def test_front_write():
    """Test writing fronts to VGF."""
    lat = np.array([27.29, 28.28, 28.53, 27.93, 26.8, 26.03, 25.51, 25.47], dtype='float32')
    lon = np.array(
        [-82.44, -81.66, -80.99, -80.48, -79.89, -80.03, -80.45, -81.12], dtype='float32'
    )

    front = Front(lon, lat, FrontCodes.DRYLINE, 5)

    out = VGFile.from_elements(front)

    kwargs = {'dir': '.', 'suffix': '.vgf', 'delete': False}

    try:
        with tempfile.NamedTemporaryFile(**kwargs) as tmp:
            out.to_vgf(tmp.name)
            vgf = Path(tmp.name)

        in_vgf = VectorGraphicFile(vgf)
        test_front = in_vgf.get_fronts()[0]

        test_lat = test_front.lat
        test_lon = test_front.lon

        assert test_front.front_code == FrontCodes.DRYLINE
        assert test_front.vg_type == VGType.front
        assert test_front.vg_class == VGClass.fronts
        assert test_front.major_color == 5
        assert test_front.minor_color == 5
        assert test_front.closed == 0
        assert test_front.smooth == 2
        assert test_front.filled == 0
        assert test_front.record_size == 132
        assert test_front.group_type == 0
        assert test_front.group_number == 0
        assert test_front.width == 2
        assert test_front.pip_size == 100
        assert test_front.pip_direction == 1
        assert test_front.pip_stroke == 1
        assert test_front.label == 'SPC'
        assert test_front.max_lat == pytest.approx(lat.max())
        assert test_front.max_lon == pytest.approx(lon.max())
        assert test_front.min_lat == pytest.approx(lat.min())
        assert test_front.min_lon == pytest.approx(lon.min())
        np.testing.assert_allclose(test_lat, lat, rtol=1e-6, atol=0)
        np.testing.assert_allclose(test_lon, lon, rtol=1e-6, atol=0)
    finally:
        vgf.unlink()


def test_line_base_attributes():
    """Test line base attributes."""
    line = LineBase([0, 1], [0, 1], 1, 1, 0, 0, 0)

    with pytest.raises(ValueError):
        line.width = -1

    with pytest.raises(ValueError):
        line.width = 20


def test_line_attributes():
    """Test line attributes."""
    line = Line([0, 1], [0, 1], 1, LineCodes.DOTTED, 0)

    with pytest.raises(ValueError):
        line.line_type = 100

    with pytest.raises(ValueError):
        line.line_type = -1


def test_line_write():
    """Test writing lines to VGF."""
    lat = np.array([27.29, 28.28, 28.53, 27.93, 26.8, 26.03, 25.51, 25.47], dtype='float32')
    lon = np.array(
        [-82.44, -81.66, -80.99, -80.48, -79.89, -80.03, -80.45, -81.12], dtype='float32'
    )

    line = Line(lon, lat, 2, LineCodes.LONG_DASH, 0)

    out = VGFile.from_elements(line)

    kwargs = {'dir': '.', 'suffix': '.vgf', 'delete': False}

    try:
        with tempfile.NamedTemporaryFile(**kwargs) as tmp:
            out.to_vgf(tmp.name)
            vgf = Path(tmp.name)

        in_vgf = VectorGraphicFile(vgf)
        test_line = in_vgf.get_lines()[0]

        test_lat = test_line.lat
        test_lon = test_line.lon

        assert test_line.line_type == LineCodes.LONG_DASH
        assert test_line.vg_type == VGType.line
        assert test_line.vg_class == VGClass.lines
        assert test_line.major_color == 2
        assert test_line.minor_color == 2
        assert test_line.closed == 0
        assert test_line.smooth == 0
        assert test_line.filled == 0
        assert test_line.record_size == 124
        assert test_line.group_type == 0
        assert test_line.group_number == 0
        assert test_line.width == 2
        assert test_line.max_lat == pytest.approx(lat.max())
        assert test_line.max_lon == pytest.approx(lon.max())
        assert test_line.min_lat == pytest.approx(lat.min())
        assert test_line.min_lon == pytest.approx(lon.min())
        np.testing.assert_allclose(test_lat, lat, rtol=1e-6, atol=0)
        np.testing.assert_allclose(test_lon, lon, rtol=1e-6, atol=0)
    finally:
        vgf.unlink()


def test_special_line_attributes():
    """Test special line attributes."""
    line = SpecialLine([0, 1], [0, 1], 1, SpecialLineCodes.SCALLOP, 0)

    with pytest.raises(ValueError):
        line.line_type = -1

    with pytest.raises(ValueError):
        line.line_type = 30

    with pytest.raises(ValueError):
        line.direction = 50

    with pytest.raises(ValueError):
        line.size = -1

    with pytest.raises(ValueError):
        line.size = 500

    line.flip()
    assert line.direction == -1


def test_special_line_write():
    """Test writing special lines to VGF."""
    lat = np.array([27.29, 28.28, 28.53, 27.93, 26.8, 26.03, 25.51, 25.47], dtype='float32')
    lon = np.array(
        [-82.44, -81.66, -80.99, -80.48, -79.89, -80.03, -80.45, -81.12], dtype='float32'
    )

    line = SpecialLine(lon, lat, 2, SpecialLineCodes.SCALLOP, 0, smooth=2)

    out = VGFile.from_elements(line)

    kwargs = {'dir': '.', 'suffix': '.vgf', 'delete': False}

    try:
        with tempfile.NamedTemporaryFile(**kwargs) as tmp:
            out.to_vgf(tmp.name)
            vgf = Path(tmp.name)

        in_vgf = VectorGraphicFile(vgf)
        test_line = in_vgf.get_special_lines()[0]

        test_lat = test_line.lat
        test_lon = test_line.lon

        assert test_line.line_type == SpecialLineCodes.SCALLOP
        assert test_line.vg_type == VGType.special_line
        assert test_line.vg_class == VGClass.lines
        assert test_line.major_color == 2
        assert test_line.minor_color == 2
        assert test_line.closed == 0
        assert test_line.smooth == 2
        assert test_line.filled == 0
        assert test_line.record_size == 128
        assert test_line.group_type == 0
        assert test_line.group_number == 0
        assert test_line.width == 2
        assert test_line.direction == 1
        assert test_line.size == 1
        assert test_line.max_lat == pytest.approx(lat.max())
        assert test_line.max_lon == pytest.approx(lon.max())
        assert test_line.min_lat == pytest.approx(lat.min())
        assert test_line.min_lon == pytest.approx(lon.min())
        np.testing.assert_allclose(test_lat, lat, rtol=1e-6, atol=0)
        np.testing.assert_allclose(test_lon, lon, rtol=1e-6, atol=0)
    finally:
        vgf.unlink()


def test_text_base_attribute():
    """Test text base attributes."""
    text = TextBase(0, 0, 'TEST', 4, 14, FontCodes.HELVETICA, 1, 0, 0, 0, 0, 2)

    with pytest.raises(ValueError):
        text.text_color = 50

    with pytest.raises(ValueError):
        text.size = 50

    with pytest.raises(ValueError):
        text.font = 8

    with pytest.raises(ValueError):
        text.align = 8


def test_special_text_attributes():
    """Test special text attributes."""
    text = SpecialText(
        0, 0, 'TEST', 8, 18, SpecialTextCodes.GENERAL_TEXT, FontCodes.HELVETICA_BOLD
    )

    with pytest.raises(ValueError):
        text.text_type = 20

    with pytest.raises(ValueError):
        text.edgecolor = 50

    with pytest.raises(ValueError):
        text.facecolor = 50

    text.text_type = 7
    with pytest.raises(ValueError):
        text.turbulence_symbol = 15

    text.text_type = 12
    with pytest.raises(ValueError):
        text.turbulence_symbol = 15

    with pytest.raises(ValueError):
        text.text_flag = 8


def test_special_text_write():
    """Test writing special text to VGF."""
    lat = 38.701065
    lon = -98.326084

    text = SpecialText(
        lon,
        lat,
        'THIS IS A TEST!',
        8,
        18,
        SpecialTextCodes.GENERAL_TEXT,
        FontCodes.HELVETICA_BOLD,
    )

    out = VGFile.from_elements(text)

    kwargs = {'dir': '.', 'suffix': '.vgf', 'delete': False}

    try:
        with tempfile.NamedTemporaryFile(**kwargs) as tmp:
            out.to_vgf(tmp.name)
            vgf = Path(tmp.name)

        in_vgf = VectorGraphicFile(vgf)
        test_text = in_vgf.get_special_text()[0]

        assert test_text.text_type == SpecialTextCodes.GENERAL_TEXT
        assert test_text.vg_type == VGType.special_text
        assert test_text.vg_class == VGClass.text
        assert test_text.major_color == 8
        assert test_text.minor_color == 8
        assert test_text.closed == 0
        assert test_text.smooth == 0
        assert test_text.filled == 0
        assert test_text.record_size == 116
        assert test_text.group_type == 0
        assert test_text.group_number == 0
        assert test_text.width == 1
        assert test_text.text == 'THIS IS A TEST!'
        assert test_text.line_color == 8
        assert test_text.fill_color == 8
        assert test_text.font == FontCodes.HELVETICA_BOLD
        assert test_text.max_lat == pytest.approx(lat, rel=1e-4, abs=0)
        assert test_text.max_lon == pytest.approx(lon, rel=1e-4, abs=0)
        assert test_text.min_lat == pytest.approx(lat, rel=1e-4, abs=0)
        assert test_text.min_lon == pytest.approx(lon, rel=1e-4, abs=0)
        assert test_text.lat == pytest.approx(lat, rel=1e-4, abs=0)
        assert test_text.lon == pytest.approx(lon, rel=1e-4, abs=0)
    finally:
        vgf.unlink()


def test_symbol_base_attributes():
    """Test base symbol class attributes."""
    lat = 38.701065
    lon = -98.326084

    symbol = SymbolBase(lon, lat, SpecialSymbolCodes.CIRCLE_FILLED, ColorCodes.MAGENTA)

    with pytest.raises(ValueError):
        symbol.width = -1

    with pytest.raises(ValueError):
        symbol.width = 20

    with pytest.raises(ValueError):
        symbol.size = -1

    with pytest.raises(ValueError):
        symbol.size = 20

    with pytest.raises(TypeError):
        symbol.symbol_code = '10'


def test_special_symbol_attributes():
    """Test special symbol class attributes."""
    lat = 38.701065
    lon = -98.326084

    symbol = SpecialSymbol(lon, lat, SpecialSymbolCodes.CIRCLE_FILLED, ColorCodes.MAGENTA)

    with pytest.raises(ValueError):
        symbol.symbol_code = 42


def test_marker_attributes():
    """Test marker class attributes."""
    lat = 38.701065
    lon = -98.326084

    marker = Marker(lon, lat, MarkerCodes.ASTERISK, ColorCodes.MAGENTA)

    with pytest.raises(ValueError):
        marker.symbol_code = 25


def test_special_symbol_write():
    """Test writing special symbol to VGF."""
    lat = 38.701065
    lon = -98.326084

    symbol = SpecialSymbol(lon, lat, SpecialSymbolCodes.CIRCLE_FILLED, ColorCodes.MAGENTA)

    out = VGFile.from_elements(symbol)

    kwargs = {'dir': '.', 'suffix': '.vgf', 'delete': False}

    try:
        with tempfile.NamedTemporaryFile(**kwargs) as tmp:
            out.to_vgf(tmp.name)
            vgf = Path(tmp.name)

        in_vgf = VectorGraphicFile(vgf)
        test_symbol = in_vgf.get_special_symbols()[0]

        assert test_symbol.vg_type == VGType.special_symbol
        assert test_symbol.vg_class == VGClass.symbols
        assert test_symbol.symbol_code == SpecialSymbolCodes.CIRCLE_FILLED
        assert test_symbol.major_color == ColorCodes.MAGENTA
        assert test_symbol.minor_color == ColorCodes.MAGENTA
        assert test_symbol.closed == 0
        assert test_symbol.smooth == 0
        assert test_symbol.filled == 0
        assert test_symbol.record_size == 76
        assert test_symbol.group_type == 0
        assert test_symbol.group_number == 0
        assert test_symbol.width == 1
        assert test_symbol.size == 1
        assert test_symbol.max_lat == pytest.approx(lat, rel=1e-4, abs=0)
        assert test_symbol.max_lon == pytest.approx(lon, rel=1e-4, abs=0)
        assert test_symbol.min_lat == pytest.approx(lat, rel=1e-4, abs=0)
        assert test_symbol.min_lon == pytest.approx(lon, rel=1e-4, abs=0)
        assert test_symbol.lat == pytest.approx(lat, rel=1e-4, abs=0)
        assert test_symbol.lon == pytest.approx(lon, rel=1e-4, abs=0)
    finally:
        vgf.unlink()


def test_marker_write():
    """Test writing marker to VGF."""
    lat = 38.701065
    lon = -98.326084

    symbol = Marker(lon, lat, MarkerCodes.ASTERISK, ColorCodes.MAGENTA)

    out = VGFile.from_elements(symbol)

    kwargs = {'dir': '.', 'suffix': '.vgf', 'delete': False}

    try:
        with tempfile.NamedTemporaryFile(**kwargs) as tmp:
            out.to_vgf(tmp.name)
            vgf = Path(tmp.name)

        in_vgf = VectorGraphicFile(vgf)
        test_marker = in_vgf.get_markers()[0]

        assert test_marker.vg_type == VGType.marker
        assert test_marker.vg_class == VGClass.symbols
        assert test_marker.symbol_code == MarkerCodes.ASTERISK
        assert test_marker.major_color == ColorCodes.MAGENTA
        assert test_marker.minor_color == ColorCodes.MAGENTA
        assert test_marker.closed == 0
        assert test_marker.smooth == 0
        assert test_marker.filled == 0
        assert test_marker.record_size == 76
        assert test_marker.group_type == 0
        assert test_marker.group_number == 0
        assert test_marker.width == 1
        assert test_marker.size == 1
        assert test_marker.max_lat == pytest.approx(lat, rel=1e-4, abs=0)
        assert test_marker.max_lon == pytest.approx(lon, rel=1e-4, abs=0)
        assert test_marker.min_lat == pytest.approx(lat, rel=1e-4, abs=0)
        assert test_marker.min_lon == pytest.approx(lon, rel=1e-4, abs=0)
        assert test_marker.lat == pytest.approx(lat, rel=1e-4, abs=0)
        assert test_marker.lon == pytest.approx(lon, rel=1e-4, abs=0)
    finally:
        vgf.unlink()
