# Copyright (c) 2022 Nathan Wendt.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Setup script for installing gempakIO."""

import numpy as np
from setuptools import Extension, setup

if __name__ == '__main__':
    c_decode = Extension(
        'gempakio.decode.c_decode',
        ['src/gempakio/decode/c_decode.c'],
        include_dirs=[np.get_include()]
    )

    c_gemlib = Extension(
        'gempakio.c_gemlib',
        ['src/gempakio/c_gemlib.c'],
        include_dirs=[np.get_include()]
    )

    setup(
        ext_modules=[c_decode]
    )
