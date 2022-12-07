# Copyright (c) 2022 Nathan Wendt.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Setup script for installing gempakIO."""

from setuptools import find_packages, setup

NAME = 'gempakio'
VERSION = '0.8.2'
DESCR = 'Read GEMPAK data with pure Python.'
URL = 'https://github.com/nawendt/gempakio'
REQUIRES = ['numpy', 'pandas', 'pyproj', 'xarray']
AUTHOR = 'Nathan Wendt'
EMAIL = 'nathan.wendt@noaa.gov'
LICENSE = 'BSD 3-clause'
PACKAGES = find_packages()
EXTRAS = {
    'examples': [
        'metpy',
        'cartopy',
    ],
    'lint': [
        'flake8'
        'pycodestyle',
        'pyflakes',
        'flake8-bugbear',
        'flake8-builtins',
        'flake8-comprehensions',
        'flake8-continuation',
        'flake8-copyright',
        'flake8-isort',
        'isort',
        'flake8-mutable',
        'flake8-pie',
        'flake8-print',
        'flake8-quotes',
        'flake8-requirements',
        'flake8-simplify',
        'flake8-docstrings',
        'pydocstyle',
    ],
    'test': ['pytest'],
}

if __name__ == '__main__':
    setup(install_requires=REQUIRES,
          packages=PACKAGES,
          zip_safe=True,
          name=NAME,
          version=VERSION,
          description=DESCR,
          author=AUTHOR,
          author_email=EMAIL,
          url=URL,
          license=LICENSE,
          extras_require=EXTRAS,
          )
