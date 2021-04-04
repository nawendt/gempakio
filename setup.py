# Copyright (c) 2021 Nathan Wendt.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Setup script for installing gempakIO."""

from setuptools import find_packages, setup

NAME = 'gempakio'
VERSION = '0.5'
DESCR = 'Read GEMPAK data with pure Python.'
URL = 'https://github.com/nawendt/gempakio'
REQUIRES = ['pyproj', 'xarray']
AUTHOR = 'Nathan Wendt'
EMAIL = 'nathan.wendt@noaa.gov'
LICENSE = 'BSD 3-clause'
PACKAGES = find_packages()
EXTRAS = {
    'test': ['pytest']
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
