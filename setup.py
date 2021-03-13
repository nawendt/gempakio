from setuptools import setup, Extension, find_packages

NAME = 'gempakio'
VERSION = '0.1'
DESCR = 'Read GEMPAK data with pure Python.'
URL = 'https://github.com/nawendt/gempakio'
REQUIRES = ['pyproj', 'xarray']

AUTHOR = 'Nathan Wendt'
EMAIL = 'nathan.wendt@noaa.gov'

LICENSE = 'BSD 3-clause'

PACKAGES = find_packages()

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
          )
