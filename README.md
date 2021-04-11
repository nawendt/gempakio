# gempakIO

gempakIO is a pure-Python package that will read and decode GEMPAK grid, sounding, and surface data. Because Python is platform independent, this brings the ability to read GEMPAK data in Windows much more easily.

### How It Works
gempakIO uses three classes (`GempakGrid`, `GempakSounding`, and `GempakSurface`) to read each respective data type. The package uses the same logic to decode and unpack the binary into data that you can use. Because of how the code is structured, data from the GEMPAK files is lazily loaded. You only do I/O on the data that you select and avoid as much unnecessary loading as possible. For more information, see the `GempakGrid.gdxarray`, `GempakSounding.snxarray`, `GempakSurface.sfjson` methods and check out the notebook with examples in it.

### Things Not Currently Implemented
*  GEMPAK grids can be packed using GRIB2 compression. These files cannot be decoded yet, but plans are in place to add that functionality.
*  GEMPAK grids packed with the NMC method cannot be read. I have not found a file to test in the wild so this may not get added.
*  GEMPAK had conversion methods for floating point number representations (e.g., IBM, IEEE, etc.). This package assumes IEEE. As it is relatively unlikely that there are much data not using IEEE floats, there is no plan to add conversions from other formats unless the need arises.

### Things Implemented With Limited Testing
*  Climate surface file type (see [GEMPAK Surface Library](https://github.com/Unidata/gempak/blob/master/gempak/txt/gemlib/sflib.txt) documentation). This is another situation where I have no files to test.
