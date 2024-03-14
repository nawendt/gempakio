# gempakIO

gempakIO is a pure-Python package that will read and write GEMPAK grid, sounding, and surface data. Because Python is platform independent, this brings the ability to read GEMPAK data in Windows much more easily.

### How It Works
gempakIO uses three decoding classes (`GempakGrid`, `GempakSounding`, and `GempakSurface`) to read each respective data type. Three other classes (`GridFile`, `SoundingFile`, and `SurfaceFile`) are used to write GEMPAK data. Because of how the code is structured, data from the GEMPAK files is lazily loaded. You only do I/O on the data that you select and avoid as much unnecessary loading as possible. For more information on reading data, see the documentation for the `GempakGrid.gdxarray`, `GempakSounding.snxarray`, `GempakSurface.sfjson` methods. More information about writing data can be found in the documentationn for `GridFile.to_gempak`, `SoundingFile.to_gempak`, and `SurfaceFile.to_gempak`. There is a notebook with examples in it to help you get started.

### Things Not Currently Implemented
*  GEMPAK grids can be packed using GRIB2 compression. These files cannot be decoded yet, but plans are in place to add that functionality.
*  GEMPAK grids packed with the NMC method cannot be read. I have not found a file to test in the wild so this may not get added.
*  GEMPAK had conversion methods for floating point number representations (e.g., IBM, IEEE, etc.). This package assumes IEEE. As it is relatively unlikely that there are much data not using IEEE floats, there is no plan to add conversions from other formats unless the need arises.
*  GEMPAK sounding and surface files can have their parameter data packed/compressed, but this is not currently implemented. GEMPAK grids do have basic GRIB packing by default, but GRIB2 packing is not implemented at this time.