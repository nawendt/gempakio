#include <math.h>
#include <Python.h>
#include <numpy/arrayobject.h>

#define GRAVITY 9.80616;
#define MISSING -9999.0;
#define R_D 287.04;

static char interp_logp_height_doc[] = \
    "Interpolate height linearly with respect to log p.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "sounding : dict\n"
    "    Sounding dictionary structure.\n"
    "\n"
    "Notes\n"
    "-----\n"
    "See GEMPAK function MR_INTZ.";

static PyObject *interp_logp_height(PyObject *self, PyObject *args) {
    size_t size;
    int idx;
    int maxlev;
    double missing = MISSING;

    int smooth_level;
    int xdim;
    int ydim;
    double *xptr;
    double *yptr;
    double *xtptr;
    double *ytptr;
    double *xcptr;
    double *ycptr;
    double chord;
    double density;
    double curve_scale;
    double t;
    size_t i;
    size_t j;
    size_t k;
    size_t m;
    size_t nout;
    uint32_t *nptr;

    npy_intp xlen;
    npy_intp ylen;
    npy_intp xshape;
    npy_intp yshape;
    npy_intp nshape;
    PyArrayObject *x;
    PyArrayObject *y;
    PyArrayObject *xtemp;
    PyArrayObject *ytemp;
    PyArrayObject *npts;
    PyArrayObject *xcurve;
    PyArrayObject *ycurve;
    PyDictObject *sounding;
    PyObject *out;
    PyObject *err;
    PyObject *hght;

    if (!PyArg_ParseTuple(
            args, "O!|d",
            &PyDict_Type, &sounding,
            &missing
            )
        ) {
        PyErr_SetString(PyExc_TypeError, "Check input types.");
        return NULL;
    }

    size = PyList_Size(PyDict_GetItemWithError(sounding, PyUnicode_FromString("HGHT")));
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_KeyError, "HGHT not in sounding object.");
        err = PyErr_GetRaisedException();
        return NULL;
    }

    idx = -1;
    maxlev = -1;
    while (size + idx != 0) {
        hght = PyList_GetItem(PyDict_GetItem(sounding, PyUnicode_FromString("HGHT")), idx);
    }
    

    xdim = PyArray_NDIM(x);
    if (xdim != 1) {
        PyErr_SetString(PyExc_ValueError, "x input must be one-dimensional.");
        return NULL;
    }

    ydim = PyArray_NDIM(x);
    if (ydim != 1) {
        PyErr_SetString(PyExc_ValueError, "y input must be one-dimensional.");
        return NULL;
    }

    xlen = PyArray_DIM(x, 0);
    ylen = PyArray_DIM(x, 0);
    if (xlen != ylen) {
        PyErr_Format(PyExc_ValueError,
                     "x an y input should be same length. Got (%d,) and (%d,).",
                     xlen,
                     ylen);
        return NULL;
    }

    x = PyArray_CastToType(x, PyArray_DescrFromType(NPY_DOUBLE), 0);
    if(PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Unable to cast x to double.");
        return NULL;   
    }

    y = PyArray_CastToType(y, PyArray_DescrFromType(NPY_DOUBLE), 0);
    if(PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Unable to cast y to double.");
        return NULL;   
    }

    if (smooth_level <= 1) {
        density = 1;
    } else {
        density = 5;
    }

    xshape = xlen + 2;
    yshape = ylen + 2;
    xtemp = PyArray_Zeros(1, &yshape, PyArray_DescrFromType(NPY_DOUBLE), 0);
    if(PyErr_Occurred()) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate xtemp array.");
        return NULL;   
    }
    ytemp = PyArray_Zeros(1, &xshape, PyArray_DescrFromType(NPY_DOUBLE), 0);
    if(PyErr_Occurred()) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate ytemp array.");
        return NULL;   
    }

    xptr = (double *)PyArray_DATA(x);
    yptr = (double *)PyArray_DATA(y);
    xtptr = (double *)PyArray_DATA(xtemp);
    ytptr = (double *)PyArray_DATA(ytemp);
    if (xptr == NULL || yptr == NULL || xtptr == NULL || ytptr == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create input array pointers.");
        return NULL;   
    }

    for (i = 0; i < xlen; i++) {
        xtptr[i + 1] = xptr[i];
        ytptr[i + 1] = yptr[i];
    }

    nshape = xlen - 1;
    npts = PyArray_Zeros(1, &nshape, PyArray_DescrFromType(NPY_UINT32), 0);
    if(PyErr_Occurred()) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate npts array.");
        return NULL;   
    }

    nptr = (uint32_t *)PyArray_DATA(npts);
    if (nptr == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create npts array pointer.");
        return NULL;   
    }

    out = PyTuple_New(2);
    if(PyErr_Occurred()) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate output tuple.");
        return NULL;   
    }
    PyTuple_SetItem(out, 0, PyArray_Return(xcurve));
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to set output tuple item.");
        return NULL;   
    }
    PyTuple_SetItem(out, 1, PyArray_Return(ycurve));
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to set output tuple item.");
        return NULL;   
    }

    Py_DECREF(sounding);
}

static PyMethodDef c_gemlib_functions[] = {
    // {"parametric_curve", parametric_curve, METH_VARARGS, parametric_curve_doc},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef c_gemlib_module = {
    PyModuleDef_HEAD_INIT,
    "c_gemlib",
    "C GEMPAK Calculation Module",
    -1,
    c_gemlib_functions
};

PyMODINIT_FUNC PyInit_c_utils(void) {
    import_array();
    return PyModule_Create(&c_gemlib_module);
}