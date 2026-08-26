#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>

void mpf_transform(int64_t, int64_t, int64_t, int64_t, int64_t);

static PyObject *transform(PyObject *self, PyObject *args) {
    PyObject *value;
    int mode;
    unsigned long long map_address;
    if (!PyArg_ParseTuple(args, "OiK", &value, &mode, &map_address))
        return NULL;
    if (!PyUnicode_Check(value) || !PyUnicode_IS_ASCII(value)) {
        PyErr_SetString(PyExc_TypeError, "value must be an ASCII str");
        return NULL;
    }
    Py_ssize_t size;
    const char *source = PyUnicode_AsUTF8AndSize(value, &size);
    if (source == NULL)
        return NULL;
    PyObject *result = PyUnicode_New(size, 127);
    if (result == NULL)
        return NULL;
    if (size)
        mpf_transform((int64_t)(uintptr_t)source,
                      (int64_t)(uintptr_t)PyUnicode_1BYTE_DATA(result),
                      (int64_t)map_address, (int64_t)size, (int64_t)mode);
    return result;
}

static PyMethodDef methods[] = {
    {"transform", transform, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_native", NULL, -1, methods,
};

PyMODINIT_FUNC PyInit__native(void) {
    return PyModule_Create(&module);
}
