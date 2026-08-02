"""Build and load the Mojo byte kernels."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_PYFASTX_LIB") or os.path.join(ROOT, "dist", "libmojo-pyfastx.so")
I = ctypes.c_int64

_complement = bytearray(range(256))
for _left, _right in ((b"AT", b"TA"), (b"CG", b"GC"), (b"URYSWKMBVDH", b"AYRSWMKVBHD")):
    for _source, _target in zip(_left + _left.lower(), _right + _right.lower()):
        _complement[_source] = _target
_COMPLEMENT = np.frombuffer(bytes(_complement), dtype=np.uint8)
_IDENTITY = np.arange(256, dtype=np.uint8)


def build() -> str:
    sources = [os.path.join(ROOT, "src", "capi.mojo")]
    if os.path.exists(LIB) and os.path.getmtime(LIB) >= max(map(os.path.getmtime, sources)):
        return LIB
    proc = subprocess.run(["bash", os.path.join(ROOT, "build", "build.sh")], cwd=ROOT,
                          text=True, capture_output=True, timeout=1800)
    if proc.returncode or not os.path.exists(LIB):
        raise RuntimeError((proc.stderr or proc.stdout).strip())
    return LIB


_handle: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _handle
    if _handle is None:
        _handle = ctypes.CDLL(build())
        _handle.mpf_transform.argtypes = [I, I, I, I, I]
        _handle.mpf_transform.restype = None
        _handle.mpf_count_bytes.argtypes = [I, I, I]
        _handle.mpf_count_bytes.restype = None
    return _handle


def bytes_array(value: str | bytes) -> np.ndarray:
    raw = value.encode() if isinstance(value, str) else value
    return np.frombuffer(raw, dtype=np.uint8).copy()


_unicode_utf8 = ctypes.pythonapi.PyUnicode_AsUTF8AndSize
_unicode_utf8.argtypes = [ctypes.py_object, ctypes.POINTER(ctypes.c_ssize_t)]
_unicode_utf8.restype = ctypes.c_void_p
_bytes_data = ctypes.pythonapi.PyBytes_AsString
_bytes_data.argtypes = [ctypes.py_object]
_bytes_data.restype = ctypes.c_void_p


def _address_and_size(value: str | bytes) -> tuple[int, int]:
    if isinstance(value, bytes):
        return int(_bytes_data(value)), len(value)
    size = ctypes.c_ssize_t()
    address = _unicode_utf8(value, ctypes.byref(size))
    return int(address), size.value


def transform(value: str, mode: int) -> str:
    if not isinstance(value, str):
        raise TypeError("value must be a str")
    if mode not in (0, 1, 2):
        raise ValueError("mode must be 0 (complement), 1 (reverse), or 2 (reverse-complement)")
    # Keep value, dst, and mapping strongly referenced until the C call returns.
    src, size = _address_and_size(value)
    dst = ctypes.create_string_buffer(size)
    if size:
        mapping = _COMPLEMENT if mode != 1 else _IDENTITY
        lib().mpf_transform(src, ctypes.addressof(dst), mapping.ctypes.data, size, mode)
    return bytes(dst).decode()


def histogram(value: str | bytes) -> np.ndarray:
    if not isinstance(value, (str, bytes)):
        raise TypeError("value must be str or bytes")
    # counts is a contiguous, writable int64[256], exactly matching IntPtr.
    src, size = _address_and_size(value)
    counts = np.zeros(256, dtype=np.int64)
    if size:
        lib().mpf_count_bytes(src, counts.ctypes.data, size)
    return counts
