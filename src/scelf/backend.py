"""Backend selection. A missing/incompatible optional extension falls back in auto mode."""
from importlib import import_module

_native = None
_native_error = None
_checked = False


def _load_cpp():
    global _native, _native_error, _checked
    if not _checked:
        try:
            _native = import_module('._cpp', __package__)
        except (ImportError, OSError) as exc:
            _native_error = str(exc)
        _checked = True
    return _native


def resolve_backend(name):
    if name not in ('auto', 'numba', 'cpp'):
        raise ValueError("backend must be 'auto', 'numba', or 'cpp'")
    if name == 'numba':
        return 'numba', None
    native = _load_cpp()
    if native is not None:
        return 'cpp', native
    if name == 'cpp':
        raise ImportError(f"C++ backend is unavailable: {_native_error}. Build the extension or use backend='numba'.")
    return 'numba', None


def backend_info():
    """Report actual extension availability, including the reason for fallback."""
    native = _load_cpp()
    return {'cpp_available': native is not None,
            'auto_backend': 'cpp' if native is not None else 'numba',
            'cpp_error': _native_error,
            'blas_provider': native.blas_provider if native is not None else None,
            'fastmath': False}
