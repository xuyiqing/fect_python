from setuptools import setup, Extension
import sys

try:
    import pybind11
    PYBIND11_INCLUDE = pybind11.get_include()
except Exception:  # fall back if not present at import time
    PYBIND11_INCLUDE = None


def get_ext_modules():
    # Build C++ extensions: _fe (existing) and _ife (R-aligned IFE)
    include_dirs = []
    if PYBIND11_INCLUDE is not None:
        include_dirs.append(PYBIND11_INCLUDE)
    extra_compile_args = ["-O3", "-std=c++17"]
    if sys.platform == "darwin":
        extra_compile_args.append("-mmacosx-version-min=10.14")
    ext_fe = Extension(
        name="fect._fe",
        sources=["src/fect/_fe.cpp"],
        include_dirs=include_dirs,
        language="c++",
        extra_compile_args=extra_compile_args,
    )
    ext_ife = Extension(
        name="fect._ife",
        sources=["src/fect/_ife.cpp"],
        include_dirs=include_dirs,
        language="c++",
        extra_compile_args=extra_compile_args,
    )
    return [ext_fe, ext_ife]

if __name__ == "__main__":
    # setup.cfg carries most metadata; we only add the optional extension here
    try:
        ext_modules = get_ext_modules()
    except Exception:
        ext_modules = []
    setup(ext_modules=ext_modules)
