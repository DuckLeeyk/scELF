"""Optional C++ extension: auto builds when possible, 1 requires it, 0 skips it."""
import os
from pathlib import Path
import subprocess
import shutil
import sys
import sysconfig

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11


class NativeBuild(build_ext):
    def build_extensions(self):
        portable = os.environ.get('SCELF_ZIG')
        if portable:
            if os.name != 'nt':
                raise RuntimeError('SCELF_ZIG currently supports Windows only; use the standard compiler elsewhere.')
            for ext in self.extensions:
                target = Path(self.get_ext_fullpath(ext.name)).resolve()
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = Path(self.build_temp).resolve() / ext.name.replace('.', '_')
                temporary.mkdir(parents=True, exist_ok=True)
                compiled = temporary / target.name
                libname = f'python{sys.version_info.major}{sys.version_info.minor}.lib'
                library = Path(sys.base_prefix) / 'libs' / libname
                args = [portable, 'c++', '-O3', '-std=c++17', '-shared', '-fno-fast-math', '-ffp-contract=off',
                        '-DNDEBUG', '-I' + sysconfig.get_path('include')]
                args.extend('-I' + str(p) for p in ext.include_dirs)
                args.extend(ext.sources)
                # Keep compiler sidecars (.pdb/.lib) outside the package/wheel.
                args.extend([str(library), '-o', str(compiled)])
                subprocess.run(args, check=True)
                shutil.copy2(compiled, target)
            return
        for ext in self.extensions:
            ext.extra_compile_args = ['/O2', '/std:c++17', '/EHsc', '/fp:precise'] if self.compiler.compiler_type == 'msvc' else ['-O3', '-std=c++17', '-fno-fast-math', '-ffp-contract=off']
        super().build_extensions()


mode = os.environ.get('SCELF_BUILD_CPP', 'auto')
if mode not in ('0', '1', 'auto'):
    raise RuntimeError('SCELF_BUILD_CPP must be 0, 1, or auto')
extensions = [] if mode == '0' else [Extension(
    'scelf._cpp', sources=['src/scelf/native/kernels.cpp'],
    include_dirs=[pybind11.get_include()], language='c++', optional=(mode == 'auto'),
)]
setup(ext_modules=extensions, cmdclass={'build_ext': NativeBuild})
