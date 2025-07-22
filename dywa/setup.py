#!/usr/bin/env python3

import sys
import os
from pathlib import Path
from setuptools import setup
# from cmake_build_extension import (CMakeExtension, BuildExtension)
from torch.utils.cpp_extension import (BuildExtension, CUDAExtension)

# NOTE: we'll override `ext_modules` and `cmdclass`
# later when we implement pybind11 cxx sub-modules.
# for eigen:
# wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip && unzip eigen-3.4.0.zip && rm -f eigen-3.4.0.zip
ext_modules = [
    CUDAExtension('cxx.ur5_kin_cuda', [
        'c_src/ur5_kin_cuda.cpp',
        'c_src/ur5_kin_cuda_kernel.cu',
    ]),
    CUDAExtension('cxx.franka_kin_cuda', [
        'c_src/franka_kin_cuda.cpp',
        'c_src/franka_kin_cuda_kernel.cu',
    ],
        include_dirs=['/tmp/eigen-3.4.0/'],
        extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O3']}
    )
]
cmdclass = {
    'build_ext': BuildExtension
}


if __name__ == '__main__':
    setup(name='dywa',
          use_scm_version=dict(
              root='..',
              relative_to=__file__,
              version_scheme='no-guess-dev'
          ),
          ext_modules=ext_modules,
          cmdclass=cmdclass,
          scripts=[]
          )
