#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from setuptools import setup, find_packages, Extension
import sys


if sys.version_info < (3, 5):
    sys.exit('Sorry, Python >=3.5 is required for fairseq.')


if sys.platform == 'darwin':
    extra_compile_args = ['-stdlib=libc++', '-O3']
else:
    extra_compile_args = ['-std=c++11', '-O3']


class NumpyExtension(Extension):
    """Source: https://stackoverflow.com/a/54128391"""

    def __init__(self, *args, **kwargs):
        self.__include_dirs = []
        super().__init__(*args, **kwargs)

    @property
    def include_dirs(self):
        import numpy
        return self.__include_dirs + [numpy.get_include()]

    @include_dirs.setter
    def include_dirs(self, dirs):
        self.__include_dirs = dirs


extensions = [
    NumpyExtension(
        'fairseq.data.data_utils_fast',
        sources=['fairseq/data/data_utils_fast.pyx'],
        language='c++',
        extra_compile_args=extra_compile_args,
    ),
]


cmdclass = {}


try:
    # torch is not available when generating docs
    from torch.utils import cpp_extension
    cmdclass['build_ext'] = cpp_extension.BuildExtension
except ImportError:
    pass


if 'READTHEDOCS' in os.environ:
    # don't build extensions when generating docs
    extensions = []
    if 'build_ext' in cmdclass:
        del cmdclass['build_ext']

    # use CPU build of PyTorch
    dependency_links = [
        'https://download.pytorch.org/whl/cpu/torch-1.3.0%2Bcpu-cp36-cp36m-linux_x86_64.whl'
    ]
else:
    dependency_links = []


if 'clean' in sys.argv[1:]:
    # Source: https://bit.ly/2NLVsgE
    print("deleting Cython files...")
    import subprocess
    subprocess.run(['rm -f fairseq/*.so fairseq/**/*.so'], shell=True)


setup(
    name='fairseq',
    version='0.9.0',
    setup_requires=[
        'cython',
        'numpy',
        'setuptools>=18.0',
    ],
    install_requires=[
        'cffi',
        'cython',
        'numpy',
        'regex',
        'torch',
        'tqdm',
    ],
    dependency_links=dependency_links,
    packages=find_packages(exclude=['scripts', 'tests']),
    ext_modules=extensions,
    cmdclass=cmdclass,
    zip_safe=False,
)
