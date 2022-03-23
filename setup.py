import os
import subprocess
import sys

from setuptools import (
    setup,
    find_packages,
)

try:
    import Cython
    import numpy as np
except ModuleNotFoundError:
    subprocess.call(['pip', 'install', 'Cython'])
    subprocess.call(['pip', 'install', 'numpy==1.20.3'])

from Cython.Build import cythonize
import numpy as np

if 'install' in sys.argv:
    subprocess.call(
        ['conda', 'install', '-c', 'nvidia', 'cudatoolkit=11.4.1']
    )
    '''
    Make sure the cuda and cudatoolkit versions are the same.
    Available cudatoolkit versions are:
                11.6.0
                11.5.1
                11.5.0
                11.4.1
                11.4.0
                11.3.1
                11.2.2
                11.2.1
                11.2.0
                11.1.11
                11.0.3
                10.2.89
    '''
    subprocess.call(
        ['conda', 'install', '-c', 'conda-forge', 'cupy', 'cudatoolkit=11.4.1', 'numpy=1.20.3']
    )
    subprocess.call(
        ['conda', 'install', '-c', 'conda-forge', 'mdtraj']
    )

setup(
    author='Andy Stokely',
    email='amstokely@ucsd.edu',
    name='cuwisp',
    install_requires=[
        "numpy==1.20.3",
        "scipy",
        "numba",
        "sklearn",
        "pytest",
        "cython",
        "data-science-types",
        "abserdes",
        "colour",
    ],
    platforms=['Linux',
        'Unix', ],
    python_requires="3.9",
    ext_modules=cythonize(["cuwisp/*.pyx"]),
    include_dirs=[np.get_include()],
    py_modules=["cuwisp/cparse/cparse", "cuwisp/cfrechet/cfrechet"],
    packages=find_packages() + [''],
    zip_safe=False,
    package_data={'': [
        'cuwisp/cparse/_cparse.cpython-39-x86_64-linux-gnu.so',
        'cuwisp/cfrechet/_cfrechet.cpython-39-x86_64-linux-gnu.so',
        'cuwisp/bin/catdcd',
        'cuwisp/bin/libexpat.so.0',
    ]},
)

if 'install' in sys.argv:
    subprocess.call(
        ['pip', 'install', 'cupy']
    )
