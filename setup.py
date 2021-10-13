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
    subprocess.call(['pip', 'install', 'numpy==1.19.5'])

from Cython.Build import cythonize
import numpy as np

if 'install' in sys.argv:
    subprocess.call(
        ['conda', 'install', '-c', 'anaconda', 'cudatoolkit']
    )

setup(
    author='Andy Stokely',
    email='amstokely@ucsd.edu',
    name='cuwisp',
    install_requires=[
        "numpy==1.19.5",
        "scipy",
        "numba",
        "sklearn",
        "pytest",
        "mdtraj",
        "cython",
        "data-science-types",
        "abserdes",
        "colour",
    ],
    platforms=['Linux',
        'Unix', ],
    python_requires=">=3.8",
    ext_modules=cythonize(["cuwisp/*.pyx"]),
    include_dirs=[np.get_include()],
    py_modules=["cuwisp/cparse/cparse", "cuwisp/cfrechet/cfrechet"],
    packages=find_packages() + [''],
    zip_safe=False,
    package_data={'': [
        'cuwisp/cparse/_cparse.cpython-38-x86_64-linux-gnu.so',
        'cuwisp/cfrechet/_cfrechet.cpython-38-x86_64-linux-gnu.so',
        'cuwisp/bin/catdcd',
        'cuwisp/bin/libexpat.so.0',
    ]},
)

if 'install' in sys.argv:
    subprocess.call(
        ['pip', 'install', 'cupy']
    )
