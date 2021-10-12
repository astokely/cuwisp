import os
import subprocess
import sys

from setuptools import (
    setup,
    find_packages,
)

try:
    import Cython
    import numpy
except ModuleNotFoundError:
    subprocess.call(['pip', 'install', 'Cython'])
    subprocess.call(['pip', 'install', 'numpy==1.19.5'])

from Cython.Build import cythonize
import numpy as np

setup(
    author='Andy Stokely',
    email='amstokely@ucsd.edu',
    name='cuwisp',
    install_requires=[
        "scipy",
        "numpy==1.19.5",
        "numba",
        "pytest",
        "mdtraj",
        "cython",
        "data-science-types",
        "abserdes",
        "matplotlib",
        "colour",
        "cupy",
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
        ['conda', 'install', '-c', 'anaconda', 'cudatoolkit']
    )
    from cuwisp.nodes import Nodes

    module_path = (
        lambda
            mod_cls_fn_import: (os.path.dirname(
                sys.modules[
                    Nodes.__module__
                ].__file__
            ))
    )
    print(
        f'Add {module_path(Nodes)}/bin to LD_LIBRARY_PATH for DCD '
        f'support.'
    )
