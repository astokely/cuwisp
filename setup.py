import os
import re
import subprocess
import sys
from typing import Optional

from setuptools import find_packages
from setuptools import setup


class CudaDriverNotFound(Exception):

    def __init__(
            self,
            message=(
                    f"\n\nNVIDIA-SMI has failed because "
                    f"it couldn't communicate with the \n"
                    f"NVIDIA driver. Make sure that the "
                    f"latest NVIDIA driver is installed and running."
            )
    ):
        self.message = message
        super().__init__(self.message)


def get_cuda_version():
    p = subprocess.Popen(
        'nvidia-smi', stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, shell=True
    )
    output, err = p.communicate()
    if not len(output):
        raise CudaDriverNotFound()
    version = re.findall(
        '\d+\.\d+',
        output.decode('utf-8').split('CUDA')[-1]
    )[0]
    return version


def get_python_version() -> str:
    return (
        re.findall('\d\.\d', sys.version).pop(0)
    )


def get_binary_fname(
        base_binary_fname: str,
        version_delimiter: Optional[str] = 'VERSION'
) -> str:
    python_version = get_python_version()
    binary_fname = base_binary_fname.replace(
        version_delimiter,
        python_version.replace('.', '')
    )
    return binary_fname


def get_cparse_binary_fname():
    fname = get_binary_fname(
        base_binary_fname=
        '_cparse.cpython-VERSION-x86_64-linux-gnu.so'
    )
    return fname


def get_cudatoolkit_version():
    cuda_version = get_cuda_version()
    cudatoolkit_version = f'cudatoolkit={cuda_version}'
    return cudatoolkit_version


try:
    import Cython
    import numpy as np
except ModuleNotFoundError:
    subprocess.call(['pip', 'install', 'Cython'])
    subprocess.call(['pip', 'install', 'numpy==1.20.3'])

from Cython.Build import cythonize
import numpy as np

if 'build_ext' in sys.argv:
    subprocess.call([
        'python',
        f'{os.getcwd()}/cuwisp/cparse/setup.py',
        'build_ext',
        '--inplace'
    ])

if 'install' in sys.argv:
    cudatoolkit_version = get_cudatoolkit_version()
    subprocess.call(
        [
            'conda', 'install', '-c',
            'nvidia', f'{cudatoolkit_version}'
        ]
    )
    subprocess.call(
        [
            'conda', 'install', '-c',
            'conda-forge', 'cupy',
            f'{cudatoolkit_version}', 'numpy=1.20.3'
        ]
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
    python_requires="<=3.8",
    ext_modules=cythonize(["cuwisp/*.pyx"]),
    include_dirs=[np.get_include()],
    py_modules=["cuwisp/cparse/cparse", "cuwisp/parser/parser"],
    packages=find_packages() + [''],
    zip_safe=False,
    package_data={
        '': [
            f'cuwisp/cparse/{get_cparse_binary_fname()}',
            'cuwisp/bin/catdcd',
            'cuwisp/bin/libexpat.so.0',
            'cuwisp/parser/_parser.so'
        ]
    },
)

if 'install' in sys.argv:
    subprocess.call(
        ['pip', 'install', 'cupy']
    )
