from setuptools import setup
from setuptools import setup, find_packages, Extension
import sys
import subprocess
import fileinput
import re

try: 
	import Cython
except ModuleNotFoundError:
	subprocess.call(['pip', 'install', 'Cython'])

from Cython.Build import cythonize 


setup(
	author='Andy Stokely',
	email='amstokely@ucsd.edu',
	name='cuwisp',
    install_requires=[
		"numpy==1.19.5",
		"numba",
		"pytest", 
		"mdtraj", 
		"cython",
		"data-science-types",
		"abserdes",
		"matplotlib",
		"colour",
	],              
    platforms=['Linux',
                'Unix',],
    python_requires=">=3.8",          
    ext_modules = cythonize("cuwisp/calccom.pyx"),
	py_modules = ["cuwisp/cparse/cparse"],
    packages=find_packages() + [''],
	package_data={'': ['cuwisp/cparse/_cparse.cpython-38-x86_64-linux-gnu.so']},
)

if 'install' in sys.argv:
	subprocess.call(['conda', 'install', '-c', 'anaconda', 'cudatoolkit'])
	
