import inspect
import subprocess
from distutils.core import Extension
from distutils.core import setup
import shutil
import os

cparse_dir_path = (
    f'{inspect.stack()[0][1].replace("setup.py", "")[:-1]}'
)


for f in os.listdir(cparse_dir_path):
    if 'cparse.cpython' in f:
        os.remove(f'{cparse_dir_path}/{f}')
    if 'cparse_wrap.c' in f:
        os.remove(f'{cparse_dir_path}/{f}')
    if 'cparse.py' in f:
        os.remove(f'{cparse_dir_path}/{f}')
    if f == 'build':
        shutil.rmtree(f'{cparse_dir_path}/{f}')
subprocess.call(['conda', 'install', 'swig'])
subprocess.call(
    [
        'swig',
        '-python',
        f'{cparse_dir_path}/cparse.i',
    ]
)
cparse_module = Extension(
    f'{cparse_dir_path}/_cparse',
    sources=[
        f'{cparse_dir_path}/cparse.c',
        f'{cparse_dir_path}/cparse_wrap.c'
    ]
)

setup(
    version='0.1',
    author='Andy Stokely',
    ext_modules=[cparse_module],
    py_modules=['cparse']
)
