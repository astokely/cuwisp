from distutils.core import setup, Extension

cparse_module = Extension(
    '_cparse',
    sources = ['cparse_wrap.c', 'cparse.c']
)

setup(
    version = '0.1',
    author = 'Andy Stokely',
    ext_modules = [cparse_module],
    py_modules = ['cparse']
)
    
