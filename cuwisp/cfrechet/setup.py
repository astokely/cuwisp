import distutils.core
import numpy as np

frechet_module = distutils.core.Extension(
    '_cfrechet',
    sources=['cfrechet_wrap.c', 'cfrechet.c']
)

distutils.core.setup(
    version='1.0',
    author='Andy Stokely',
    ext_modules=[frechet_module],
    include_dirs=[np.get_include()],
    py_modules=['cfrechet']
)
