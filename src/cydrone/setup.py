# python setup.py build_ext --inplace
import os
import glob
import shutil
import numpy as np

try:
    shutil.rmtree('build/')
    for cf in glob.glob('*.c'):
        os.remove(cf)
    for cf in glob.glob('*.cpp'):
        os.remove(cf)
    for cf in glob.glob('*.html'):
        os.remove(cf)
    for cf in glob.glob('*.pyd'):
        os.remove(cf)
except:
    pass


from setuptools import setup, Extension
from Cython.Build import cythonize

compiler_directives = {"language_level": 3, "embedsignature": True, "boundscheck": False, "wraparound": False, "cdivision": True, 'nonecheck': False}
ext = Extension("drone", ["drone.pyx"], language="c++", include_dirs=[np.get_include()])
setup(
    name='Hello cydrone app',
    ext_modules=cythonize(ext, compiler_directives=compiler_directives, annotate=True),
    zip_safe=False
)