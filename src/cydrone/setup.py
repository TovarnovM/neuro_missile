# python setup.py build_ext --inplace
# pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
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

compiler_directives = {"language_level": 3, "embedsignature": True, "boundscheck": False, "wraparound": False, "cdivision": True, 'nonecheck': False, 'initializedcheck':False}
ext = Extension("drone", ["drone.pyx"], language="c++", include_dirs=[np.get_include()])
setup(
    name='Hello cydrone app',
    ext_modules=cythonize(ext, compiler_directives=compiler_directives, annotate=True),
    zip_safe=False
)