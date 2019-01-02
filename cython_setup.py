'''
How to use:
    Copy this line into terminal

    python cython_setup.py  build_ext --inplace

Python will then search the file specified using FILE_NAME
and attempt to cythonize it
'''

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

FILE_NAME = "simulation_physic"

ext_modules = [
        Extension(FILE_NAME,
                  sources=[FILE_NAME + ".pyx"],
                  libraries=["m"],
                  extra_compile_args=['-fopenmp'],
                  extra_link_args=['-fopenmp'],
                  include_dirs=[np.get_include()]
                  )
        ]

setup(
    ext_modules=cythonize(ext_modules, annotate=True)
)
