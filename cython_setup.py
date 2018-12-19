from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
'''
ext_modules = [
        Extension(FILE_NAME,
        	      sources=[FILE_NAME + ".pyx"],
        	      libraries=["m"],
        	      extra_compile_args=['-fopenmp'],
        	      extra_link_args=['-fopenmp']
        	      )
        ]
'''
setup(
	include_dirs = [np.get_include()], 
    ext_modules=cythonize("simulation_physic.pyx", annotate=True)
)
