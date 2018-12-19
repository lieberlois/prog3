from distutils.core import setup
from Cython.Build import cythonize
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
    ext_modules=cythonize("simulation_physic.pyx", annotate=True)
)
