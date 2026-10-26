import os, sys
from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext
import numpy

# get round irritating compiler warning
#class BuildExt(build_ext):
#    def build_extensions(self):
#        self.compiler.compiler_so.remove('-Wstrict-prototypes')
#        super().build_extensions()

if not os.path.exists(os.path.join('cpp-mem', 'src', 'memsys.cc')):
    sys.exit("cpp-mem submodule missing; run 'git submodule update --init'")

include_dirs = [os.path.join('cpp-mem', 'include'), numpy.get_include()]

# cpp-mem (git submodule) is a single file, so compile it in
# rather than depend on an installed libmem
doppler = Extension(
    'trm.doppler._doppler',
    define_macros = [('MAJOR_VERSION', '1'),('MINOR_VERSION', '0')],
    undef_macros = ['USE_NUMARRAY'],
    include_dirs = include_dirs,
    libraries = ['fftw3'],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],
    sources = [os.path.join('trm', 'doppler', 'doppler.cc'),
               os.path.join('cpp-mem', 'src', 'memsys.cc')]
)

setup(name='trm.doppler',
      version='1.0',
      packages = ['trm', 'trm.doppler', 'trm.doppler.scripts'],
      python_requires = '>=3.14',
      install_requires = ['numpy', 'scipy', 'astropy', 'matplotlib'],
      ext_modules=[doppler,],
      entry_points={
          'console_scripts' : [
              'comdat=trm.doppler.scripts.comdat:comdat',
              'comdef=trm.doppler.scripts.comdef:comdef',
              'drlimit=trm.doppler.scripts.drlimit:drlimit',
              'makedata=trm.doppler.scripts.makedata:makedata',
              'makegrid=trm.doppler.scripts.makegrid:makegrid',
              'makemap=trm.doppler.scripts.makemap:makemap',
              'memit=trm.doppler.scripts.memit:memit',
              'mol2dopp=trm.doppler.scripts.mol2dopp:mol2dopp',
              'mspruit=trm.doppler.scripts.mspruit:mspruit',
              'optgam=trm.doppler.scripts.optgam:optgam',
              'optscl=trm.doppler.scripts.optscl:optscl',
              'precover=trm.doppler.scripts.precover:precover',
              'psearch=trm.doppler.scripts.precover:psearch',
              'svdfit=trm.doppler.scripts.svdfit:svdfit',
              'svd=trm.doppler.scripts.svdfit:svd',
              'trtest=trm.doppler.scripts.trtest:trtest',
              'vrend=trm.doppler.scripts.vrend:vrend',
          ],
      },

      # metadata
      author='Tom Marsh',
      author_email='t.r.marsh@warwick.ac.uk',
      description="Python Doppler tomography module",
      url='http://www.astro.warwick.ac.uk/',
      long_description="""
doppler is an implementation of Doppler tomography package. It has many advantages over the
former set of F77 routines.
""",
#      cmdclass={'build_ext': BuildExt}

      )

