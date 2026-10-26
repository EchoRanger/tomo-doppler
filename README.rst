trm.doppler is a Python implementation of Doppler tomography.

Pre-requisites for installation:

  fftw    -- for taking FFTs, standard package.
  a C++ compiler with OpenMP.
  Python 3.14 or later.

cpp-mem, the C++ translation of memsys, is a git submodule and is
compiled into the extension, so clone with

git clone --recursive https://github.com/EchoRanger/tomo-doppler.git

or run 'git submodule update --init' in an existing clone. Then

pip install .

in the directory with setup.py. pip installs the Python dependencies
(numpy, scipy, astropy, matplotlib).

Some scripts need other packages. e.g. The volume rendering script
vrend.py needs 's2plot' (but I intend to explore other options for
3D visualiation so don't reply on this).
