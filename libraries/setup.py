# This file is part of SIGWfast.

# SIGWfast is free software: you can use, copy, modify, merge, publish and
# distribute, sublicense, and/or sell copies of it, and to permit persons to
# whom it is furnished to do so it under the terms of the MIT License.

# SIGWfast is distributed in the hope that it will be useful,
# but WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT 
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
# PURPOSE AND NONINFRINGEMENT. See the MIT License for more details.

# You should have received a copy of the MIT License along with SIGWfast. If
# not, see <https://spdx.org/licenses/MIT.html>.

from distutils.core import setup, Extension
import numpy, platform, sys

# Collect instructions for compilation of extension.
# Check whether the operating system is MacOS.
if platform.system() == 'Darwin':
    # Set compiler and linker flag -stdlib=libc++ explicitly to build against
    # libc++ as libstdc++ is no longer supplied by MacOS / Xcode. With this
    # compilation should work on both current and older MacOS systems.
    sigwfast_module = Extension('sigwfast',sources = ['SIGWfast.cpp'], 
                                extra_compile_args=["-stdlib=libc++"], 
                                extra_link_args=["-stdlib=libc++"], 
                                include_dirs=[numpy.get_include()])
# Check whether the operating system is Linux.
elif platform.system() == 'Linux':
    sigwfast_module = Extension('sigwfast',sources = ['SIGWfast.cpp'], 
                                include_dirs=[numpy.get_include()])
else:
    print('C++ extension not supported by your system.')
    sys.exit()
    
# Provide details on the extension.    
setup(name = 'SIGWfastExtension', version='1.0',
      description =  'This is a package to perform the integrations in the '
                    +'computation of the scalar-induced gravitational wave '
                    +'spectrum.', 
      url = 'https://github.com/Lukas-T-W',
      author = 'Lukas T. Witkowski',
      author_email = 'lukas.witkowski@iap.fr',
      license = 'MIT License',
      platforms = ['MacOS', 'Linux'],
      long_description =  'SIGWfast is a python code to compute the Scalar-'
                         +'Induced Gravitational Wave spectrum from a '
                         +'primordial scalar power spectrum that can be given '
                         +'in analytical or numerical form. This extension '
                         +'performs the two-dimensional integrations needed '
                         +'in this computation using compiled C++ code. The '
                         +'code SIGWfast, including this extension, is '
                         +'distributed under the MIT license, a copy of which '
                         +'should be included with the code. If not, see '
                         +'https://spdx.org/licenses/MIT.html.',
      ext_modules = [sigwfast_module])