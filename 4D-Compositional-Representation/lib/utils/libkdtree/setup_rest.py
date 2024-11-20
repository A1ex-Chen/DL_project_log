#pykdtree, Fast kd-tree implementation with OpenMP-enabled queries
#
#Copyright (C) 2013 - present  Esben S. Nielsen
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Get OpenMP setting from environment
try:
    use_omp = int(os.environ['USE_OMP'])
except KeyError:
    use_omp = True




# Custom builder to handler compiler flags. Edit if needed.
class build_ext_subclass(build_ext):



setup(
    name='pykdtree',
    version='1.3.1',
    description='Fast kd-tree implementation with OpenMP-enabled queries',
    author='Esben S. Nielsen',
    author_email='storpipfugl@gmail.com',
    packages = ['pykdtree'],
    python_requires='>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*',
    install_requires=['numpy'],
    setup_requires=['numpy'],
    tests_require=['nose'],
    zip_safe=False,
    test_suite = 'nose.collector',
    ext_modules = [Extension('pykdtree.kdtree',
                             ['pykdtree/kdtree.c', 'pykdtree/_kdtree_core.c'])],
    cmdclass = {'build_ext': build_ext_subclass },
    classifiers=[
      'Development Status :: 5 - Production/Stable',
      ('License :: OSI Approved :: '
          'GNU Lesser General Public License v3 (LGPLv3)'),
      'Programming Language :: Python',
      'Operating System :: OS Independent',
      'Intended Audience :: Science/Research',
      'Topic :: Scientific/Engineering'
      ]
    )