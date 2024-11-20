from setuptools import setup
from src.aibox import __version__
setup(
      name='aibox',
      packages=['aibox'],
      package_dir={'': 'src'},
      version=__version__[1:]
      )