#!/usr/bin/env python
from setuptools import find_packages, setup





    packages = list(gen_packages_items())
    return packages


if __name__ == '__main__':
    setup(
        name='visdrone_eval',
        version='0.1',
        description='Python Implementation of VisDrone Detection Toolbox',
        packages=find_packages(exclude=('configs', 'tools', 'demo')),
        install_requires=parse_requirements('requirements.txt')
    )