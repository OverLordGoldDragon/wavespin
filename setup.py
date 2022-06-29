# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------

"""
WaveSpin
========

Joint Time-Frequency Scattering, Wavelet Scattering: features for audio,
biomedical, and other applications, in Python

WaveSpin features scattering transform implementations that maximize accuracy,
flexibility, and speed. Included are visualizations and convenience utilities
for transform and coefficient introspection and debugging.
"""

import os
import re
from setuptools import setup, find_packages

current_path = os.path.abspath(os.path.dirname(__file__))


def read_file(*parts):
    with open(os.path.join(current_path, *parts), encoding='utf-8') as reader:
        return reader.read()


def get_requirements(*parts):
    with open(os.path.join(current_path, *parts), encoding='utf-8') as reader:
        return list(map(lambda x: x.strip(), reader.readlines()))


def find_version(*file_paths):
    version_file = read_file(*file_paths)
    version_matched = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                                version_file, re.M)
    if version_matched:
        return version_matched.group(1)
    raise RuntimeError('Unable to find version')


setup(
    name="WaveSpin",
    version=find_version('wavespin', '__init__.py'),
    packages=find_packages(exclude=['tests', 'examples']),
    url="https://github.com/OverLordGoldDragon/wavespin",
    license="MIT",
    author="John Muradeli",
    author_email="john.muradeli@gmail.com",
    description=("Joint Time-Frequency Scattering, Wavelet Scattering: features "
                 "for audio, biomedical, and other applications, in Python"),
    long_description=read_file('README.md'),
    long_description_content_type="text/markdown",
    keywords=(
        "scattering-transform wavelets signal-processing visualization "
        "pytorch tensorflow python"
    ),
    install_requires=get_requirements('requirements.txt'),
    tests_require=["pytest>=4.0", "pytest-cov"],
    include_package_data=True,
    zip_safe=True,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Topic :: Utilities",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
