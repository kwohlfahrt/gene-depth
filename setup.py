#!/usr/bin/env python3

from setuptools import setup

setup(name="gene_depth",
      version="0.0.2",
      description="A set of binaries for running depth analysis on microscope images",
      packages=['gene_depth'],
      requires=['numpy (>=1.10)', 'scipy', 'skimage', 'pandas'],
      entry_points={'console_scripts': ['gene-depth=gene_depth.main:main']}
)
