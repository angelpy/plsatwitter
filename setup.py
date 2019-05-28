# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


setup(name='plsatwitter',
      version='0.1',
      author='Ángel Pérez',
      author_email='alperezmi@hotmail.com',
      packages=find_packages(),
      install_requires=[
          'nimfa',
          'nonnegfac',  
          'numpy',
          'sklearn'
      ],
      zip_safe=False)
