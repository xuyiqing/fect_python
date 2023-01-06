#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 10:00:58 2023

@author: liushijian
"""
from setuptools import setup, find_packages
import os, re, sys


VERSION = '0.1.0'
install_requires = [
    'numpy>=1.17',
      'pandas>=1.1.2',
      'rpy2<=3.5.4'
    ]

# with open("README.md", "r", encoding = "utf-8") as fh:
#     long_description = fh.read()

here = os.path.abspath(os.path.dirname(__file__))
_version = {}
_version_path = os.path.join(here, 'fect_py', '__version__.py')
with open(_version_path, 'r', 'utf-8') as f:
    exec(f.read(), _version)

setup(
  name = 'fect_py',         
  packages = ['fect_py'],   
  version = VERSION,      
  license='MIT',        
  description = 'counterfactural estimation and general synthetic control (FEct)',   
  author = 'Yiqing Xu, Shijian Liu',                   
  author_email = 'lshijian405@gmail.com',      
  url = 'https://github.com/xuyiqing/fect_python',   
  download_url = 'https://github.com/xuyiqing/fect_python/archive/refs/heads/main.zip',
  keywords = ['Fect', 'Counterfactual Estimation','Generalized Synthetic Control'],   
  install_requires=install_requires,
  classifiers=[
    'Development Status :: 3 - Alpha',      
    'Intended Audience :: Developers',      
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',   
    'Programming Language :: Python :: 3',      
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
  ],
)
