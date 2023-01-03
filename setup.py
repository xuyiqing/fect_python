#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 10:00:58 2023

@author: liushijian
"""
from distutils.core import setup
setup(
  name = 'fect_py',         # How you named your package folder (foo)
  packages = ['fect_py'],   # Chose the same as "name"
  version = '0.1',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'counterfactural estimation and general synthetic control (FEct)',   # Give a short description about your library
  author = 'Yiqing Xu, Shijian Liu',                   # Type in your name
  author_email = 'lshijian405@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/Yiguan/Cookie2Dict/',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/Yiguan/Cookie2Dict/archive/master.zip',
  keywords = ['Cookie', 'Dictionary'],   # Keywords that define your package best
  install_requires=[],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
