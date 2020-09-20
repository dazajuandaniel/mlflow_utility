# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name='mlflow_utility',
    version='0.1.0',
    description='Utility package for Mlflow',
    #long_description=readme,
    author='People',
    author_email='mail@people.com',
    url='https://github.com/dazajuandaniel/mlflow_utility',
    #license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)