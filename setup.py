# setup.py
from setuptools import setup, find_packages

setup(
    name="electric-load-forecast",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'torch',
        'pytest',
    ],
)