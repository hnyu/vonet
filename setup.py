from setuptools import setup, find_packages
import os

setup(
    name='vonet',
    version='0.0.1',
    python_requires='>=3.7.0',
    install_requires=[
        'alf@git+https://github.com/HorizonRobotics/alf.git@pytorch#egg=ALF',
        'scikit-image',
        'moviepy',
        'pillow==9.4.0',
        'imageio==2.27.0'
    ],
    packages=find_packages(),
)