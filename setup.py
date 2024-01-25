from setuptools import setup, find_packages
import os

setup(
    name='vonet',
    version='0.0.1',
    python_requires='>=3.7.0',
    install_requires=[
        'alf@git+https://github.com/HorizonRobotics/alf.git#9d4898954de9eea6aebb4c5ad8ebbd19eaa5d8f1',
        'scikit-image',
        'moviepy'
    ],
    packages=find_packages(),
)