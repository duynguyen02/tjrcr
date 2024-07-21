from setuptools import find_packages, setup
from codecs import open
from os import path

from tjrcr import __version__, __author__

HERE = path.abspath(path.dirname(__file__))
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

PACKAGES = [
    'tjrcr'
]

REQUIRED_PACKAGES = [
    "numpy",
    "pandas",
    "tjwb"
]

setup(
    name='tjrcr',
    packages=find_packages(include=PACKAGES),
    version=__version__,
    description='Python library used to calculate reservoir comprehensive regulation.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=__author__,
    url="https://github.com/duynguyen02/tjrcr",
    install_requires=REQUIRED_PACKAGES,
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
    ]
)
