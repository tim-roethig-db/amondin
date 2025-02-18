"""
This module sets up the package 'amondin' using setuptools.
"""

from setuptools import setup, find_packages


setup(
    name="amondin",
    version="0.0.1",
    author="Tim Röthig",
    python_requires=">=3.11",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "openpyxl",
        "pandas",
        "pyannote.audio",
        "pyannote.core",
        "pyyaml",
        "torch",
        "torchaudio",
        "transformers",
        "black",
        "pylint",
        "mypy",
        "black",
        "pytest",
        "hypothesis",
    ],
)
