#!/usr/bin/env python
"""Setup environment for nmf."""

from setuptools import setup, find_packages

exec(open("kde_gpu/__version__.py").read())
readme = open("README.md").read()

setup(
    name="kde_gpu",
    version=__version__,
    description="We implemented nadaraya waston kernel density and kernel conditional probability estimator using cuda through cupy. It is much faster than cpu version but it requires GPU with high memory.",
    long_description=readme,
    author="Chen Chen",
    author_email="chen.chen.adl@gmail.com",
    url="https://github.com/JoyceXinyueWang/nmf",
    packages=find_packages(),
    package_dir={"kde_gpu": "kde_gpu"},
    include_package_data=True,   
    install_requires=[
        "numpy>=1.14.0",
        "scipy>=1.0.0",
        "cupy>=6.2.0",
        "pandas>=0.20.2",
    ],
    license="MIT License",
    zip_safe=False,
    keywords="Nadaraya-Watson, Nadaraya Watson, GPU, CUDA, cupy, kernel smoothing, conditional probability, KNN",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Operating System :: POSIX",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
