#!/usr/bin/env python
""" Setup script """

from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

DEBUG = False

setup(
    name="farms_amphibious",
    version="0.1",
    author="farmsdev",
    author_email="jonathan.arreguitoneill@epfl.ch",
    description="FARMS package for amphibious simulations",
    # license="BSD-3",
    keywords="farms amphibious control simulation",
    # url="",
    # packages=["farms_amphibious"],
    packages=find_packages(),
    # long_description=read("README"),
    # classifiers=[
    #     "Development Status :: 3 - Alpha",
    #     "Topic :: Utilities",
    #     "License :: OSI Approved :: BSD License",
    # ],
    scripts=[],
    # package_data={"farms_amphibious": [
    #     "farms_amphibious/templates/*",
    #     "farms_amphibious/config/*"
    # ]},
    include_package_data=True,
    include_dirs=[np.get_include()],
    ext_modules=cythonize(
        [
            Extension(
                "farms_amphibious.{}*".format(folder.replace("/", "_") + "." if folder else ""),
                sources=["farms_amphibious/{}*.pyx".format(folder + "/" if folder else "")],
                extra_compile_args=["-O3"],  # , "-fopenmp"
                extra_link_args=["-O3"]  # , "-fopenmp"
            )
            for folder in [
                "data",
                "controllers",
            ]
        ],
        include_path=[np.get_include()],
        compiler_directives={
            "embedsignature": True,
            "cdivision": True,
            "language_level": 3,
            "infer_types": True,
            "profile": True,
            "wraparound": False,
            "boundscheck": DEBUG,
            "nonecheck": DEBUG,
            "initializedcheck": DEBUG,
            "overflowcheck": DEBUG,
        }
    ),
    zip_safe=False,
    # install_requires=[
    #     "cython",
    #     "numpy",
    #     "trimesh",
    #     "pyamphibious"
    # ],
)
