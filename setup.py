#!/usr/bin/env python
""" Setup script """

from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools import dist

dist.Distribution().fetch_build_eggs(['numpy'])
import numpy as np  # pylint: disable=wrong-import-position

dist.Distribution().fetch_build_eggs(['Cython>=0.15.1'])
from Cython.Build import cythonize  # pylint: disable=wrong-import-position
from Cython.Compiler import Options  # pylint: disable=wrong-import-position

dist.Distribution().fetch_build_eggs(['farms_core'])
from farms_core import get_include_paths  # pylint: disable=wrong-import-position


# Cython options
DEBUG = False
Options.docstrings = True
Options.embed_pos_in_docstring = False
Options.generate_cleanup_code = False
Options.clear_to_none = True
Options.annotate = False
Options.fast_fail = False
Options.warning_errors = False
Options.error_on_unknown_names = True
Options.error_on_uninitialized = True
Options.convert_range = True
Options.cache_builtins = True
Options.gcc_branch_hints = True
Options.lookup_module_cpdef = False
Options.embed = None
Options.cimport_from_pyx = False
Options.buffer_max_dims = 8
Options.closure_freelist_size = 8


setup(
    name='farms_sim',
    version='0.1',
    author='farmsdev',
    author_email='biorob-farms@groupes.epfl.ch',
    description='FARMS package for sim simulations',
    # license='BSD-3',
    keywords='farms control simulation',
    # url='',
    # packages=['farms_sim'],
    packages=find_packages(),
    # long_description=read('README'),
    # classifiers=[
    #     'Development Status :: 3 - Alpha',
    #     'Topic :: Utilities',
    #     'License :: OSI Approved :: BSD License',
    # ],
    scripts=[],
    # package_data={'farms_sim': [
    #     'farms_sim/templates/*',
    #     'farms_sim/config/*'
    # ]},
    include_package_data=True,
    include_dirs=[np.get_include()],
    ext_modules=cythonize(
        [
            Extension(
                f'farms_sim.{folder}.*',
                sources=[f'farms_sim/{folder}/*.pyx'],
                extra_compile_args=['-O3'],  # , '-fopenmp'
                extra_link_args=['-O3']  # , '-fopenmp'
            )
            for folder in []
        ],
        include_path=[np.get_include()] + get_include_paths(),
        compiler_directives={
            # Directives
            'binding': False,
            'embedsignature': True,
            'cdivision': True,
            'language_level': 3,
            'infer_types': True,
            'profile': True,
            'wraparound': False,
            'boundscheck': DEBUG,
            'nonecheck': DEBUG,
            'initializedcheck': DEBUG,
            'overflowcheck': DEBUG,
            'overflowcheck.fold': DEBUG,
            'cdivision_warnings': DEBUG,
            'always_allow_keywords': DEBUG,
            'linetrace': DEBUG,
            # Optimisations
            'optimize.use_switch': True,
            'optimize.unpack_method_calls': True,
            # Warnings
            'warn.undeclared': True,
            'warn.unreachable': True,
            'warn.maybe_uninitialized': True,
            'warn.unused': True,
            'warn.unused_arg': True,
            'warn.unused_result': True,
            'warn.multiple_declarators': True,
        }
    ),
    zip_safe=False,
    install_requires=[
        'farms_core',
        'farms_mujoco',
        'cython',
        'numpy',
        'trimesh',
    ],
)
