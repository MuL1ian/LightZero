# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import re
import sys
from distutils.core import setup

import numpy as np
from Cython.Build import cythonize
from setuptools import find_packages, Extension

here = os.path.abspath(os.path.dirname(__file__))

from distutils.sysconfig import get_python_inc

python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
python_include_dir = get_python_inc()
include_dirs = [np.get_include(), python_include_dir]
if sys.platform == 'darwin':
    homebrew_python_path = f'/usr/local/opt/python@{python_version}/Frameworks/Python.framework/Versions/{python_version}/include/python{python_version}'
    if os.path.exists(homebrew_python_path):
        include_dirs.append(homebrew_python_path)
print(f"Python version: {python_version}")
print(f"Include directories: {include_dirs}")


def _load_req(file: str):
    with open(file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

requirements = _load_req('requirements.txt')

_REQ_PATTERN = re.compile('^requirements-([a-zA-Z0-9_]+)\\.txt$')
group_requirements = {
    item.group(1): _load_req(item.group(0))
    for item in [_REQ_PATTERN.fullmatch(reqpath) for reqpath in os.listdir()] if item
}

# Set C++11 compile parameters according to the operating system
extra_compile_args = []
extra_link_args = []

if sys.platform == 'win32':
    # Use the VS compiler on Windows platform
    extra_compile_args = ["/std:c++11"]
    extra_link_args = ["/std:c++11"]
else:
    # Linux/macOS Platform
    extra_compile_args = ["-std=c++11"]
    extra_link_args = ["-std=c++11"]


def find_pyx(path=None):
    path = path or os.path.join(here, 'lzero')
    pyx_files = []
    for root, dirs, filenames in os.walk(path):
        for fname in filenames:
            if fname.endswith('.pyx'):
                pyx_files.append(os.path.join(root, fname))

    path = os.path.join(here, 'zoo')
    for root, dirs, filenames in os.walk(path):
        for fname in filenames:
            if fname.endswith('.pyx'):
                pyx_files.append(os.path.join(root, fname))

    return pyx_files


def find_cython_extensions(path=None):
    extensions = []
    for item in find_pyx(path):
        relpath = os.path.relpath(os.path.abspath(item), start=here)
        rpath, _ = os.path.splitext(relpath)
        extname = '.'.join(rpath.split(os.path.sep))
        extensions.append(Extension(
            extname, [item],
            # include_dirs=[np.get_include()],
            include_dirs=include_dirs,
            language="c++",
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        ))

    return extensions


_LINETRACE = not not os.environ.get('LINETRACE', None)

setup(
    name='LightZero',
    version='0.2.0',
    description='A lightweight and efficient MCTS/AlphaZero/MuZero algorithm toolkits.',
    long_description_content_type='text/markdown',
    author='opendilab',
    author_email='opendilab@pjlab.org.cn',
    url='https://github.com/opendilab/LightZero',
    license='Apache License, Version 2.0',
    keywords='Reinforcement Learning, MCTS, MuZero',
    packages=[
        # framework
        *find_packages(include=('lzero', "lzero.*")),
        # application zoo
        *find_packages(include=('zoo', 'zoo.*')),
    ],
    package_data={
        package_name: ['*.yaml']
        for package_name in find_packages(include=('lzero.*',))
    },
    python_requires=">=3.7",
    install_requires=requirements,
    tests_require=group_requirements['test'],
    extras_require=group_requirements,
    ext_modules=cythonize(
        find_cython_extensions(),
        language_level=3,
        compiler_directives=dict(
            linetrace=_LINETRACE,
        ),
    ),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        "Intended Audience :: Science/Research",
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        # 'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
