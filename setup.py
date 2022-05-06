# Copyright 2021 cstsunfu. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, find_packages
from dlk import __version__
import os


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

configures = package_files('dlk/configures')
with open('README.md', encoding='utf-8') as f:
    readme = f.read()

with open('LICENSE', encoding='utf-8') as f:
    license = f.read()

with open('requirements.txt', encoding='utf-8') as f:
    requirements = f.read()

pkgs = [p for p in find_packages() if p.startswith('dlk')]

setup(
    name='dlk',
    version=__version__,
    url='https://github.com/cstsunfu/dlk',
    description='dlk: Deep Learning Kit',
    long_description=readme,
    long_description_content_type='text/markdown',
    license="Apache Software License",
    author='cstsunfu',
    author_email='cstsunfu@gmail.com',
    python_requires='>=3.7',
    package_data={"": configures},
    include_package_data=True,
    packages=pkgs,
    install_requires=requirements.strip().split('\n'),
)
