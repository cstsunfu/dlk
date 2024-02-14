# Copyright cstsunfu.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import os

from setuptools import find_packages, setup


def write_version_py():
    with open(os.path.join("dlk", "version.txt")) as f:
        version = f.read().strip()

    # write version info to fairseq/version.py
    with open(os.path.join("dlk", "version.py"), "w") as f:
        f.write(f'__version__ = "{version}"\n')
    return version


version = write_version_py()


def package_files(directory):
    paths = []
    for path, directories, filenames in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join(".", path, filename))
    return paths


added_files = []
added_files.extend(
    package_files(os.path.join("dlk", "utils", "display", "label_colors"))
)
added_files.extend(package_files(os.path.join("dlk", "utils", "display", "fonts")))
with open("README.md", encoding="utf-8") as f:
    readme = f.read()

with open("LICENSE", encoding="utf-8") as f:
    license = f.read()

with open("requirements.txt", encoding="utf-8") as f:
    requirements = f.read()

pkgs = [p for p in find_packages() if p.startswith("dlk")]

setup(
    name="dlk",
    version=version,
    url="https://github.com/cstsunfu/dlk",
    description="dlk: Deep Learning Kit",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="Apache Software License",
    author="cstsunfu",
    author_email="cstsunfu@gmail.com",
    python_requires=">=3.7",
    package_data={"": added_files},
    include_package_data=True,
    packages=pkgs,
    install_requires=requirements.strip().split("\n"),
)
