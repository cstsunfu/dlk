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


cmdclass = {}
extensions = []
if "CUDA_HOME" in os.environ or "CUDA_PATH" in os.environ:
    from torch.utils import cpp_extension

    extensions = [
        cpp_extension.CppExtension(
            "dlk.ngram_repeat_block_cuda",
            sources=[
                "dlk/cuda/ngram_repeat_block_cuda.cpp",
                "dlk/cuda/ngram_repeat_block_cuda_kernel.cu",
            ],
        ),
    ]
    cmdclass = {"build_ext": cpp_extension.BuildExtension}

if "READTHEDOCS" in os.environ:
    # don't build extensions when generating docs
    extensions = []
    if "build_ext" in cmdclass:
        del cmdclass["build_ext"]


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
    url="https://github.com/cstsunfu/dlk",
    description="dlk: Deep Learning Kit",
    long_description_content_type="text/markdown",
    version=version,
    ext_modules=extensions,
    cmdclass=cmdclass,
    package_data={"": added_files},
    license=license,
    include_package_data=True,
    packages=pkgs,
    install_requires=requirements.strip().split("\n"),
)
