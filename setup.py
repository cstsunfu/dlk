# Copyright the author(s) of DLK.
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
if ("CUDA_HOME" in os.environ) and int(os.environ.get("BUILD_CUDA", "0")) == 1:
    try:
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
    except:
        pass

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
    python_requires=">=3.7",
    include_package_data=True,
    packages=pkgs,
    install_requires=[
        "intc>=0.1.0,<0.2",
        "hjson>=3.0.2",
        "pandas>=1.4",
        "pyarrow>=10.0.1",
        "fsspec>=2023.5,<2024.2",
        "torchmetrics>=1.2.1",
        "transformers>=4.24.0,<4.45",
    ],
    extras_require={
        "train": [
            "lightning>=2.1.0,<=2.2",
            "torch>=2.1,<=2.6",
            "tensorboard>=2.7.0,<3.0",
            "datasets>=2.17.0,<2.20",
            "torchmetrics>=1.2.1",
        ],
        "export": ["torch>=1.8.1,<2.5"],
        "process": [
            "datasets>=2.17.0,<2.20",
        ],
        "all": [
            "lightning>=2.1.0,<=2.2",
            "torch>=2.1,<=2.6",
            "tensorboard>=2.7.0,<3.0",
            "datasets>=2.17.0,<2.20",
            "streamlit>=1.32",
            "svgwrite>=1.4",
            "pyecharts>=2.0",
        ],
    },
)
