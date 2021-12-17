from setuptools import setup, find_packages
import os


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

configures = package_files('dkit/configures')
with open('README.md', encoding='utf-8') as f:
    readme = f.read()

with open('LICENSE', encoding='utf-8') as f:
    license = f.read()

# with open('requirements.txt', encoding='utf-8') as f:
    # reqs = f.read()

pkgs = [p for p in find_packages() if p.startswith('dkit')]

setup(
    name='dkit',
    version='0.1.0',
    # url='https://github.com/cstsunfu/dkit',
    description='dkit: Deep Learning Toolkit',
    # long_description=readme,
    long_description_content_type='text/markdown',
    # license='Apache License',
    author='cstsunfu',
    python_requires='>=3.7',
    package_data={"": configures},
    include_package_data=True,
    packages=pkgs,
    # install_requires=reqs.strip().split('\n'),
)
