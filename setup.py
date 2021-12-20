from setuptools import setup, find_packages
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
    version='0.0.2',
    url='https://github.com/cstsunfu/dlk',
    description='dlk: Deep Learning Kit',
    long_description=readme,
    long_description_content_type='text/markdown',
    license=license,
    author='cstsunfu',
    author_email='cstsunfu@gmail.com',
    python_requires='>=3.8',
    package_data={"": configures},
    include_package_data=True,
    packages=pkgs,
    install_requires=requirements.strip().split('\n'),
)
