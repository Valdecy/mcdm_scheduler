from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / 'README.md').read_text()

setup(
    name='mcdm_scheduler',
    version='1.5.3',
    license='GNU',
    author='Valdecy Pereira',
    author_email='valdecy.pereira@gmail.com',
    url='https://github.com/Valdecy/mcdm_scheduler',
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'numpy'
    ],
    description='A Library Incorporating a MCDM tools for Scheduling Problems',
    long_description=long_description,
    long_description_content_type='text/markdown',
)
