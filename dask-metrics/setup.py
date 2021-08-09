from setuptools import setup, find_packages

install_requires = [
    'setuptools',
    'dask',
    'distributed',
    'cudf',
    'pandas',
    'nest-asyncio'
]

setup (
    name='dask-metrics',
    version='2021.8.2',
    description='A tool for collecting metrics on distributed Dask clusters',
    author='Travis Hester',
    packages=find_packages(include=['dask_metrics']),
    install_requires=install_requires
)