"""Setup file for the data pipeline package."""

from setuptools import setup, find_packages

setup(
    name="data_pipeline",
    version="0.1.0",
    description="Data pipeline for processing and managing data",
    author="",
    author_email="",
    packages=find_packages(),
    install_requires=[
        "boto3>=1.26.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "torch>=2.0.0",
        "mne>=1.4.0",
    ],
    entry_points={
        "console_scripts": [
            "data-pipeline=data_pipeline.cli:main",
        ],
    },
)