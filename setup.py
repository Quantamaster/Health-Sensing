"""
DeepMedico™ Sleep Breathing Irregularity Detection System
Setup script for package installation
"""

from setuptools import setup, find_packages

setup(
    name="deepmedico-sleep-detection",
    version="1.0.0",
    description="Sleep breathing irregularity detection using deep learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="DeepMedico Team",
    author_email="contact@deepmedico.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "tensorflow>=2.8.0",
        "pyarrow>=5.0.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    entry_points={
        "console_scripts": [
            "deepmedico-vis=vis:main",
            "deepmedico-dataset=create_dataset:main",
            "deepmedico-model=modeling:main",
            "deepmedico-sleep-stages=sleep_stage_classification:main",
        ],
    },
)
